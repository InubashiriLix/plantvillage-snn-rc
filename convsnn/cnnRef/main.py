"""
Hardware-aware CNN training on CIFAR-10 using memristive device model.

Pipeline (from README):
  1. Device measurement -> LTP/LTD curves
  2. Build LUT: (G, delta_G) -> #pulses
  3. Forward pass with actual G (mapped to w_hw)
  4. Backprop -> gradients
  5. Ideal weight update: dw = -lr * grad  (SGD)
  6. Convert dw -> dG via per-layer scaling
  7. Accumulate dG; apply pulses when |accum| exceeds threshold
  8. Apply pulses via device model -> actual G
  9. Map actual G back to w_hw
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
from device import DeviceModel


# ---------------------------------------------------------------------------
# CNN for CIFAR-10 (3x32x32 -> 10)
# ---------------------------------------------------------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(32 * 4 * 4, 64)
        self.fc2   = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# HWState: stores conductance G per layer, with gradient accumulation
# ---------------------------------------------------------------------------
class HWState:
    def __init__(self, model, device_model):
        self.dev = device_model
        self.G_ref = device_model.G_mid
        self.G_half = (device_model.G_max - device_model.G_min) / 2.0

        self.G = {}
        self.G_accum = {}
        self.scales = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # BN running stats (1D) treated as ideal
            if "bn" in name and len(param.shape) == 1:
                self.G[name] = param.detach().cpu().numpy().astype(np.float32)
                self.G_accum[name] = np.zeros_like(self.G[name], dtype=np.float64)
                self.scales[name] = 1.0
                continue

            w = param.detach().cpu().numpy()
            w_abs_max = max(np.abs(w).max(), 1e-6)
            scale = w_abs_max * 1.5
            self.scales[name] = scale

            w_norm = np.clip(w / scale, -0.5, 0.5)
            G_init = self.G_ref + w_norm * self.G_half
            G_init = np.clip(G_init, device_model.G_min, device_model.G_max)
            self.G[name] = G_init.astype(np.float32)
            self.G_accum[name] = np.zeros_like(G_init, dtype=np.float64)

    def get_tensor(self, name):
        G_np = self.G[name]
        if self.scales[name] == 1.0:
            return torch.tensor(G_np, dtype=torch.float32)
        w_norm = (G_np - self.G_ref) / self.G_half
        return torch.tensor(w_norm * self.scales[name], dtype=torch.float32)

    def accumulate_and_update(self, name, grad_np, lr, max_pulses_per_update=8):
        scale = self.scales[name]
        delta_w = -lr * grad_np

        if scale == 1.0:
            self.G[name] = (self.G[name] + delta_w).astype(np.float32)
            self.G_accum[name] *= 0.0
            return 0, 0

        delta_G = (delta_w / scale).astype(np.float64) * self.G_half
        self.G_accum[name] += delta_G

        min_step = self.dev.G_span / self.dev.n_G_bins
        accum = self.G_accum[name]
        mask = np.abs(accum) >= min_step
        pulses_total = 0
        n_updated = mask.sum()

        if n_updated > 0:
            G_curr = self.G[name][mask]
            dG_apply = accum[mask]
            max_dG = self.dev.G_span * max_pulses_per_update / max(self.dev.effective_ltp_levels, 1)
            dG_apply = np.clip(dG_apply, -max_dG, max_dG)
            new_G, pulses = self.dev.apply_pulses_vectorized(G_curr, dG_apply)
            self.G[name][mask] = new_G.astype(np.float32)
            accum[mask] = dG_apply - (new_G - G_curr)
            pulses_total = int(pulses.sum())

        return pulses_total, n_updated


# ---------------------------------------------------------------------------
# HWTrainer
# ---------------------------------------------------------------------------
class HWTrainer:
    def __init__(self, model, device_model, lr=0.1, lr_decay=0.92):
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.hw = HWState(model, device_model)
        self.param_names = [n for n, p in model.named_parameters() if p.requires_grad]

    def decay_lr(self):
        self.lr *= self.lr_decay

    def _sync_hw_to_model(self):
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for name in self.param_names:
                params[name].copy_(self.hw.get_tensor(name))

    def train_epoch(self, loader, epoch, max_batches=None):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_pulses = 0
        total_updates = 0
        n_batches = 0

        for batch_idx, (data, target) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            n_batches += 1

            self._sync_hw_to_model()

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.model.zero_grad()
            loss.backward()

            if batch_idx % 10 == 0 and batch_idx > 0:
                for name in self.param_names:
                    self.hw.G_accum[name] *= 0.9

            params = dict(self.model.named_parameters())
            for name in self.param_names:
                p = params[name]
                if p.grad is not None:
                    pulses, n_upd = self.hw.accumulate_and_update(
                        name, p.grad.numpy(), self.lr)
                    total_pulses += pulses
                    total_updates += n_upd

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

            if batch_idx % 200 == 0:
                print(f"  Batch {batch_idx:4d}: loss={loss.item():.4f}, "
                      f"acc={100.*correct/total:.1f}%, "
                      f"pulses={total_pulses}, upd={total_updates}, lr={self.lr:.4f}")

        return total_loss / n_batches, 100.0 * correct / total, total_pulses, total_updates

    @torch.no_grad()
    def evaluate(self, loader, max_batches=None):
        self.model.eval()
        self._sync_hw_to_model()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            output = self.model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
        return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Ideal baseline training
# ---------------------------------------------------------------------------
def train_ideal_baseline(train_loader, test_loader, n_epochs=15):
    model = CIFAR10CNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for data, target in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        acc = 100.0 * correct / total
        print(f"    Ideal Epoch {epoch:2d}: Loss={train_loss/n_batches:.4f}, Test Acc={acc:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Hardware-Aware CNN Training on CIFAR-10")
    print("=" * 60)

    print("\n[1] Loading device model ...")
    dev = DeviceModel("source/data.csv")
    print(f"    G range: [{dev.G_min*1e9:.2f}, {dev.G_max*1e9:.2f}] nS")
    print(f"    Model: empirical dG(G) from data, {dev.n_G_bins} G-bins")

    print("\n[2] Loading CIFAR-10 ...")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("data/cifar10", train=True, download=True,
                                transform=train_transform)
    test_ds = datasets.CIFAR10("data/cifar10", train=False, download=True,
                               transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=500)

    print("\n[3] Building CNN ...")
    cnn = CIFAR10CNN()
    total_params = sum(p.numel() for p in cnn.parameters())
    weight_params = sum(p.numel() for n, p in cnn.named_parameters()
                        if p.requires_grad and len(p.shape) >= 2)
    print(f"    Conv(3,16)-Conv(16,32)-Conv(32,32)-Pool-FC(512,64)-FC(64,10)")
    print(f"    Total params: {total_params:,}, weight params (HW-mapped): {weight_params:,}")

    print("\n[4] Training IDEAL baseline (Adam, CosineLR) ...")
    ideal_acc = train_ideal_baseline(train_loader, test_loader, n_epochs=15)

    print(f"\n[5] Initializing HW trainer (lr=0.08, decay=0.93) ...")
    cnn_hw = CIFAR10CNN()
    trainer = HWTrainer(cnn_hw, dev, lr=0.08, lr_decay=0.93)

    print("\n[6] Training HARDWARE-AWARE ...")
    n_epochs = 15
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, pulses, upd = trainer.train_epoch(train_loader, epoch)
        t1 = time.time()
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\n  Epoch {epoch:2d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"  Pulses: {pulses:,} | Updates: {upd:,} | Time: {t1-t0:.1f}s")
        trainer.decay_lr()

    print("\n[7] Final evaluation ...")
    test_loss, hw_acc = trainer.evaluate(test_loader)
    print(f"    HW-aware Test Accuracy: {hw_acc:.2f}%")

    print(f"\n[8] Summary")
    print(f"    Ideal baseline (Adam, fp32):  {ideal_acc:.2f}%")
    print(f"    Hardware-aware (device model): {hw_acc:.2f}%")
    print(f"    Accuracy penalty:              {ideal_acc - hw_acc:.2f}%")

    print(f"\n[9] Conductance stats (nS):")
    for name in trainer.param_names:
        if trainer.hw.scales.get(name, 0) == 1.0:
            continue
        g = trainer.hw.G[name]
        ac = np.abs(trainer.hw.G_accum[name])
        print(f"    {name:20s}: mean={g.mean()*1e9:7.2f}, std={g.std()*1e9:7.2f}, "
              f"min={g.min()*1e9:7.2f}, max={g.max()*1e9:7.2f}, "
              f"|accum|_max={ac.max()*1e9:.2f}")

    return hw_acc


if __name__ == "__main__":
    main()
