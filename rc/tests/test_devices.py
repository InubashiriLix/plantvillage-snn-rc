import numpy as np
import pytest
import torch

from plantvillage_rc.v4_experiments import TorchRidgeClassifier
from plantvillage_rc.v5_experiments import train_linear_softmax


def test_explicit_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x, y = np.ones((4, 3), dtype=np.float32), np.array([0, 1, 0, 1])
    with pytest.raises(RuntimeError, match="--device cpu"):
        TorchRidgeClassifier(1, device="cuda").fit(x, y)
    with pytest.raises(RuntimeError, match="--device cpu"):
        train_linear_softmax(x, y, class_count=2, weight_decay=0, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA driver/device unavailable")
def test_cuda_readout_training():
    x = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)
    y = np.array([0, 1] * 4)
    ridge = TorchRidgeClassifier(1, device="cuda").fit(x, y)
    assert ridge.predict(x).shape == (8,)
    model, _ = train_linear_softmax(x, y, class_count=2, weight_decay=0, device="cuda", fixed_epochs=1)
    assert model.predict(x).shape == (8,)
