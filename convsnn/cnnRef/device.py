"""
Memristive device model built from measured data (source/data.csv).

Extracts per-pulse dG vs G from all LTP/LTD segments, fits a smoothed
dG(G) function, numerically integrates to get full-range G(pulse) curves,
and builds a vectorized LUT.

Purely empirical — no parametric assumptions beyond smoothness.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class DeviceModel:
    def __init__(self, csv_path: str = "source/data.csv", V_bias: float = -5.0,
                 n_G_bins: int = 200, max_pulses: int = 200):
        self.V_bias = V_bias
        self.n_G_bins = n_G_bins
        self.max_pulses = max_pulses
        self._load_data(csv_path)
        self._extract_segments()
        self._build_dG_functions()
        self._build_curves()
        self._build_lut()

    # ------------------------------------------------------------------
    def _load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.I = df["Id"].values
        self.G_raw = np.abs(self.I / self.V_bias)
        self.G_min = self.G_raw.min()
        self.G_max = self.G_raw.max()
        self.G_span = self.G_max - self.G_min
        self.G_mid_val = (self.G_min + self.G_max) / 2.0
        print(f"    G range (nS): [{self.G_min*1e9:.2f}, {self.G_max*1e9:.2f}], "
              f"span={self.G_span*1e9:.2f}")

    # ------------------------------------------------------------------
    def _extract_segments(self):
        dG = np.diff(self.G_raw)
        signs = np.sign(dG)
        signs[signs == 0] = 1
        ltp_segs, ltd_segs = [], []
        start = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1] or i == len(signs) - 1:
                seg = self.G_raw[start:i + 1]
                if len(seg) >= 5:
                    if signs[i - 1] > 0:
                        ltp_segs.append(seg)
                    else:
                        ltd_segs.append(seg)
                start = i
        self.ltp_segments = ltp_segs
        self.ltd_segments = ltd_segs
        print(f"    LTP segments: {len(ltp_segs)}, LTD segments: {len(ltd_segs)}")

    # ------------------------------------------------------------------
    def _build_dG_functions(self):
        """
        Collect per-pulse (G, dG) from all segments, bin by G,
        build smooth dG(G) interpolators for LTP and LTD.
        """
        # LTP: collect all (G, dG) where dG > 0
        G_ltp_vals, dG_ltp_vals = [], []
        for seg in self.ltp_segments:
            dg = np.diff(seg)
            g = seg[:-1]
            mask = dg > 0
            G_ltp_vals.append(g[mask])
            dG_ltp_vals.append(dg[mask])

        # LTD: collect all (G, dG) where dG < 0 (store |dG|)
        G_ltd_vals, dG_ltd_vals = [], []
        for seg in self.ltd_segments:
            dg = -np.diff(seg)  # make positive for LTD magnitude
            g = seg[:-1]
            mask = dg > 0
            G_ltd_vals.append(g[mask])
            dG_ltd_vals.append(dg[mask])

        # Bin by G and average
        n_bins = 50
        G_bins = np.linspace(self.G_min, self.G_max, n_bins + 1)
        G_ctr = 0.5 * (G_bins[:-1] + G_bins[1:])

        def bin_average(G_vals, dG_vals):
            avg = np.zeros(n_bins)
            cnt = np.zeros(n_bins, dtype=int)
            all_g = np.concatenate(G_vals) if G_vals else np.array([])
            all_d = np.concatenate(dG_vals) if dG_vals else np.array([])
            for bi in range(n_bins):
                mask = (all_g >= G_bins[bi]) & (all_g < G_bins[bi + 1])
                if mask.sum() > 0:
                    avg[bi] = all_d[mask].mean()
                    cnt[bi] = mask.sum()
            # Smooth: interpolate NaNs, then apply a running mean
            valid = cnt > 0
            if valid.sum() >= 2:
                f = interp1d(G_ctr[valid], avg[valid], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
                avg = f(G_ctr)
            elif valid.sum() == 1:
                avg[:] = avg[valid][0]
            # Light smoothing
            kernel = np.ones(5) / 5
            avg = np.convolve(avg, kernel, mode='same')
            avg = np.maximum(avg, 0)  # dG must be >= 0
            return interp1d(G_ctr, avg, kind='linear', bounds_error=False,
                            fill_value='extrapolate')

        self._dG_ltp_fn = bin_average(G_ltp_vals, dG_ltp_vals)
        self._dG_ltd_fn = bin_average(G_ltd_vals, dG_ltd_vals)

        # Stats
        g_mid = self.G_mid_val
        print(f"    dG_LTP({g_mid*1e9:.1f}nS) = {self._dG_ltp_fn(g_mid)*1e9:.3f} nS/pulse")
        print(f"    dG_LTD({g_mid*1e9:.1f}nS) = {self._dG_ltd_fn(g_mid)*1e9:.3f} nS/pulse")

    # ------------------------------------------------------------------
    def _build_curves(self):
        """
        Numerically integrate dG(G) to get full G(p) curves
        for each starting G bin.
        """
        n = self.n_G_bins
        max_p = self.max_pulses
        self.G_centers = np.linspace(self.G_min, self.G_max, n)

        self.ltp_curves = np.zeros((n, max_p + 1))
        self.ltd_curves = np.zeros((n, max_p + 1))

        for bi in range(n):
            G0 = self.G_centers[bi]
            # LTP
            G_cur = G0
            self.ltp_curves[bi, 0] = G0
            for p in range(1, max_p + 1):
                dG = float(self._dG_ltp_fn(G_cur))
                G_cur = min(G_cur + max(dG, 0), self.G_max)
                self.ltp_curves[bi, p] = G_cur
            # LTD
            G_cur = G0
            self.ltd_curves[bi, 0] = G0
            for p in range(1, max_p + 1):
                dG = float(self._dG_ltd_fn(G_cur))
                G_cur = max(G_cur - max(dG, 0), self.G_min)
                self.ltd_curves[bi, p] = G_cur

        # Effective levels
        G0 = self.G_min
        for p in range(max_p + 1):
            dG = float(self._dG_ltp_fn(G0))
            G0 = min(G0 + max(dG, 0), self.G_max)
        ltp_unique = len(np.unique(np.round(self.ltp_curves[0], 12)))
        G0 = self.G_max
        for p in range(max_p + 1):
            dG = float(self._dG_ltd_fn(G0))
            G0 = max(G0 - max(dG, 0), self.G_min)
        ltd_unique = len(np.unique(np.round(self.ltd_curves[-1], 12)))
        print(f"    Effective LTP levels: {ltp_unique}, LTD levels: {ltd_unique}")

    # ------------------------------------------------------------------
    def _build_lut(self):
        n = self.n_G_bins
        max_p = self.max_pulses

        self.lut_pulses_ltp = np.zeros((n, n), dtype=np.int32)
        self.lut_pulses_ltd = np.zeros((n, n), dtype=np.int32)
        self.lut_new_G_ltp  = np.zeros((n, n), dtype=np.float32)
        self.lut_new_G_ltd  = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            ltp_c = self.ltp_curves[i]
            ltd_c = self.ltd_curves[i]
            for j in range(i, n):
                Gj = self.G_centers[j]
                hits = np.where(ltp_c >= Gj - 1e-15)[0]
                p = int(hits[0]) if len(hits) > 0 else max_p
                p = min(p, max_p)
                self.lut_pulses_ltp[i, j] = p
                self.lut_new_G_ltp[i, j] = float(ltp_c[p])
            for j in range(0, i + 1):
                Gj = self.G_centers[j]
                hits = np.where(ltd_c <= Gj + 1e-15)[0]
                p = int(hits[0]) if len(hits) > 0 else max_p
                p = min(p, max_p)
                self.lut_pulses_ltd[i, j] = p
                self.lut_new_G_ltd[i, j] = float(ltd_c[p])

        self.lut_G = self.G_centers.copy()

    # ------------------------------------------------------------------
    def apply_pulses_vectorized(self, G_current, delta_G):
        shape = G_current.shape
        G_flat = G_current.ravel().astype(np.float64)
        dG_flat = delta_G.ravel().astype(np.float64)
        G_target = np.clip(G_flat + dG_flat, self.G_min, self.G_max)

        edges = np.linspace(self.G_min, self.G_max, self.n_G_bins + 1)
        bi = np.clip(np.digitize(G_flat, edges) - 1, 0, self.n_G_bins - 1)
        tj = np.clip(np.digitize(G_target, edges) - 1, 0, self.n_G_bins - 1)

        new_G = np.zeros_like(G_flat)
        pulses = np.zeros_like(G_flat, dtype=np.int32)

        m_ltp = dG_flat > 0
        m_ltd = dG_flat < 0
        m_zero = ~m_ltp & ~m_ltd

        if m_ltp.any():
            pulses[m_ltp] = self.lut_pulses_ltp[bi[m_ltp], tj[m_ltp]]
            new_G[m_ltp] = self.lut_new_G_ltp[bi[m_ltp], tj[m_ltp]]
        if m_ltd.any():
            pulses[m_ltd] = self.lut_pulses_ltd[bi[m_ltd], tj[m_ltd]]
            new_G[m_ltd] = self.lut_new_G_ltd[bi[m_ltd], tj[m_ltd]]
        if m_zero.any():
            new_G[m_zero] = G_flat[m_zero]

        return new_G.reshape(shape), pulses.reshape(shape)

    def apply_pulses(self, G_current, delta_G_target):
        new_G, pulses = self.apply_pulses_vectorized(
            np.array([G_current]), np.array([delta_G_target]))
        return float(new_G[0]), int(pulses[0])

    def G_to_weight(self, G, G_ref=None):
        if G_ref is None:
            G_ref = self.G_mid_val
        return (G - G_ref) / (self.G_max - self.G_min) * 2.0

    def weight_to_G(self, w, G_ref=None):
        if G_ref is None:
            G_ref = self.G_mid_val
        return np.clip(G_ref + w * (self.G_max - self.G_min) / 2.0,
                       self.G_min, self.G_max)

    @property
    def effective_ltp_levels(self):
        """Rough effective number of conductance levels for LTP."""
        dG_mid = float(self._dG_ltp_fn(self.G_mid_val))
        return max(int(self.G_span / max(dG_mid, 1e-15)), 1)

    @property
    def G_range(self):
        return self.G_min, self.G_max

    @property
    def G_mid(self):
        return self.G_mid_val
