# vmd_decomposer.py
# ─────────────────────────────────────────────────────────────────────────────
# Variational Mode Decomposition (VMD) — pure NumPy/SciPy, no vmdpy needed.
#
# Math recap:
#   Decomposes price P(t) = Σ u_k(t),  k=1..K
#   Each mode u_k has a compact bandwidth around centre frequency ω_k.
#   Solved via ADMM in Fourier space.
#
#   Mode 0 (lowest frequency)  → TREND
#   Mode 1 (mid frequency)     → SEASONAL
#   Mode 2 (highest frequency) → SHOCK / RESIDUAL
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from config import (
    VMD_K, VMD_ALPHA, VMD_TAU, VMD_TOL, VMD_MAX_ITER, MODEL_DIR
)

VMD_CACHE = MODEL_DIR / "vmd_modes.pkl"


# ─────────────────────────────────────────────────────────────────────────────
#  Core VMD algorithm
# ─────────────────────────────────────────────────────────────────────────────
def vmd(signal: np.ndarray,
        K: int        = VMD_K,
        alpha: float  = VMD_ALPHA,
        tau: float    = VMD_TAU,
        tol: float    = VMD_TOL,
        max_iter: int = VMD_MAX_ITER) -> tuple:
    """
    Variational Mode Decomposition.

    Parameters
    ----------
    signal   : 1-D price array of length T
    K        : number of modes
    alpha    : bandwidth constraint (larger → smoother modes)
    tau      : Lagrangian multiplier step (0 = no noise tolerance)
    tol      : convergence criterion
    max_iter : max ADMM iterations

    Returns
    -------
    u       : ndarray (K, T) — reconstructed modes
    omega   : ndarray (K,)   — final centre frequencies
    n_iter  : int             — iterations until convergence
    """
    T  = len(signal)

    # ── Mirror signal to reduce edge effects ──────────────────────────────
    f_m  = np.concatenate([signal[::-1], signal, signal[::-1]])
    Tm   = len(f_m)
    t    = np.arange(1, Tm + 1) / Tm
    freqs = t - 0.5 - 1.0 / Tm

    # One-sided FFT of mirrored signal
    F_hat   = np.fft.fftshift(np.fft.fft(f_m))
    F_hat_p = F_hat[Tm // 2:]          # positive freqs only
    freqs_p = freqs[Tm // 2:]
    N       = len(freqs_p)

    # ── Initialise mode spectra & Lagrange multiplier ────────────────────
    U_hat   = np.zeros((K, N), dtype=complex)
    lam     = np.zeros(N, dtype=complex)

    # Initialise centre frequencies uniformly in [0, 0.5]
    omega = np.array([(k + 1) / (2.0 * K) for k in range(K)])

    u_prev  = U_hat.copy()
    n_iter  = 0
    diff    = tol + 1.0

    # ── ADMM iterations ───────────────────────────────────────────────────
    while diff > tol and n_iter < max_iter:
        for k in range(K):
            # Sum of all OTHER modes
            others = U_hat.sum(axis=0) - U_hat[k]

            # Update u_k in frequency domain
            numer        = F_hat_p - others + lam / 2.0
            denom        = 1.0 + 2.0 * alpha * (freqs_p - omega[k]) ** 2
            U_hat[k]     = numer / denom

            # Update centre frequency (weighted spectral centroid)
            power        = np.abs(U_hat[k]) ** 2
            omega[k]     = np.dot(freqs_p, power) / (power.sum() + 1e-10)

        # Update Lagrange multiplier
        lam  += tau * (F_hat_p - U_hat.sum(axis=0))

        # Convergence check
        diff    = np.sum(np.abs(U_hat - u_prev) ** 2) / (N + 1e-10)
        u_prev  = U_hat.copy()
        n_iter += 1

    # ── Reconstruct time-domain modes from one-sided spectrum ────────────
    u_out = np.zeros((K, T))
    for k in range(K):
        # Build full (two-sided) spectrum
        full = np.zeros(Tm, dtype=complex)
        full[Tm // 2:] = U_hat[k]
        full[1: Tm // 2 + 1] = np.conj(U_hat[k, Tm // 2 - 1::-1]
                                         if Tm // 2 - 1 >= 0
                                         else U_hat[k, ::-1][1:])
        u_full = np.real(np.fft.ifft(np.fft.ifftshift(full)))
        u_out[k] = u_full[Tm // 4: Tm // 4 + T]

    # Sort by energy descending (mode 0 = most energy = TREND)
    energies = np.array([np.sum(u_out[k] ** 2) for k in range(K)])
    order    = np.argsort(-energies)
    u_out    = u_out[order]
    omega    = omega[order]

    return u_out, omega, n_iter


# ─────────────────────────────────────────────────────────────────────────────
#  High-level wrapper
# ─────────────────────────────────────────────────────────────────────────────
class VMDDecomposer:
    """
    Decomposes a price series and returns a labeled DataFrame.

    Usage
    ─────
    vd = VMDDecomposer()
    modes_df = vd.decompose(prices)          # DataFrame: Trend, Seasonal, Shock
    trend    = vd.get_mode("Trend")          # np.ndarray
    """

    LABELS = {3: ["Trend", "Seasonal", "Shock"],
              2: ["Trend", "Shock"],
              4: ["Trend", "Seasonal", "Cycle", "Shock"]}

    def __init__(self, K: int = VMD_K, alpha: float = VMD_ALPHA):
        self.K      = K
        self.alpha  = alpha
        self.modes  = None   # ndarray (K, T) after decompose()
        self.omega  = None   # centre frequencies
        self.labels = self.LABELS.get(K, [f"Mode_{i}" for i in range(K)])

    # ── Decompose ─────────────────────────────────────────────────────────
    def decompose(self, prices: np.ndarray) -> pd.DataFrame:
        """
        Run VMD on raw price array.

        Returns pd.DataFrame with columns = self.labels
        """
        print(f"[VMD] Decomposing {len(prices)} samples into {self.K} modes...")
        u, omega, n_iter = vmd(prices, K=self.K, alpha=self.alpha)

        self.modes = u
        self.omega = omega

        df = pd.DataFrame(
            {self.labels[k]: u[k] for k in range(self.K)}
        )

        # Reconstruction quality check
        recon   = df.sum(axis=1).values
        rmse    = np.sqrt(np.mean((prices - recon) ** 2))
        rel_pct = rmse / (np.std(prices) + 1e-8) * 100
        print(f"[VMD] ✅ Converged in {n_iter} iterations")
        print(f"[VMD] Reconstruction RMSE: {rmse:.2f}  ({rel_pct:.1f}% of std) "
              f"{'✅' if rel_pct < 5 else '⚠️ try higher alpha'}")

        for k, lbl in enumerate(self.labels):
            print(f"  {lbl:12s}: ω={omega[k]:.4f}  "
                  f"range=[{u[k].min():.1f}, {u[k].max():.1f}]  "
                  f"std={u[k].std():.2f}")
        return df

    # ── Accessors ─────────────────────────────────────────────────────────
    def get_mode(self, name: str) -> np.ndarray:
        if self.modes is None:
            raise RuntimeError("Call decompose() first.")
        idx = self.labels.index(name)
        return self.modes[idx]

    # ── Save / Load ───────────────────────────────────────────────────────
    def save(self, path: Path = VMD_CACHE):
        with open(path, "wb") as f:
            pickle.dump({"modes": self.modes, "omega": self.omega,
                         "labels": self.labels, "K": self.K}, f)
        print(f"[VMD] Modes saved → {path}")

    def load(self, path: Path = VMD_CACHE):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.modes  = d["modes"]
        self.omega  = d["omega"]
        self.labels = d["labels"]
        self.K      = d["K"]
        print(f"[VMD] Modes loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    N = 300
    t = np.arange(N)
    sig = (1000 + 0.5*t
           + 80*np.sin(2*np.pi*t/365)
           + 30*np.sin(2*np.pi*t/30)
           + np.random.normal(0, 15, N))
    sig[150] += 400   # inject shock

    vd  = VMDDecomposer(K=3)
    mdf = vd.decompose(sig)
    print(mdf.head())