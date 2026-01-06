"""
Partial correlations and Seco-Granados R1, R2, R3 metrics.
"""

from __future__ import annotations

import numpy as np


class PartialCorrelator:
    """
    Calculates partial correlations between the beginning and the end of each symbol
    for both the authentic and received (possibly spoofed) signals,
    and then derives the Seco-Granados metrics R1, R2 and R3.
    """

    def __init__(self, window_fraction: float = 0.125):
        # 0.125 equals 0.5 ms if symbol duration is 4 ms
        if not (0 < window_fraction <= 0.5):
            raise ValueError("window_fraction must be in (0, 0.5]")
        self.window_fraction = window_fraction

    # ------------------------------------------------------------------
    def _per_symbol_partial_corr(
        self,
        auth_sym: np.ndarray,
        recv_sym: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Returns (Bbeg_aligned, Bend_aligned, bhat)
        where bhat is the estimated symbol sign (+1/-1) from the late window.
        """
        if auth_sym.shape != recv_sym.shape:
            raise ValueError("auth_sym and recv_sym must have the same length")

        n = auth_sym.size
        wlen = int(n * self.window_fraction)
        if wlen < 1:
            raise ValueError("Window length (wlen) must be at least 1.")

        auth_beg = auth_sym[:wlen]
        auth_end = auth_sym[-wlen:]
        recv_beg = recv_sym[:wlen]
        recv_end = recv_sym[-wlen:]

        # Raw partial correlations
        Bbeg_raw = float(np.dot(auth_beg, recv_beg))
        Bend_raw = float(np.dot(auth_end, recv_end))

        # Realistic wipe-off: estimate symbol sign from late window
        # (handle exact zero conservatively)
        if Bend_raw > 0.0:
            bhat = 1.0
        elif Bend_raw < 0.0:
            bhat = -1.0
        else:
            # tie-breaker (rare): fall back to beg, or set +1
            bhat = 1.0 if Bbeg_raw >= 0.0 else -1.0

        # Align both partial correlations to a common sign
        Bbeg = bhat * Bbeg_raw
        Bend = bhat * Bend_raw  # == abs(Bend_raw) except tie
        return Bbeg, Bend, bhat

    # ------------------------------------------------------------------
    def partial_correlations(
        self,
        auth_samples: np.ndarray,
        recv_samples: np.ndarray,
        samples_per_symbol: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Bbeg(k) and Bend(k) for all symbols using realistic sign estimation.
        Returns (Bbeg, Bend, bhat_arr).
        """
        if len(auth_samples) != len(recv_samples):
            raise ValueError("auth_samples and recv_samples must have the same length")
        if len(auth_samples) % samples_per_symbol != 0:
            raise ValueError("Total number of samples is not a multiple of samples_per_symbol.")

        n_symbols = len(auth_samples) // samples_per_symbol

        auth_reshaped = auth_samples.reshape(n_symbols, samples_per_symbol)
        recv_reshaped = recv_samples.reshape(n_symbols, samples_per_symbol)

        Bbeg = np.zeros(n_symbols, dtype=float)
        Bend = np.zeros(n_symbols, dtype=float)
        bhat = np.zeros(n_symbols, dtype=float)

        for k in range(n_symbols):
            Bbeg[k], Bend[k], bhat[k] = self._per_symbol_partial_corr(
                auth_reshaped[k],
                recv_reshaped[k],
            )
        return Bbeg, Bend, bhat

    # ------------------------------------------------------------------
    def r_metrics(
        self,
        Bbeg: np.ndarray,
        Bend: np.ndarray,
        eps: float = 1e-12,
    ) -> dict[str, float]:
        """
        Compute the global metrics R1, R2 and R3 from arrays of partial correlations.
        The formulas follow Seco-Granados et al. (GPS Solutions, 2021, eqs. (6)-(8)).
        """
        if Bbeg.shape != Bend.shape:
            raise ValueError("Bbeg and Bend must have the same shape")
        Nb = Bbeg.size
        if Nb == 0:
            raise ValueError("Empty correlation arrays")

        sum_beg = float(np.sum(Bbeg))
        sum_end = float(np.sum(Bend))

        ratio = 0.0 if abs(sum_end) < eps else (sum_beg / sum_end)

        R1 = abs(ratio)
        R2 = abs(ratio - 1.0)
        R3 = abs(float(np.mean(Bbeg - Bend)))

        return {"R1": float(R1), "R2": float(R2), "R3": float(R3)}

    # ------------------------------------------------------------------
    def sequence_metrics(
        self,
        auth_samples: np.ndarray,
        recv_samples: np.ndarray,
        samples_per_symbol: int,
        bits: np.ndarray,
    ) -> dict[str, float]:
        """
        Convenience wrapper:
        1) compute Bbeg(k), Bend(k);
        2) compute R1, R2, R3.
        """
        Bbeg, Bend, bhat = self.partial_correlations(
            auth_samples,
            recv_samples,
            samples_per_symbol,
        )
        return self.r_metrics(Bbeg, Bend)
