"""
partial_correlator.py — Partial correlations and Seco-Granados R1/R2/R3 metrics.

This module computes per-symbol partial correlations over early/late windows and
derives the aggregation metrics R1, R2, and R3 (Seco-Granados-style).

Key design choice
-----------------
The symbol sign (bhat) is estimated from the FULL-symbol correlation:

    Bfull_raw = <auth_sym, recv_sym>

This is consistent with a realistic receiver that can integrate over the full
symbol interval (assuming tracking and symbol boundary alignment). The early/late
partial correlations are then aligned (wipe-off) using this bhat so that the
metrics are computed on coherently signed quantities.
"""

from __future__ import annotations

import numpy as np


class PartialCorrelator:
    """
    Compute partial correlations per symbol and global R-metrics.

    For each symbol:
      - Compute early partial correlation Bbeg_raw over the first wlen samples
      - Compute late  partial correlation Bend_raw over the last  wlen samples
      - Estimate bhat from the full-symbol correlation Bfull_raw
      - Align partial correlations:
            Bbeg = bhat * Bbeg_raw
            Bend = bhat * Bend_raw

    Aggregation:
      - R1, R2, R3 are computed from the arrays {Bbeg_k}, {Bend_k} over Nb symbols.
    """

    def __init__(self, window_fraction: float = 0.125):
        """
        Parameters
        ----------
        window_fraction:
            Fraction of the symbol used for each partial window. Must be in (0, 0.5].
            For Galileo E1-B with 4 ms symbols, 0.125 corresponds to 0.5 ms windows.
        """
        if not (0.0 < window_fraction <= 0.5):
            raise ValueError("window_fraction must be in (0, 0.5].")
        self.window_fraction = float(window_fraction)

    def _per_symbol_partial_corr(
        self,
        auth_sym: np.ndarray,
        recv_sym: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Compute partial correlations for a single symbol (scalar helper).

        Returns
        -------
        Bbeg:
            Aligned early partial correlation.
        Bend:
            Aligned late partial correlation.
        bhat:
            Estimated symbol sign (+1 or -1) from the full-symbol correlation.
        """
        auth_sym = np.asarray(auth_sym, dtype=float)
        recv_sym = np.asarray(recv_sym, dtype=float)

        if auth_sym.shape != recv_sym.shape:
            raise ValueError("auth_sym and recv_sym must have the same shape.")

        n = auth_sym.size
        wlen = int(n * self.window_fraction)
        if wlen < 1:
            raise ValueError("Window length (wlen) must be at least 1.")

        # Early/late slices
        auth_beg = auth_sym[:wlen]
        auth_end = auth_sym[-wlen:]
        recv_beg = recv_sym[:wlen]
        recv_end = recv_sym[-wlen:]

        # Raw partial correlations (detection windows)
        Bbeg_raw = float(np.dot(auth_beg, recv_beg))
        Bend_raw = float(np.dot(auth_end, recv_end))

        # Full-symbol correlation for sign estimation (wipe-off)
        Bfull_raw = float(np.dot(auth_sym, recv_sym))

        # Estimate bhat from full-symbol correlation (conservative tie-breaking on zero).
        if Bfull_raw > 0.0:
            bhat = 1.0
        elif Bfull_raw < 0.0:
            bhat = -1.0
        else:
            # Rare tie: fall back to late window, then early window.
            if Bend_raw > 0.0:
                bhat = 1.0
            elif Bend_raw < 0.0:
                bhat = -1.0
            else:
                bhat = 1.0 if Bbeg_raw >= 0.0 else -1.0

        # Align both partial correlations using the same bhat.
        Bbeg = bhat * Bbeg_raw
        Bend = bhat * Bend_raw

        return Bbeg, Bend, bhat

    def partial_correlations(
        self,
        auth_samples: np.ndarray,
        recv_samples: np.ndarray,
        samples_per_symbol: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized per-symbol partial correlations for a full sequence.

        Parameters
        ----------
        auth_samples:
            Authentic/reference samples (e.g., local replica or authentic TX), length = Nsym * Ns.
        recv_samples:
            Received samples (baseline or attack), length = Nsym * Ns.
        samples_per_symbol:
            Number of samples per symbol (Ns).

        Returns
        -------
        Bbeg:
            Aligned early partial correlations, shape (Nsym,).
        Bend:
            Aligned late partial correlations, shape (Nsym,).
        bhat:
            Estimated symbol signs from full-symbol correlations, shape (Nsym,).
        """
        auth_samples = np.asarray(auth_samples, dtype=float)
        recv_samples = np.asarray(recv_samples, dtype=float)

        if auth_samples.size != recv_samples.size:
            raise ValueError("auth_samples and recv_samples must have the same length.")
        if samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive.")
        if auth_samples.size % samples_per_symbol != 0:
            raise ValueError("Total number of samples must be a multiple of samples_per_symbol.")

        n_symbols = auth_samples.size // samples_per_symbol
        n = samples_per_symbol

        wlen = int(n * self.window_fraction)
        if wlen < 1:
            raise ValueError("Window length (wlen) must be at least 1.")

        auth_mat = auth_samples.reshape(n_symbols, n)
        recv_mat = recv_samples.reshape(n_symbols, n)

        # Raw partial correlations (vectorized)
        Bbeg_raw = np.sum(auth_mat[:, :wlen] * recv_mat[:, :wlen], axis=1)
        Bend_raw = np.sum(auth_mat[:, -wlen:] * recv_mat[:, -wlen:], axis=1)

        # Full-symbol correlation for sign estimation (vectorized)
        Bfull_raw = np.sum(auth_mat * recv_mat, axis=1)

        # bhat = sign(Bfull_raw), with explicit tie-breaking for Bfull_raw == 0.
        bhat = np.ones(n_symbols, dtype=float)

        pos = Bfull_raw > 0.0
        neg = Bfull_raw < 0.0
        zero = ~(pos | neg)

        bhat[pos] = 1.0
        bhat[neg] = -1.0

        if np.any(zero):
            z = np.where(zero)[0]
            br_end = Bend_raw[z]
            br_beg = Bbeg_raw[z]

            bh_z = np.ones(z.size, dtype=float)
            bh_z[br_end < 0.0] = -1.0

            # If Bend_raw == 0, fall back to Bbeg_raw.
            tie2 = (br_end == 0.0)
            bh_z[tie2] = np.where(br_beg[tie2] >= 0.0, 1.0, -1.0)

            bhat[z] = bh_z

        # Align partial correlations (wipe-off)
        Bbeg = bhat * Bbeg_raw
        Bend = bhat * Bend_raw

        return Bbeg, Bend, bhat

    def r_metrics(
        self,
        Bbeg: np.ndarray,
        Bend: np.ndarray,
        eps: float = 1e-12,
    ) -> dict[str, float]:
        """
        Compute the global metrics R1, R2, and R3 from arrays of partial correlations.

        Definitions (aggregation over Nb symbols)
        ----------------------------------------
        Let:
            S_beg = sum_k Bbeg_k
            S_end = sum_k Bend_k
            r     = S_beg / S_end

        Then:
            R1 = |r|
            R2 = |r - 1|
            R3 = |mean_k (Bbeg_k - Bend_k)|

        Parameters
        ----------
        Bbeg, Bend:
            Arrays of aligned partial correlations (same shape).
        eps:
            Small constant to avoid division by zero when |S_end| is extremely small.

        Returns
        -------
        dict with keys {"R1", "R2", "R3"}.
        """
        Bbeg = np.asarray(Bbeg, dtype=float)
        Bend = np.asarray(Bend, dtype=float)

        if Bbeg.shape != Bend.shape:
            raise ValueError("Bbeg and Bend must have the same shape.")
        if Bbeg.size == 0:
            raise ValueError("Empty correlation arrays.")

        sum_beg = float(np.sum(Bbeg))
        sum_end = float(np.sum(Bend))

        ratio = 0.0 if abs(sum_end) < eps else (sum_beg / sum_end)

        R1 = abs(ratio)
        R2 = abs(ratio - 1.0)
        R3 = abs(float(np.mean(Bbeg - Bend)))

        return {"R1": float(R1), "R2": float(R2), "R3": float(R3)}
