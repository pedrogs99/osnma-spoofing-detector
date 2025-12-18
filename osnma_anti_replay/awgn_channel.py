"""
AWGN channel model for chip-rate discrete-time signals, specified via C/N0 (dB-Hz).
"""

from __future__ import annotations

import numpy as np


class AWGNChannel:
    """
    Adds real-valued AWGN to a chip-rate signal (one sample per chip).

    Noise is specified by:
      - C/N0 in dB-Hz
      - chip_rate_hz (chips per second)

    Assumes:
      - The clean signal has roughly unit power per sample (e.g. +/-1 chips).
      - Each sample corresponds to one chip (SAMPLES_PER_CHIP = 1).

    The per-chip noise variance is chosen so that the SNR after coherent
    integration over any interval T_int satisfies:

        SNR_coh_linear = (C/N0_linear) * T_int
    """

    def __init__(self, cn0_db_hz: float, chip_rate_hz: float, seed: int | None = None):
        self.cn0_db_hz = float(cn0_db_hz)
        self.chip_rate_hz = float(chip_rate_hz)
        if self.chip_rate_hz <= 0:
            raise ValueError("chip_rate_hz must be positive")
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Returns noisy_signal = signal + w, where w ~ N(0, sigma^2),
        with sigma^2 derived from C/N0 and the chip rate.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            raise ValueError("Empty signal")

        # 1) Estimate signal power per sample (in case you scale the amplitude later)
        P_signal = np.mean(signal**2)  # ~1 for +/-1 chips

        # 2) Convert C/N0 to linear
        cn0_linear = 10.0**(self.cn0_db_hz / 10.0)

        # 3) Chip duration
        T_chip = 1.0 / self.chip_rate_hz

        # 4) sigma^2 = P_signal / (C/N0_linear * T_chip)
        sigma2 = P_signal / (cn0_linear * T_chip)
        sigma = np.sqrt(sigma2)

        # 5) Draw noise and add
        noise = sigma * self.rng.normal(size=signal.shape)
        return signal + noise
