"""
awgn_channel.py — AWGN channel model for chip-rate discrete-time signals (C/N0-based).

This module provides a simple real-valued AWGN channel where the noise level is
specified via the carrier-to-noise density ratio C/N0 (in dB-Hz) and the chip
rate (chips/s). It is intended for simulations that operate at one sample per
chip (chip-rate sampling).

Model assumptions:
- Real baseband samples.
- One sample per chip.
- Unit signal power per sample is a reasonable approximation for BPSK-like chips
  (e.g., +/-1).
"""

from __future__ import annotations

import numpy as np


class AWGNChannel:
  """
  Real-valued AWGN channel for chip-rate signals (one sample per chip).

  Parameters
  ----------
  cn0_db_hz:
      Carrier-to-noise density ratio in dB-Hz.
  chip_rate_hz:
      Chip rate in chips per second. With one sample per chip, this is also
      the sampling rate in samples per second.
  seed:
      Optional RNG seed for reproducibility.

  Notes
  -----
  This implementation selects the per-sample noise standard deviation so that,
  for a unit-power signal, the coherent-integration SNR over an interval
  T_int satisfies:

      SNR_coh = (C/N0) * T_int

  With chip-rate sampling:
      T_chip = 1 / chip_rate_hz

  For a unit-power signal and real AWGN, the per-sample noise variance is
  chosen as:

      sigma^2 = 1 / ((C/N0)_linear * T_chip)

  Therefore:
      sigma = sqrt(1 / ((C/N0)_linear * T_chip))
  """

  def __init__(self, cn0_db_hz: float, chip_rate_hz: float, seed: int | None = None):
    self.cn0_db_hz = float(cn0_db_hz)
    self.chip_rate_hz = float(chip_rate_hz)

    if self.chip_rate_hz <= 0.0:
        raise ValueError("chip_rate_hz must be positive.")

    self.rng = np.random.default_rng(seed)

    cn0_linear = 10.0 ** (self.cn0_db_hz / 10.0)
    t_chip = 1.0 / self.chip_rate_hz

    # Unit signal power per sample is assumed (e.g., +/-1 chips).
    self.sigma = np.sqrt(1.0 / (cn0_linear * t_chip))

  def add_noise(self, signal: np.ndarray) -> np.ndarray:
    """
    Add AWGN to the input signal.

    Parameters
    ----------
    signal:
        Input array of real-valued samples.

    Returns
    -------
    np.ndarray
        Noisy signal (signal + noise) with the same shape as the input.
    """
    signal = np.asarray(signal, dtype=float)
    noise = self.sigma * self.rng.standard_normal(size=signal.shape)
    return signal + noise
