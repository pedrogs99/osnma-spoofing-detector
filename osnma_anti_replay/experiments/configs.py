"""
experiments/configs.py — Configuration objects for experiment sweeps.

This module defines lightweight, immutable configuration containers used by the
experiment runner (grid sweeps, Monte Carlo trials, and scenario selection).

Design goals
------------
- Keep configuration explicit and serializable.
- Make runs reproducible via a single master_seed and deterministic seeding
  offsets across components (signal generation, channels, spoofer, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Scenario(str, Enum):
    """
    Experiment scenario selector.

    BASELINE:
        No attack (H0). The victim receiver observes the authentic signal.
    SCER:
        Replay/spoofer model (H1). The attacker observes the authentic signal,
        estimates symbol signs, and replays a synthesized signal.
    """

    BASELINE = "baseline"
    SCER = "scer"


@dataclass(frozen=True)
class SignalConfig:
    """
    Signal-generation parameters.

    Parameters
    ----------
    e1b_code_hex:
        Galileo E1-B spreading code as a hexadecimal string (ICD Annex C format).
    t_symbol_ms:
        Symbol duration in milliseconds. For E1-B, this is typically 4.0 ms.
    """

    e1b_code_hex: str
    t_symbol_ms: float = 4.0


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Experiment sweep configuration.

    Parameters
    ----------
    window_fraction_corr:
        Fraction of the symbol used for early and late partial-correlation windows.
        Must be in (0, 0.5].
    num_symbols_total:
        Total number of symbols generated per Monte Carlo trial (trial length).
        Must be >= max(nb_list) if fixed-length aggregation uses Nb symbols.
    nb_list:
        List of aggregation lengths Nb (number of symbols used to compute metrics).
        Example: [50, 200, 500, 1000].
    cn0_list_dbhz:
        List of C/N0 points (dB-Hz) for the sweep.
    n_trials:
        Number of Monte Carlo trials per grid point.

    cn0_same_for_spoofer_and_rx:
        Noise-model switch:
          - True  : spoofer input C/N0 equals victim receiver C/N0 ("same-CN0" model)
          - False : spoofer and receiver may use different C/N0 settings (if supported)

    master_seed:
        Base seed used to derive deterministic per-trial/per-component seeds.
    """

    window_fraction_corr: float
    num_symbols_total: int
    nb_list: Sequence[int]
    cn0_list_dbhz: Sequence[float]
    n_trials: int

    cn0_same_for_spoofer_and_rx: bool = True
    master_seed: int = 12345


@dataclass(frozen=True)
class SpooferConfig:
    """
    Spoofer configuration used in experiment sweeps.

    Parameters
    ----------
    lock_gamma:
        Lock threshold parameter passed to the spoofer (see SpooferCfg).
    seed_offset:
        Constant offset applied when deriving spoofer seeds from master_seed to
        de-correlate RNG streams between components.
    """

    lock_gamma: float = 0.0
    seed_offset: int = 10_000
