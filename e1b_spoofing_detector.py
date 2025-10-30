"""
Simplified framework - Spoofing detection using partial correlations in Galileo E1-B signal.

This module implements three core components:
1. E1BSignalGenerator  - Generates symbols (+1/-1) and expands them into samples
2. Spoofer            - Introduces bit errors with configurable probability
3. PartialCorrelator  - Computes correlation between start/end windows of each symbol
                        and returns a global metric

Initial configuration validates maximum correlation case without spoofing or noise.

Future work:
- AWGN noise module
- Final detection metric and decision threshold
- ROC curve analysis

Author: Pedro Garcia Suarez
Date: October 2025
"""
from __future__ import annotations

import numpy as np

################################################################################
# 1. E1-B Symbol Generator (simplified)
################################################################################

class E1BSignalGenerator:
    """Generates a sequence of bits (+1/-1) and expands them into discrete samples."""

    def __init__(self, symbol_duration_ms: float = 4.0, samples_per_symbol: int = 8):
        if samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be >= 2 to take two windows")
        self.symbol_duration_ms = symbol_duration_ms
        self.samples_per_symbol = samples_per_symbol

    # ---------------------------------------------------------------------
    def generate_bits(self, num_symbols: int, seed: int | None = None) -> np.ndarray:
        """Returns an array of +1/-1 of length *num_symbols*."""
        rng = np.random.default_rng(seed)
        return rng.choice([-1, 1], size=num_symbols)

    # ---------------------------------------------------------------------
    def bits_to_samples(self, bits: np.ndarray) -> np.ndarray:
        """Converts bits to samples by repeating each one *samples_per_symbol* times."""
        return np.repeat(bits, self.samples_per_symbol)

################################################################################
# 2. Spoofer (currently disabled -error_prob = 0)
################################################################################

class Spoofer:
    """Introduces bit inversions with a given probability (error_prob)."""

    def __init__(self, error_prob: float = 0.0, seed: int | None = None):
        if not 0.0 <= error_prob <= 1.0:
            raise ValueError("error_prob must be in [0, 1]")
        self.error_prob = error_prob
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def spoof(self, samples: np.ndarray, samples_per_symbol: int) -> np.ndarray:
        """Flip the first window of each symbol with probability *error_prob*."""
        out = samples.copy()
        wlen = int(samples_per_symbol * 0.125)      # 0.5 ms window
        n_symbols = len(samples) // samples_per_symbol

        for k in range(n_symbols):
            if self.rng.random() < self.error_prob:
                start = k * samples_per_symbol
                out[start : start + wlen] *= -1      # invert only the first window
        return out
    
    def spoof_sample(self, samples: np.ndarray, samples_per_symbol: int, window_fraction: float = 0.125) -> np.ndarray:
        """Flip each *sample* in the first window with probability P."""
        wlen  = int(samples_per_symbol * window_fraction)
        if wlen < 1:
            raise ValueError("window too short; adjust samples_per_symbol or window_fraction")
        
        total = len(samples)

        # Build a boolean mask that selects ONLY the first window of EVERY symbol
        first_windows = np.zeros(total, dtype=bool)
        for k in range(total // samples_per_symbol):
            start = k * samples_per_symbol
            first_windows[start : start + wlen] = True

        # For those positions, decide flips with independent probability P
        flip_mask = first_windows & (self.rng.random(total) < self.error_prob)
        spoofed   = samples.copy()
        spoofed[flip_mask] *= -1
        return spoofed

################################################################################
# 3. Partial Correlator
################################################################################

class PartialCorrelator:
    """Calculates partial correlation between the start and end of each symbol."""

    def __init__(self, window_fraction: float = 0.125):
        # 0.125 equals 0.5 ms if symbol duration is 4 ms
        if not (0 < window_fraction <= 0.5):
            raise ValueError("window_fraction must be in (0, 0.5]")
        self.window_fraction = window_fraction

    # ------------------------------------------------------------------
    def _symbol_metric(self, samples: np.ndarray, samples_per_symbol: int) -> float:
        wlen = int(samples_per_symbol * self.window_fraction)
        if wlen < 1:
            raise ValueError("Window length (wlen) must be at least 1. Adjust samples_per_symbol or window_fraction.")
        first = samples[:wlen]
        last = samples[-wlen:]
        # correlation normalized to window length
        return float(np.dot(first, last) / wlen)

    def sequence_metric(self, all_samples: np.ndarray, samples_per_symbol: int) -> float:
        if len(all_samples) % samples_per_symbol != 0:
            raise ValueError("Total number of samples is not a multiple of samples_per_symbol.")
        n_symbols = len(all_samples) // samples_per_symbol
        symbols = all_samples[: n_symbols * samples_per_symbol].reshape(n_symbols, samples_per_symbol)
        return float(np.mean([self._symbol_metric(s, samples_per_symbol) for s in symbols]))

################################################################################
# 4. Usage Example (informal unit test)
################################################################################

if __name__ == "__main__":
    SAMPLES_PER_CHIP   = 16
    CHIPS_PER_SYMBOL   = 4092
    SAMPLES_PER_SYMBOL = CHIPS_PER_SYMBOL * SAMPLES_PER_CHIP

    NUM_SYMBOLS        = 1000

    # 1) authentic signal
    gen = E1BSignalGenerator(samples_per_symbol=SAMPLES_PER_SYMBOL)
    bits = gen.generate_bits(NUM_SYMBOLS, seed=42)
    samples  = gen.bits_to_samples(bits)

    # 2) spoofer: flip the first window of each symbol with X % probability
    spoofer    = Spoofer(error_prob=0, seed=24)
    spoofed    = spoofer.spoof(samples, gen.samples_per_symbol)

    # 3) partial-correlation metric
    correlator = PartialCorrelator(window_fraction=0.125)
    metric = correlator.sequence_metric(spoofed, gen.samples_per_symbol)
    print(f"Partial-correlation metric: {metric:.3f}")

    # 4) quick debug on the first 20 samples
    dbg = 20
    print("Original:", samples[:dbg])
    print("Spoofed :", spoofed[:dbg])
    print("Flipped :", (samples[:dbg] != spoofed[:dbg]).astype(int))



    # ------------------------------------------------------------------
    # TODO: Add AWGN and iterate to calculate ROC curves
    # ------------------------------------------------------------------
