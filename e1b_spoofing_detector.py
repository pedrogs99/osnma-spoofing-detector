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
    def spoof(self, bits: np.ndarray) -> np.ndarray:
        mask = self.rng.random(size=bits.size) < self.error_prob
        spoofed = bits.copy()
        spoofed[mask] *= -1  # inverts marked bits
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
    NUM_SYMBOLS = 100  # quick test

    # Generate "authentic" sequence
    gen = E1BSignalGenerator()
    bits = gen.generate_bits(NUM_SYMBOLS, seed=42)

    # Currently no spoofer (error_prob = 0)
    spoofer = Spoofer(error_prob=0.0, seed=24)
    spoofed_bits = spoofer.spoof(bits)

    # Convert to samples
    samples = gen.bits_to_samples(spoofed_bits)

    # Partial correlation
    correlator = PartialCorrelator(window_fraction=0.125)
    metric = correlator.sequence_metric(samples, gen.samples_per_symbol)

    print(f"Partial correlation metric (without spoofer): {metric:.2f}")

    # ------------------------------------------------------------------
    # TODO: Add AWGN and iterate to calculate ROC curves
    # ------------------------------------------------------------------
