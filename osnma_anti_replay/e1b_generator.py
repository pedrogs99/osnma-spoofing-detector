"""
E1B signal generator: produces random bits (+1/-1) and expands them into samples.
"""

from __future__ import annotations

import numpy as np


class E1BSignalGenerator:
    """Generates a sequence of bits (+1/-1) and expands them into discrete samples."""

    def __init__(self, symbol_duration_ms: float = 4.0, samples_per_symbol: int = 8):
        if samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be >= 2 to take two windows")
        self.symbol_duration_ms = symbol_duration_ms
        self.samples_per_symbol = samples_per_symbol

    # ------------------------------------------------------------------
    def generate_bits(self, num_symbols: int, seed: int | None = None) -> np.ndarray:
        """Returns an array of +1/-1 of length *num_symbols*."""
        rng = np.random.default_rng(seed)
        return rng.choice([-1, 1], size=num_symbols)

    # ------------------------------------------------------------------
    def bits_to_samples(self, bits: np.ndarray) -> np.ndarray:
        """Converts bits to samples by repeating each one *samples_per_symbol* times."""
        return np.repeat(bits, self.samples_per_symbol)
