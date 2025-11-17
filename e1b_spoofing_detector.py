"""
Simplified framework - Spoofing detection using partial correlations in Galileo E1-B signal.

This module implements three core components:
1. E1BSignalGenerator  - Generates symbols (+1/-1) and expands them into samples
2. Spoofer            - Introduces bit errors with configurable probability
3. PartialCorrelator   - Computes Seco-Granados R1, R2 and R3 metrics from partial correlations
                         between a 'perfect' replica and a (possibly spoofed) received signal.

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
    """Introduces bit/sample inversions with a given probability (error_prob)."""

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
        """
        Flip each *sample* in the first window with probability P.

        In this simplified model, one 'sample' can be interpreted as one chip.
        Later we can introduce an oversampling factor (samples per chip).
        """
        wlen  = int(samples_per_symbol * window_fraction)
        if wlen < 1:
            raise ValueError("window too short; adjust samples_per_symbol or window_fraction")
        
        total = len(samples)

        # Boolean mask that selects ONLY the first window of EVERY symbol
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
# 3. Partial Correlator (R1, R2, R3)
################################################################################

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
        symbol_bit: float,
    ) -> tuple[complex, complex]:
        """
        Compute Bbeg(k) and Bend(k) (after removing the symbol sign) for one symbol.

        auth_sym and recv_sym are the samples of one symbol of the authentic and received signals.
        symbol_bit is b(k) in {+1,-1}.
        """
        if auth_sym.shape != recv_sym.shape:
            raise ValueError("auth_sym and recv_sym must have the same length")

        n = auth_sym.size
        wlen = int(n * self.window_fraction)
        if wlen < 1:
            raise ValueError("Window length (wlen) must be at least 1. Adjust samples_per_symbol or window_fraction.")

        # Early and late windows
        auth_beg = auth_sym[:wlen]
        auth_end = auth_sym[-wlen:]
        recv_beg = recv_sym[:wlen]
        recv_end = recv_sym[-wlen:]

        # Raw partial cross-correlations (like eqs. (2)-(3) in Seco-Granados, real baseband)
        # vdot does conj(first) * second and sums -> <auth, recv>
        Bbeg_raw = np.vdot(auth_beg, recv_beg)
        Bend_raw = np.vdot(auth_end, recv_end)

        # Remove the sign of the unpredictable symbol (eqs. (4)-(5))
        b = float(symbol_bit)
        if b not in (-1.0, 1.0):
            raise ValueError("symbol_bit must be +/-1")
        Bbeg = b * Bbeg_raw
        Bend = b * Bend_raw
        return complex(Bbeg), complex(Bend)

    # ------------------------------------------------------------------
    def partial_correlations(
        self,
        auth_samples: np.ndarray,
        recv_samples: np.ndarray,
        samples_per_symbol: int,
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Bbeg(k) and Bend(k) for all symbols.
        Returns two complex-valued arrays of length Nb.
        """
        if len(auth_samples) != len(recv_samples):
            raise ValueError("auth_samples and recv_samples must have the same length")
        if len(auth_samples) % samples_per_symbol != 0:
            raise ValueError("Total number of samples is not a multiple of samples_per_symbol.")

        n_symbols = len(auth_samples) // samples_per_symbol
        if len(bits) != n_symbols:
            raise ValueError("Length of bits must be equal to number of symbols")

        auth_reshaped = auth_samples.reshape(n_symbols, samples_per_symbol)
        recv_reshaped = recv_samples.reshape(n_symbols, samples_per_symbol)

        Bbeg = np.zeros(n_symbols, dtype=complex)
        Bend = np.zeros(n_symbols, dtype=complex)

        for k in range(n_symbols):
            Bbeg[k], Bend[k] = self._per_symbol_partial_corr(
                auth_reshaped[k],
                recv_reshaped[k],
                bits[k],
            )
        return Bbeg, Bend

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

        sum_beg = np.sum(Bbeg)
        sum_end = np.sum(Bend)

        # Avoid division by (almost) zero in very low SNR cases
        if abs(sum_end) < eps:
            ratio = 0.0 + 0.0j
        else:
            ratio = sum_beg / sum_end

        # R1: | sum(Bbeg) / sum(Bend) |
        R1 = abs(ratio)

        # R2: | sum(Bbeg) / sum(Bend) - 1 |
        R2 = abs(ratio - 1.0)

        # R3: | mean( Bbeg(k) - Bend(k) ) |
        R3 = abs(np.mean(Bbeg - Bend))

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
        Bbeg, Bend = self.partial_correlations(
            auth_samples,
            recv_samples,
            samples_per_symbol,
            bits,
        )
        return self.r_metrics(Bbeg, Bend)

################################################################################
# 4. AWGN Channel
################################################################################

class AWGNChannel:
    """
    Adds real-valued Additive White Gaussian Noise (AWGN) to a discrete-time signal.

    The noise variance is chosen such that:
        SNR_linear = P_signal / P_noise
        P_noise    = sigma^2

    with:
        SNR_linear = 10^(snr_db / 10)

    where P_signal is estimated from the input signal as mean(|x[n]|^2).
    """

    def __init__(self, snr_db: float, seed: int | None = None):
        self.snr_db = float(snr_db)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Returns noisy_signal = signal + w, where w ~ N(0, sigma^2),
        and sigma is chosen to achieve the target SNR (in dB).
        """
        # 1) Estimate signal power
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            raise ValueError("Empty signal")

        P_signal = np.mean(signal**2)  # average power

        # 2) Compute noise variance from SNR
        snr_linear = 10.0**(self.snr_db / 10.0)
        if snr_linear <= 0:
            raise ValueError("snr_db must define a positive linear SNR")

        sigma2 = P_signal / snr_linear
        sigma = np.sqrt(sigma2)

        # 3) Draw noise and add
        noise = sigma * self.rng.normal(size=signal.shape)
        return signal + noise


################################################################################
# 5. Usage Example (informal unit test)
################################################################################

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example configuration (toy values)
    # ------------------------------------------------------------------
    CHIPS_PER_SYMBOL   = 4092
    SAMPLES_PER_CHIP   = 1     # interpret each discrete-time sample as one chip
    SAMPLES_PER_SYMBOL = CHIPS_PER_SYMBOL * SAMPLES_PER_CHIP

    NUM_SYMBOLS        = 1000

    # 1) authentic signal (what the receiver would store as 'perfect' replica)
    gen = E1BSignalGenerator(samples_per_symbol=SAMPLES_PER_SYMBOL)
    bits = gen.generate_bits(NUM_SYMBOLS, seed=42)
    auth_samples = gen.bits_to_samples(bits)

    # 2) spoofer: flip samples in the first window of each symbol with X % probability
    spoofer      = Spoofer(error_prob=0, seed=48)
    recv_samples = spoofer.spoof_sample(auth_samples, gen.samples_per_symbol)

    # 3) AWGN channel: e.g. SNR = 20 dB
    channel      = AWGNChannel(snr_db=40.0, seed=24)
    noisy_recv   = channel.add_noise(recv_samples)

    # 4) partial-correlation metrics R1, R2, R3
    correlator = PartialCorrelator(window_fraction=0.125)
    metrics = correlator.sequence_metrics(
        auth_samples,
        noisy_recv,
        gen.samples_per_symbol,
        bits,
    )
    print(f"R1 = {metrics['R1']:.6f}, R2 = {metrics['R2']:.6f}, R3 = {metrics['R3']:.6f}")



    # ------------------------------------------------------------------
    # TODO: Add AWGN and iterate to calculate ROC curves
    # ------------------------------------------------------------------
