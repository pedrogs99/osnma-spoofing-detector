"""
spoofer.py — Simple SCER-like spoofer for chip-rate signals.

The spoofer estimates each navigation-bit sign b_k ∈ {±1} from the received
authentic signal by correlating against a local spreading-code replica using a
running (sample-by-sample) correlation. It then replays a synthesized signal:

    s[n] = b_hat[n] · c[n]

where c[n] is the known spreading code for one symbol.

This implementation makes the spoofer performance depend only on the quality of
its input (e.g., C/N0 at the spoofer input), rather than on an artificial error
probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpooferCfg:
    """
    Configuration for the spoofer running-decision logic.

    Parameters
    ----------
    lock_gamma:
        If > 0, a simple lock threshold is applied:
            lock when |S[n]| >= lock_gamma * sqrt(n)
        where S[n] is the running correlation at sample index n (1-based in the
        threshold expression). This is a heuristic to avoid noisy sign flips
        when S[n] is close to zero.

        If <= 0, the spoofer decides from the first sample:
            b_hat[n] = sign(S[n])

    prelock_mode:
        Output behavior before lock is achieved (only used when lock_gamma > 0):
          - "hold"   : hold the last decided value (initially +1)
          - "random" : output random ±1 until lock
          - "zero"   : output 0 until lock (energy-loss model)
    """

    lock_gamma: float = 0.0
    prelock_mode: str = "hold"  # {"hold", "random", "zero"}


class Spoofer:
    """Chip-rate spoofer with running correlation sign estimation."""

    def __init__(self, cfg: SpooferCfg | None = None, seed: int | None = None):
        self.cfg = cfg if cfg is not None else SpooferCfg()

        if self.cfg.prelock_mode not in {"hold", "random", "zero"}:
            raise ValueError("prelock_mode must be one of: 'hold', 'random', 'zero'.")

        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sign_pm1(x: np.ndarray) -> np.ndarray:
        """
        Map sign to {+1, -1}, using 0 -> +1.

        This avoids zero decisions when S[n] is exactly 0.
        """
        out = np.ones_like(x, dtype=float)
        out[x < 0] = -1.0
        return out

    def spoof_from_rx(
        self,
        rx_samples: np.ndarray,
        local_code: np.ndarray,
        samples_per_symbol: int,
        return_debug: bool = False,
    ):
        """
        Estimate per-sample and per-symbol signs from the spoofer input and
        synthesize a replay signal.

        Parameters
        ----------
        rx_samples:
            Spoofer input samples (authentic signal + noise), length = Nsym * Ns.
        local_code:
            Local replica of the spreading code for ONE symbol, length Ns.
            (In this simulator: E1B chips in ±1.)
        samples_per_symbol:
            Number of samples per symbol (Ns), e.g., 4092 for Galileo E1-B.
        return_debug:
            If True, also return the full bhat_running array.

        Returns
        -------
        spoofed:
            Spoofer synthesized replay signal, same length as rx_samples.
        bhat_final:
            Final per-symbol sign estimate (length Nsym), taken as the last
            running decision within each symbol.
        lock_idx:
            Index within each symbol where lock is achieved (length Nsym).
            - If lock_gamma <= 0: lock_idx = 0 for all symbols.
            - If never locks: lock_idx = Ns (meaning "no lock").
        bhat_running: (only if return_debug=True)
            Running sign decision per sample, shape (Nsym, Ns).
        """
        rx_samples = np.asarray(rx_samples, dtype=float)
        local_code = np.asarray(local_code, dtype=float)

        if local_code.size != samples_per_symbol:
            raise ValueError("local_code must have length equal to samples_per_symbol.")

        n_symbols = rx_samples.size // samples_per_symbol
        if n_symbols * samples_per_symbol != rx_samples.size:
            raise ValueError("rx_samples length must be a multiple of samples_per_symbol.")

        rx_mat = rx_samples.reshape(n_symbols, samples_per_symbol)

        # Per-sample correlation contribution: r[n] * c[n]
        prod = rx_mat * local_code[None, :]

        # Running correlation S[n] = sum_{i=0..n} r[i] * c[i]
        S = np.cumsum(prod, axis=1)

        if self.cfg.lock_gamma <= 0.0:
            # Decide from the first sample (classic running sign).
            bhat_running = self._sign_pm1(S)
            lock_idx = np.zeros(n_symbols, dtype=int)

        else:
            # Lock heuristic: |S[n]| >= gamma * sqrt(n), with n starting at 1.
            n = np.arange(1, samples_per_symbol + 1, dtype=float)
            thr = self.cfg.lock_gamma * np.sqrt(n)[None, :]  # shape (1, Ns)
            locked = np.abs(S) >= thr

            lock_idx = np.full(n_symbols, samples_per_symbol, dtype=int)
            bhat_running = np.empty_like(S, dtype=float)

            # Pre-lock sequence generator (symbol-local)
            if self.cfg.prelock_mode == "hold":
                pre = np.ones(samples_per_symbol, dtype=float)
            elif self.cfg.prelock_mode == "random":
                pre = self.rng.choice([-1.0, 1.0], size=samples_per_symbol)
            else:  # "zero"
                pre = np.zeros(samples_per_symbol, dtype=float)

            for k in range(n_symbols):
                lk = int(np.argmax(locked[k])) if np.any(locked[k]) else samples_per_symbol
                lock_idx[k] = lk

                if lk >= samples_per_symbol:
                    # Never locked: use pre-lock behavior for the full symbol.
                    bhat_running[k, :] = pre
                else:
                    # Before lock
                    bhat_running[k, :lk] = pre[:lk]
                    # After lock: sign decision based on running correlation S
                    bhat_running[k, lk:] = self._sign_pm1(S[k, lk:])

        # Spoofer output: s[n] = b_hat_running[n] * c[n]
        spoofed_mat = bhat_running * local_code[None, :]

        # Final per-symbol estimate: last running decision (robust and explicit).
        bhat_final = bhat_running[:, -1].copy()

        if return_debug:
            return spoofed_mat.reshape(-1), bhat_final, lock_idx, bhat_running

        return spoofed_mat.reshape(-1), bhat_final, lock_idx
