"""
spoofer.py

Spoofer that estimates each symbol b_k ∈ {±1} from the received authentic signal
by correlating against a local code replica (sample-by-sample / running correlation),
and then replays a synthesized signal s[n] = b_hat[n] * c[n].

This makes the spoofer performance depend ONLY on the quality of what it receives
(e.g., C/N0 at the spoofer input), not on an artificial error probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpooferCfg:
    """
    lock_gamma:
        If > 0, we define a simple lock threshold proportional to sqrt(n):
            lock when |S[n]| >= lock_gamma * sqrt(n)
        This is a heuristic to avoid noisy sign flips when S[n] is close to 0.

        If = 0, the spoofer decides from the first sample: b_hat[n]=sign(S[n]).

    prelock_mode:
        What to output before lock is achieved (only used when lock_gamma>0):
          - "hold": keep last decided value (starts at +1)
          - "random": output random ±1 until lock
          - "zero": output 0 until lock (energy loss model)
    """
    lock_gamma: float = 0.0
    prelock_mode: str = "hold"  # {"hold","random","zero"}


class Spoofer:
    def __init__(self, cfg: SpooferCfg | None = None, seed: int | None = None):
        self.cfg = cfg if cfg is not None else SpooferCfg()
        if self.cfg.prelock_mode not in {"hold", "random", "zero"}:
            raise ValueError("prelock_mode must be one of: 'hold', 'random', 'zero'")
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sign_pm1(x: np.ndarray) -> np.ndarray:
        """Return sign mapped to {+1,-1} with 0 -> +1."""
        out = np.ones_like(x, dtype=float)
        out[x < 0] = -1.0
        return out

    def spoof_from_rx(
        self,
        rx_samples: np.ndarray,
        local_code: np.ndarray,
        samples_per_symbol: int,
        return_debug: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        rx_samples:
            What the spoofer receives (authentic signal + noise), length = Nsym * Ns.
        local_code:
            Local replica of the spreading code for ONE symbol, length Ns.
            (For your current simulator: E1B chips in ±1)
        samples_per_symbol:
            Ns (4092 in your case)

        Returns
        -------
        spoofed:
            Spoofer synthesized replay signal, same length as rx_samples.
        bhat_final:
            Final symbol estimate per symbol (length Nsym), using last sample decision.
        lock_idx:
            Index within each symbol where lock is achieved (length Nsym).
            If lock_gamma==0: lock_idx = 0 for all symbols.
            If never locks: lock_idx = Ns (meaning "no lock").
        """
        rx_samples = np.asarray(rx_samples, dtype=float)
        local_code = np.asarray(local_code, dtype=float)

        if local_code.size != samples_per_symbol:
            raise ValueError("local_code must have length samples_per_symbol")

        n_symbols = rx_samples.size // samples_per_symbol
        if n_symbols * samples_per_symbol != rx_samples.size:
            raise ValueError("rx_samples length must be multiple of samples_per_symbol")

        rx_mat = rx_samples.reshape(n_symbols, samples_per_symbol)

        # Correlation contribution per sample: r[n] * c[n]
        prod = rx_mat * local_code[None, :]

        # Running correlation S[n] = cumsum(prod)
        S = np.cumsum(prod, axis=1)

        if self.cfg.lock_gamma <= 0.0:
            # Decide from the first sample (classic running sign)
            bhat_running = self._sign_pm1(S)
            lock_idx = np.zeros(n_symbols, dtype=int)
        else:
            # Lock heuristic: |S[n]| >= gamma * sqrt(n+1)
            n = np.arange(1, samples_per_symbol + 1, dtype=float)
            thr = self.cfg.lock_gamma * np.sqrt(n)[None, :]  # shape (1, Ns)
            locked = np.abs(S) >= thr

            lock_idx = np.full(n_symbols, samples_per_symbol, dtype=int)
            bhat_running = np.empty_like(S, dtype=float)

            # prelock initial state
            if self.cfg.prelock_mode == "hold":
                pre = np.ones(samples_per_symbol, dtype=float)
            elif self.cfg.prelock_mode == "random":
                pre = self.rng.choice([-1.0, 1.0], size=samples_per_symbol)
            else:  # "zero"
                pre = np.zeros(samples_per_symbol, dtype=float)

            for k in range(n_symbols):
                lk = np.argmax(locked[k]) if np.any(locked[k]) else samples_per_symbol
                lock_idx[k] = lk

                if lk >= samples_per_symbol:
                    # never locked: use prelock mode for whole symbol
                    bhat_running[k, :] = pre
                else:
                    # before lock
                    bhat_running[k, :lk] = pre[:lk]
                    # after lock: sign decision based on S
                    bhat_running[k, lk:] = self._sign_pm1(S[k, lk:])

        # Spoofer output: s[n] = b_hat_running[n] * c[n]
        spoofed_mat = bhat_running * local_code[None, :]

        # Final per-symbol estimate = last running decision (or 0 if prelock_mode="zero" and no lock)
        bhat_final = spoofed_mat[:, -1] * local_code[-1]  # invert last chip spreading
        # But easier/cleaner: use last bhat_running value directly:
        bhat_final = bhat_running[:, -1].copy()

        if return_debug:
            return spoofed_mat.reshape(-1), bhat_final, lock_idx, bhat_running
        return spoofed_mat.reshape(-1), bhat_final, lock_idx
