"""
main.py — Simple spoofing simulator over the Galileo E1-B code

This script supports two main workflows:

(1) Single-symbol illustration (one E1-B symbol, 4092 chips):
    Produces a 5-panel figure:
      1) Authentic transmitted symbol (TX)
      2) Spoofer input (TX + AWGN at spoofer C/N0)
      3) Spoofer output (spoofed symbol)
      4) Receiver cumulative correlation
      5) Receiver normalized cumulative correlation

(2) Multi-symbol statistics (E1-B code repeated over many symbols):
    - Applies spoofing + AWGN
    - Computes indicator diagnostics (spoofer BER, stabilization index stats, partial-corr features)
    - Plots distributions, scatter, and metric convergence
    - Selects a “representative” symbol and can plot it using the SAME run arrays
      (no re-simulation, ensuring consistency between statistics and the plotted example).

Notes:
- Representative-symbol plotting uses the SAME run arrays (no re-simulation).
- stabilization_idx is O(Ns) per symbol.
- slice_symbol() and plot_symbol_from_run() are pure visualization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from spoofer import Spoofer, SpooferCfg
from awgn_channel import AWGNChannel
from partial_correlator import PartialCorrelator
from e1b_code_helpers import e1b_hex_to_chips


# =============================================================================
# 1) Simulation configuration
# =============================================================================

# Paste here the E1-B code in hex as provided by the ICD (Annex C).
E1B_CODE_HEX = """
F5D710130573541B9DBD4FD9E9B20A0D59D144C54BC7935539D2E75810FB51E494093A0A19DD79C70C5A98E5657AA578097777E86BCC4651CC72F2F974DC766E07AEA3D0B557EF42FF57E6A58E805358CE9257669133B18F80FDBDFB38C5524C7FB1DE079842482990DF58F72321D9201F8979EAB159B2679C9E95AA6D53456C0DF75C2B4316D1E2309216882854253A1FA60CA2C94ECE013E2A8C943341E7D9E5A8464B3AD407E0AE465C3E3DD1BE60A8C3D50F831536401E776BE02A6042FC4A27AF653F0CFC4D4D013F115310788D68CAEAD3ECCCC5330587EB3C22A1459FC8E6FCCE9CDE849A5205E70C6D66D125814D698DD0EEBFEAE52CC65C5C84EEDF207379000E169D318426516AC5D1C31F2E18A65E07AE6E33FDD724B13098B3A444688389EFBBB5EEAB588742BB083B679D42FB26FF77919EAB21DE0389D9997498F967AE05AF0F4C7E177416E18C4D5E6987ED3590690AD127D872F14A8F4903A12329732A9768F82F295BEE391879293E3A97D51435A7F03ED7FBE275F102A83202DC3DE94AF4C712E9D006D182693E9632933E6EB773880CF147B922E74539E4582F79E39723B4C80E42EDCE4C08A8D02221BAE6D17734817D5B531C0D3C1AE723911F3FFF6AAC02E97FEA69E376AF4761E6451CA61FDB2F9187642EFCD63A09AAB680770C1593EEDD4FF4293BFFD6DD2C3367E85B14A654C834B6699421A
"""

# E1-B symbol duration: 4 ms -> 4092 chips at chip rate (1 sample/chip in this simulator).
T_SYMBOL_MS = 4.0

# Spoofer input C/N0 (controls attacker estimation quality).
CN0_SPOOFER_DBHZ = 30.0

# Victim receiver C/N0 (final channel after spoofing).
CN0_RX_DBHZ = 40.0

# Partial correlator window fraction for early/late partial correlations.
WINDOW_FRACTION_CORR = 0.125  # ~0.5 ms over a 4 ms symbol

# Number of symbols for the multi-symbol statistics run.
NUM_SYMBOLS_STATS = 1000

# Deterministic seeds (keep them explicit and centralized).
SEED_BITS_SINGLE = 50
SEED_BITS_STATS = 11
SEED_CH_SPOOFER_SINGLE = 123
SEED_SPOOFER_SINGLE = 42
SEED_CH_RX_SINGLE = 200
SEED_CH_SPOOFER_STATS = 100
SEED_SPOOFER_STATS = 101
SEED_CH_RX_STATS = 200

# Stabilization detection parameter: require at least this many remaining samples.
STAB_MIN_RUN = 32


# =============================================================================
# 2) Small utilities
# =============================================================================

def save_det_rows_csv(path: str, det_rows: list[dict[str, Any]]) -> None:
    """Save a list of dict rows to CSV with header inferred from the first row."""
    if not det_rows:
        raise ValueError("No rows to save.")

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        w.writeheader()
        w.writerows(det_rows)


def slice_symbol(x: np.ndarray, k: int, samples_per_symbol: int) -> np.ndarray:
    """Return the sample block corresponding to symbol index k."""
    x = np.asarray(x)
    start = k * samples_per_symbol
    end = start + samples_per_symbol
    return x[start:end]


# =============================================================================
# 3) Signal construction
# =============================================================================

def build_signals(num_symbols: int, seed_bits: int = 1):
    """
    Build an authentic signal consisting of repeated E1-B code symbols with random sign bits.

    Returns:
      e1b_chips            (Ns,)       code in +/-1 at chip rate (Ns = 4092)
      local_code_samples   (num_symbols*Ns,) repeated code-only replica
      tx_samples           (num_symbols*Ns,) transmitted samples: code * per-symbol bit
      samples_per_symbol   int         Ns
      chip_rate_hz         float       Ns / T_symbol
      bits                 (num_symbols,) per-symbol bits in {+1, -1}
    """
    # 1) E1B code mapped to +/-1 (4092 chips)
    e1b_chips = e1b_hex_to_chips(E1B_CODE_HEX)  # shape: (4092,)
    samples_per_symbol = int(e1b_chips.size)

    # 2) Unpredictable symbol sign bits per symbol (+/-1)
    rng = np.random.default_rng(seed_bits)
    bits = rng.choice([-1.0, 1.0], size=num_symbols)

    # 3) Local replica (code only, repeated)
    local_code_matrix = np.tile(e1b_chips, (num_symbols, 1))
    local_code_samples = local_code_matrix.reshape(-1)

    # 4) Transmitted authentic signal (code * bit)
    tx_matrix = local_code_matrix * bits[:, None]
    tx_samples = tx_matrix.reshape(-1)

    # 5) Chip rate
    t_symbol_s = T_SYMBOL_MS * 1e-3
    chip_rate_hz = samples_per_symbol / t_symbol_s

    return (e1b_chips, local_code_samples, tx_samples,
            samples_per_symbol, chip_rate_hz, bits)


# =============================================================================
# 4) Indicators and helper metrics
# =============================================================================

def summarize_lock_idx(lock_idx: np.ndarray, samples_per_symbol: int) -> dict[str, float]:
    """
    Summarize stabilization indices (in samples/chips).

    Convention:
      - lock_idx == Ns means "never stabilized"
      - valid indices are < Ns
    """
    lock_idx = np.asarray(lock_idx, dtype=int)
    never_lock = float(np.mean(lock_idx >= samples_per_symbol))
    valid = lock_idx[lock_idx < samples_per_symbol]

    return {
        "lock_never_frac": never_lock,
        "lock_mean": float(np.mean(valid)) if valid.size else float("nan"),
        "lock_median": float(np.median(valid)) if valid.size else float("nan"),
        "lock_p10": float(np.percentile(valid, 10)) if valid.size else float("nan"),
        "lock_p90": float(np.percentile(valid, 90)) if valid.size else float("nan"),
    }


def spoofer_ber(bits_true: np.ndarray, bhat_final: np.ndarray) -> float:
    """
    Compute spoofer BER: compare final hard decisions vs. true bits.

    Note:
      - Any non-negative decision is mapped to +1, negative to -1.
      - If a future prelock_mode produces zeros, they map to +1 here.
    """
    bits_true = np.asarray(bits_true, dtype=float)
    bhat_final = np.asarray(bhat_final, dtype=float)
    bhat_hard = np.where(bhat_final >= 0, 1.0, -1.0)
    return float(np.mean(bhat_hard != bits_true))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Safe Pearson correlation returning NaN for degenerate inputs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_symbol_features(Bbeg: np.ndarray, Bend: np.ndarray) -> dict[str, float]:
    """Compute basic magnitude-based features from partial correlations."""
    Bbeg = np.asarray(Bbeg)
    Bend = np.asarray(Bend)

    abs_Bbeg = np.abs(Bbeg)
    abs_Bend = np.abs(Bend)
    abs_dB = np.abs(Bend - Bbeg)

    return {
        "abs_Bbeg_mean": float(np.mean(abs_Bbeg)),
        "abs_Bend_mean": float(np.mean(abs_Bend)),
        "abs_dB_mean": float(np.mean(abs_dB)),
        "abs_Bbeg_p90": float(np.percentile(abs_Bbeg, 90)),
        "abs_Bend_p90": float(np.percentile(abs_Bend, 90)),
        "abs_dB_p90": float(np.percentile(abs_dB, 90)),
    }


def stabilization_idx(bhat_running: np.ndarray, min_run: int = STAB_MIN_RUN) -> np.ndarray:
    """
    O(Ns) per symbol stabilization index.

    For each symbol, returns the first index n such that:
      - from n to the end, the running decision matches the final decision, and
      - at least min_run samples remain (robustness constraint).
    If not satisfied, returns Ns (meaning: no stabilization).
    """
    bh = np.asarray(bhat_running, dtype=float)
    if bh.ndim != 2:
        raise ValueError(f"bhat_running must be 2D (Nsym, Ns), got shape={bh.shape}")

    Nsym, Ns = bh.shape
    final = bh[:, -1]
    idx = np.full(Nsym, Ns, dtype=int)

    for k in range(Nsym):
        mism = np.flatnonzero(bh[k] != final[k])
        n0 = 0 if mism.size == 0 else int(mism[-1] + 1)  # 1 + last mismatch
        if n0 <= Ns - min_run:
            idx[k] = n0

    return idx


def print_indicator_report(title: str,
                           ber: float,
                           stab_stats: dict[str, float],
                           frac_stable_early: float,
                           sym_feats: dict[str, float],
                           corr_stats: dict[str, float]) -> None:
    """Print a compact, human-readable diagnostic report."""
    print(f"\n=== {title} ===")
    print(f"  Spoofer BER (bhat_final vs. bits): {ber:.4f}")

    print("  Stabilization index stats (chips):")
    print(f"    never-stable frac: {stab_stats['lock_never_frac']:.3f}")
    print(f"    mean/median      : {stab_stats['lock_mean']:.1f} / {stab_stats['lock_median']:.1f}")
    print(f"    p10 / p90        : {stab_stats['lock_p10']:.1f} / {stab_stats['lock_p90']:.1f}")
    print(f"    frac stable within early window: {frac_stable_early:.3f}")

    print("  Partial-correlation magnitudes (per symbol):")
    print(f"    mean |Bbeg|      : {sym_feats['abs_Bbeg_mean']:.3f}")
    print(f"    mean |Bend|      : {sym_feats['abs_Bend_mean']:.3f}")
    print(f"    mean |Bend-Bbeg| : {sym_feats['abs_dB_mean']:.3f}")

    print("  Correlations with stabilization_idx (Pearson):")
    print(f"    corr(stab, |Bbeg|)      : {corr_stats['corr_stab_absBbeg']:.3f}")
    print(f"    corr(stab, |Bend|)      : {corr_stats['corr_stab_absBend']:.3f}")
    print(f"    corr(stab, |Bend-Bbeg|) : {corr_stats['corr_stab_absdB']:.3f}")


def pick_representative_symbol(stab_idx: np.ndarray,
                               samples_per_symbol: int,
                               window_fraction_corr: float,
                               bits: np.ndarray,
                               desired_bit: float = +1.0) -> int:
    """
    Pick a representative symbol whose stabilization time is closest to the early window length,
    optionally restricted to a desired bit sign (+1 or -1).
    """
    wlen = int(samples_per_symbol * window_fraction_corr)
    target = wlen

    stab_idx = np.asarray(stab_idx, dtype=int)
    bits = np.asarray(bits, dtype=float)

    valid = np.where((stab_idx < samples_per_symbol) & (bits == desired_bit))[0]
    if valid.size == 0:
        valid = np.where(stab_idx < samples_per_symbol)[0]
        if valid.size == 0:
            return 0

    return int(valid[np.argmin(np.abs(stab_idx[valid] - target))])


# =============================================================================
# 5) Scenario A: Single-symbol illustrative figure
# =============================================================================

def run_single_symbol_plot() -> None:
    num_symbols_single = 1

    (e1b_chips, _local_code_samples, tx_samples,
     samples_per_symbol, chip_rate_hz, bits) = build_signals(num_symbols_single, seed_bits=SEED_BITS_SINGLE)

    tx_sym = tx_samples[:samples_per_symbol]
    b0 = float(bits[0])

    # (1) Spoofer input: TX + AWGN at CN0_SPOOFER_DBHZ
    chan_spoofer = AWGNChannel(cn0_db_hz=CN0_SPOOFER_DBHZ, chip_rate_hz=chip_rate_hz, seed=SEED_CH_SPOOFER_SINGLE)
    rx_spoofer_sym = chan_spoofer.add_noise(tx_sym)

    # (2) Spoofer processing (debug enabled for bhat_running)
    spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=SEED_SPOOFER_SINGLE)
    spoofed_sym, _bhat_final, _lock_idx, bhat_running = spoofer.spoof_from_rx(
        rx_samples=rx_spoofer_sym,
        local_code=e1b_chips,
        samples_per_symbol=samples_per_symbol,
        return_debug=True,
    )

    # (3) Victim receiver input: spoofed + AWGN at CN0_RX_DBHZ
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=SEED_CH_RX_SINGLE)
    rx_receiver_sym = chan_rx.add_noise(spoofed_sym)

    # Verified local replica for correlation (simulation-only: receiver "knows" b0)
    local_replica_sym = b0 * e1b_chips

    prod = local_replica_sym * rx_receiver_sym
    corr_cum = np.cumsum(prod)
    n = np.arange(1, samples_per_symbol + 1, dtype=float)
    corr_norm = corr_cum / n

    t_ms = np.linspace(0.0, T_SYMBOL_MS, samples_per_symbol, endpoint=False)

    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)
    t_early_end = (wlen / samples_per_symbol) * T_SYMBOL_MS

    bh = np.asarray(bhat_running, dtype=float)
    if bh.ndim == 1:
        bh = bh.reshape(1, -1)
    stab = stabilization_idx(bh, min_run=STAB_MIN_RUN)[0]
    stab_valid = (stab < samples_per_symbol)
    t_stab_ms = (stab / samples_per_symbol) * T_SYMBOL_MS if stab_valid else None

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(11, 10), sharex=True)

    def add_vlines(ax):
        ax.axvline(t_early_end, linestyle="--", linewidth=0.8)
        if stab_valid and t_stab_ms is not None:
            ax.axvline(t_stab_ms, linestyle=":", linewidth=1.2)

    ax1.step(t_ms, tx_sym, where="post")
    ax1.set_ylabel("Amp")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title("1) Authentic transmitted symbol (TX): b · c[n]")
    ax1.grid(True)
    add_vlines(ax1)

    ax2.step(t_ms, rx_spoofer_sym, where="post")
    ax2.set_ylabel("Amp")
    ax2.set_title(f"2) Spoofer input: TX + AWGN (C/N0={CN0_SPOOFER_DBHZ:.1f} dB-Hz)")
    ax2.grid(True)
    add_vlines(ax2)

    ax3.step(t_ms, spoofed_sym, where="post")
    ax3.set_ylabel("Amp")
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_title("3) Spoofer output (spoofed symbol)")
    ax3.grid(True)
    add_vlines(ax3)

    ax4.plot(t_ms, corr_cum)
    ax4.set_ylabel("Sum corr")
    ax4.set_title("4) Receiver cumulative correlation: Σ local[i]·rx[i]")
    ax4.grid(True)
    add_vlines(ax4)

    ax5.plot(t_ms, corr_norm)
    ax5.set_xlabel("Time within symbol (ms)")
    ax5.set_ylabel("Mean corr")
    ax5.set_title("5) Receiver normalized cumulative correlation: (Σ local[i]·rx[i]) / N")
    ax5.grid(True)
    add_vlines(ax5)

    if stab_valid and t_stab_ms is not None:
        ax5.text(
            0.01, 0.05,
            f"stabilization_idx={stab} chips ({t_stab_ms:.3f} ms)",
            transform=ax5.transAxes,
        )

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6) Plots for multi-symbol runs
# =============================================================================

def plot_stabilization_distribution(stab_idx: np.ndarray, samples_per_symbol: int, window_fraction_corr: float) -> None:
    stab_idx = np.asarray(stab_idx, dtype=int)
    wlen = int(samples_per_symbol * window_fraction_corr)

    valid = stab_idx[stab_idx < samples_per_symbol]
    if valid.size == 0:
        print("No valid stabilization indices to plot.")
        return

    plt.figure(figsize=(9, 4))
    plt.hist(valid, bins=60)
    plt.axvline(wlen, linestyle="--", linewidth=1.2, label=f"early window = {wlen} chips")
    plt.xlabel("stabilization_idx (chips)")
    plt.ylabel("count")
    plt.title("Spoofer stabilization time distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    xs = np.sort(valid)
    ys = np.arange(1, xs.size + 1) / xs.size

    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys)
    plt.axvline(wlen, linestyle="--", linewidth=1.2, label=f"early window = {wlen} chips")
    plt.xlabel("stabilization_idx (chips)")
    plt.ylabel("CDF")
    plt.title("CDF of spoofer stabilization time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter_stab_vs_gap(stab_idx: np.ndarray, Bbeg: np.ndarray, Bend: np.ndarray) -> None:
    stab = np.asarray(stab_idx, dtype=float)
    gap = np.abs(np.asarray(Bend) - np.asarray(Bbeg))

    plt.figure(figsize=(8, 4))
    plt.scatter(stab, gap, s=8, alpha=0.5)
    plt.xlabel("stabilization_idx (chips)")
    plt.ylabel("|Bend - Bbeg|")
    plt.title("Per-symbol partial-corr gap vs. spoofer stabilization time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_beg_end_distributions(Bbeg: np.ndarray, Bend: np.ndarray) -> None:
    absBbeg = np.abs(np.asarray(Bbeg))
    absBend = np.abs(np.asarray(Bend))

    plt.figure(figsize=(9, 4))
    plt.hist(absBbeg, bins=60, alpha=0.6, label="|Bbeg|")
    plt.hist(absBend, bins=60, alpha=0.6, label="|Bend|")
    plt.xlabel("magnitude")
    plt.ylabel("count")
    plt.title("Distributions of partial correlations (magnitudes)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metrics_convergence(correlator: PartialCorrelator,
                             local_code_samples: np.ndarray,
                             tx_samples: np.ndarray,
                             noisy_recv_attack: np.ndarray,
                             samples_per_symbol: int,
                             cn0_rx_dbhz: float,
                             chip_rate_hz: float) -> None:
    """
    Plot running convergence of R1/R2/R3 over the first N symbols for:
      - Attack run (noisy_recv_attack)
      - Baseline run (authentic tx + AWGN at same CN0)
    """
    chan_rx = AWGNChannel(cn0_db_hz=cn0_rx_dbhz, chip_rate_hz=chip_rate_hz, seed=999)
    noisy_recv_base = chan_rx.add_noise(tx_samples)

    Bbeg_a, Bend_a, _ = correlator.partial_correlations(local_code_samples, noisy_recv_attack, samples_per_symbol)
    Bbeg_b, Bend_b, _ = correlator.partial_correlations(local_code_samples, noisy_recv_base, samples_per_symbol)

    N = int(Bbeg_a.size)
    x = np.arange(1, N + 1)

    R1a = np.zeros(N); R2a = np.zeros(N); R3a = np.zeros(N)
    R1b = np.zeros(N); R2b = np.zeros(N); R3b = np.zeros(N)

    for k in range(1, N + 1):
        ma = correlator.r_metrics(Bbeg_a[:k], Bend_a[:k])
        mb = correlator.r_metrics(Bbeg_b[:k], Bend_b[:k])
        R1a[k - 1], R2a[k - 1], R3a[k - 1] = ma["R1"], ma["R2"], ma["R3"]
        R1b[k - 1], R2b[k - 1], R3b[k - 1] = mb["R1"], mb["R2"], mb["R3"]

    plt.figure(figsize=(9, 4))
    plt.plot(x, R2a, label="R2 attack")
    plt.plot(x, R2b, label="R2 baseline")
    plt.xlabel("Number of symbols (N)")
    plt.ylabel("R2")
    plt.title("Convergence of R2 with N (attack vs. baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(x, R3a, label="R3 attack")
    plt.plot(x, R3b, label="R3 baseline")
    plt.xlabel("Number of symbols (N)")
    plt.ylabel("R3")
    plt.title("Convergence of R3 with N (attack vs. baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(x, R1a, label="R1 attack")
    plt.plot(x, R1b, label="R1 baseline")
    plt.xlabel("Number of symbols (N)")
    plt.ylabel("R1")
    plt.title("Convergence of R1 with N (attack vs. baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7) Representative symbol plotting using the SAME run arrays
# =============================================================================

def plot_symbol_from_run(e1b_chips: np.ndarray,
                         tx_samples: np.ndarray,
                         rx_spoofer: np.ndarray,
                         spoofed: np.ndarray,
                         rx_receiver: np.ndarray,
                         bits: np.ndarray,
                         bhat_running: np.ndarray,
                         samples_per_symbol: int,
                         k: int) -> None:
    """
    Plot symbol k using arrays already generated in run_many_symbols_stats().

    This function does NOT re-simulate noise nor the spoofer; it only visualizes one symbol
    from the existing run (source-of-truth arrays).
    """
    tx_sym = slice_symbol(tx_samples, k, samples_per_symbol)
    rx_spoofer_sym = slice_symbol(rx_spoofer, k, samples_per_symbol)
    spoofed_sym = slice_symbol(spoofed, k, samples_per_symbol)
    rx_receiver_sym = slice_symbol(rx_receiver, k, samples_per_symbol)

    if len(e1b_chips) != samples_per_symbol:
        raise ValueError(
            f"Length mismatch: len(e1b_chips)={len(e1b_chips)} but samples_per_symbol={samples_per_symbol}."
        )

    # Map bit to {+1, -1} robustly
    b0_raw = float(bits[k])
    if b0_raw in (0.0, 1.0):
        b0 = 1.0 if b0_raw == 1.0 else -1.0
    else:
        b0 = 1.0 if b0_raw >= 0.0 else -1.0

    local_replica_sym = b0 * e1b_chips

    prod = local_replica_sym * rx_receiver_sym
    corr_cum = np.cumsum(prod)
    n = np.arange(1, samples_per_symbol + 1, dtype=float)
    corr_norm = corr_cum / n

    t_ms = np.linspace(0.0, T_SYMBOL_MS, samples_per_symbol, endpoint=False)

    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)
    t_early_end = (wlen / samples_per_symbol) * T_SYMBOL_MS

    bh = np.asarray(bhat_running, dtype=float)
    if bh.ndim != 2:
        raise ValueError(f"bhat_running must be 2D (Nsym, Ns), got shape={bh.shape}")
    if not (0 <= k < bh.shape[0]):
        raise IndexError(f"k={k} out of range for bhat_running with shape={bh.shape}")

    stab = stabilization_idx(bh[k:k + 1, :], min_run=STAB_MIN_RUN)[0]
    stab_valid = (0 <= stab < samples_per_symbol)
    t_stab_ms = (stab / samples_per_symbol) * T_SYMBOL_MS if stab_valid else None

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(11, 10), sharex=True)

    def add_vlines(ax):
        ax.axvline(t_early_end, linestyle="--", linewidth=0.8)
        if stab_valid and t_stab_ms is not None:
            ax.axvline(t_stab_ms, linestyle=":", linewidth=1.2)

    ax1.step(t_ms, tx_sym, where="post")
    ax1.set_ylabel("Amp")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title(f"1) Transmitted symbol (TX): b·c[n] (b={b0:+.0f})")
    ax1.grid(True)
    add_vlines(ax1)

    ax2.step(t_ms, rx_spoofer_sym, where="post")
    ax2.set_ylabel("Amp")
    ax2.set_title(f"2) Spoofer input (C/N0={CN0_SPOOFER_DBHZ:.1f} dB-Hz)")
    ax2.grid(True)
    add_vlines(ax2)

    ax3.step(t_ms, spoofed_sym, where="post")
    ax3.set_ylabel("Amp")
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_title("3) Spoofed signal (attacker output)")
    ax3.grid(True)
    add_vlines(ax3)

    ax4.plot(t_ms, corr_cum)
    ax4.set_ylabel("Cumulative corr")
    ax4.set_title("4) Cumulative correlation: Σ local[i]·rx[i]")
    ax4.grid(True)
    add_vlines(ax4)

    ax5.plot(t_ms, corr_norm)
    ax5.set_title("5) Normalized cumulative correlation: (Σ local[i]·rx[i]) / N")
    ax5.set_xlabel("Time within symbol (ms)")
    ax5.set_ylabel("Mean corr")
    ax5.grid(True)
    add_vlines(ax5)

    if stab_valid and t_stab_ms is not None:
        ax5.text(
            0.01, 0.05,
            f"stabilization_idx={stab} samples ({t_stab_ms:.3f} ms)",
            transform=ax5.transAxes,
        )

    plt.tight_layout()
    plt.show()


# =============================================================================
# 8) Scenario B: Multi-symbol run (stats + representative selection)
# =============================================================================

def run_many_symbols_stats() -> dict[str, Any]:
    (e1b_chips, local_code_samples, tx_samples,
     samples_per_symbol, chip_rate_hz, bits) = build_signals(NUM_SYMBOLS_STATS, seed_bits=SEED_BITS_STATS)

    # Spoofer input channel
    chan_spoofer = AWGNChannel(cn0_db_hz=CN0_SPOOFER_DBHZ, chip_rate_hz=chip_rate_hz, seed=SEED_CH_SPOOFER_STATS)
    rx_spoofer = chan_spoofer.add_noise(tx_samples)

    # Spoofer processing (debug enabled for bhat_running)
    spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=SEED_SPOOFER_STATS)
    spoofed, bhat_final, _lock_idx, bhat_running = spoofer.spoof_from_rx(
        rx_samples=rx_spoofer,
        local_code=e1b_chips,
        samples_per_symbol=samples_per_symbol,
        return_debug=True,
    )

    # Stabilization indices (chips)
    stab_idx = stabilization_idx(bhat_running, min_run=STAB_MIN_RUN)

    # Representative symbol selection (+1 and -1)
    k_pos = pick_representative_symbol(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR, bits, desired_bit=+1.0)
    k_neg = pick_representative_symbol(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR, bits, desired_bit=-1.0)

    print(f"k_pos={k_pos} stab={stab_idx[k_pos]} b={bits[k_pos]:+.0f}")
    print(f"k_neg={k_neg} stab={stab_idx[k_neg]} b={bits[k_neg]:+.0f}")

    # Victim receiver channel
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=SEED_CH_RX_STATS)
    rx_receiver = chan_rx.add_noise(spoofed)

    correlator = PartialCorrelator(window_fraction=WINDOW_FRACTION_CORR)
    Bbeg, Bend, bhat_corr = correlator.partial_correlations(
        local_code_samples,
        rx_receiver,
        samples_per_symbol,
    )

    ber = spoofer_ber(bits, bhat_final)
    sym_feats = compute_symbol_features(Bbeg, Bend)

    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)
    frac_stable_early = float(np.mean(stab_idx <= wlen))

    corr_stats = {
        "corr_stab_absBbeg": pearson_corr(stab_idx, np.abs(Bbeg)),
        "corr_stab_absBend": pearson_corr(stab_idx, np.abs(Bend)),
        "corr_stab_absdB": pearson_corr(stab_idx, np.abs(Bend - Bbeg)),
    }

    stab_stats = summarize_lock_idx(stab_idx, samples_per_symbol)

    print_indicator_report(
        title=f"Indicators (CN0_spoofer={CN0_SPOOFER_DBHZ:.1f} dB-Hz, CN0_rx={CN0_RX_DBHZ:.1f} dB-Hz)",
        ber=ber,
        stab_stats=stab_stats,
        frac_stable_early=frac_stable_early,
        sym_feats=sym_feats,
        corr_stats=corr_stats,
    )

    # Representative plots (multi-symbol)
    plot_stabilization_distribution(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR)
    plot_scatter_stab_vs_gap(stab_idx, Bbeg, Bend)
    plot_beg_end_distributions(Bbeg, Bend)

    plot_metrics_convergence(
        correlator=correlator,
        local_code_samples=local_code_samples,
        tx_samples=tx_samples,
        noisy_recv_attack=rx_receiver,
        samples_per_symbol=samples_per_symbol,
        cn0_rx_dbhz=CN0_RX_DBHZ,
        chip_rate_hz=chip_rate_hz,
    )

    # Return the run "source of truth" arrays for consistent post-visualization
    return {
        "e1b_chips": e1b_chips,
        "local_code_samples": local_code_samples,
        "tx_samples": tx_samples,
        "rx_spoofer": rx_spoofer,
        "spoofed": spoofed,
        "rx_receiver": rx_receiver,
        "bits": bits,
        "bhat_final": bhat_final,
        "bhat_running": bhat_running,
        "stab_idx": stab_idx,
        "Bbeg": Bbeg,
        "Bend": Bend,
        "bhat_corr": bhat_corr,
        "k_pos": k_pos,
        "k_neg": k_neg,
        "samples_per_symbol": samples_per_symbol,
        "chip_rate_hz": chip_rate_hz,
    }


# =============================================================================
# 9) Optional sweeps
# =============================================================================

def run_cn0_spoofer_sweep(cn0_list_dbhz: list[float]) -> None:
    """
    Sweep the spoofer input C/N0 while keeping the receiver C/N0 fixed (CN0_RX_DBHZ).
    This is useful for attacker sensitivity studies.
    """
    (e1b_chips, local_code_samples, tx_samples,
     samples_per_symbol, chip_rate_hz, bits) = build_signals(NUM_SYMBOLS_STATS, seed_bits=SEED_BITS_STATS)

    correlator = PartialCorrelator(window_fraction=WINDOW_FRACTION_CORR)
    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)

    # Baseline: victim receiver sees authentic signal with receiver noise
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=SEED_CH_RX_STATS)
    baseline_recv = chan_rx.add_noise(tx_samples)

    Bbeg0, Bend0, _ = correlator.partial_correlations(local_code_samples, baseline_recv, samples_per_symbol)
    met0 = correlator.r_metrics(Bbeg0, Bend0)

    print("\n=== CN0 sweep (spoofer input) ===")
    print("CN0_spoofer | BER   | stab_mean | stab_p90 | frac_stab_early | R1     | R2     | R3")
    print("-" * 98)
    print(f"{'BASELINE':>10} | {0.000:5.3f} | {'-':>9} | {'-':>8} | {'-':>14} |"
          f" {met0['R1']:6.3f} | {met0['R2']:6.3f} | {met0['R3']:6.1f}")

    for cn0s in cn0_list_dbhz:
        chan_spoofer = AWGNChannel(cn0_db_hz=cn0s, chip_rate_hz=chip_rate_hz, seed=SEED_CH_SPOOFER_STATS)
        rx_spoofer = chan_spoofer.add_noise(tx_samples)

        spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=SEED_SPOOFER_STATS)
        spoofed, bhat_final, _lock_idx, bhat_running = spoofer.spoof_from_rx(
            rx_samples=rx_spoofer,
            local_code=e1b_chips,
            samples_per_symbol=samples_per_symbol,
            return_debug=True,
        )

        noisy_recv = chan_rx.add_noise(spoofed)

        Bbeg, Bend, _ = correlator.partial_correlations(local_code_samples, noisy_recv, samples_per_symbol)
        met = correlator.r_metrics(Bbeg, Bend)

        ber = spoofer_ber(bits, bhat_final)
        stab_idx = stabilization_idx(bhat_running, min_run=STAB_MIN_RUN)
        stab_stats = summarize_lock_idx(stab_idx, samples_per_symbol)
        frac_stable_early = float(np.mean(stab_idx <= wlen))

        print(f"{cn0s:10.1f} | {ber:5.3f} | {stab_stats['lock_mean']:9.1f} | {stab_stats['lock_p90']:8.1f} |"
              f" {frac_stable_early:14.3f} | {met['R1']:6.3f} | {met['R2']:6.3f} | {met['R3']:6.1f}")


# =============================================================================
# 10) Main entry point
# =============================================================================

if __name__ == "__main__":
    # Optional: a "textbook" single-symbol figure
    # run_single_symbol_plot()

    # 1) Run multi-symbol stats and return the run arrays (source of truth)
    run = run_many_symbols_stats()

    # 2) Plot a representative symbol from the SAME run (no re-simulation)
    # plot_symbol_from_run(
    #     e1b_chips=run["e1b_chips"],
    #     tx_samples=run["tx_samples"],
    #     rx_spoofer=run["rx_spoofer"],
    #     spoofed=run["spoofed"],
    #     rx_receiver=run["rx_receiver"],
    #     bits=run["bits"],
    #     bhat_running=run["bhat_running"],
    #     samples_per_symbol=run["samples_per_symbol"],
    #     k=int(run["k_pos"]),
    # )

    # 3) Optional CN0 sweep (development utility)
    # run_cn0_spoofer_sweep([30.0, 35.0, 40.0, 45.0, 50.0])
