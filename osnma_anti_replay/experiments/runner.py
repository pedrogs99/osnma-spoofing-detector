"""
experiments/runner.py — Monte Carlo experiment runner and post-processing utilities.

This module provides:
- Trial-level simulation for baseline (H0) and SCER-style replay/spoofer (H1)
- Grid sweeps over C/N0 and aggregation length Nb
- Flattening of nested trial records into CSV-friendly rows
- Summary statistics (means/percentiles)
- Threshold calibration and split-based detection evaluation
- Split-based ROC curve generation and CSV export

Conventions
-----------
- Each Monte Carlo trial generates `cfg.num_symbols_total` symbols.
- For each Nb in cfg.nb_list, metrics are computed on the first Nb symbols.
- Metrics are computed using PartialCorrelator on aligned partial correlations
  (bhat estimated from full-symbol correlation).
- Detection convention: declare "attack" if metric > gamma.

Reproducibility
---------------
Seeds are derived deterministically from:
- cfg.master_seed
- trial_index
- fixed offsets for signal bits, noise, and spoofer RNG streams
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from awgn_channel import AWGNChannel
from e1b_code_helpers import e1b_hex_to_chips
from partial_correlator import PartialCorrelator
from spoofer import Spoofer, SpooferCfg

from .configs import ExperimentConfig, SignalConfig, SpooferConfig, Scenario


# =============================================================================
# Signal generation utilities
# =============================================================================

def _build_trial_signals(sig_cfg: SignalConfig, num_symbols: int, seed_bits: int):
    """
    Build the authentic signal and a code-only local replica for one trial.

    Returns
    -------
    e1b_chips:
        (Ns,) spreading code in +/-1 (chip-rate).
    local_code_samples:
        (num_symbols*Ns,) repeated code-only replica (no data bits).
    tx_samples:
        (num_symbols*Ns,) transmitted authentic samples: bit * code.
    Ns:
        Samples per symbol (chips per symbol).
    chip_rate_hz:
        Chip rate in Hz (chips/s).
    bits:
        (num_symbols,) true per-symbol bits in {+1, -1}.
    """
    e1b_chips = e1b_hex_to_chips(sig_cfg.e1b_code_hex).astype(np.float32)
    Ns = int(e1b_chips.size)

    rng = np.random.default_rng(seed_bits)
    bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=num_symbols)

    local_code_matrix = np.tile(e1b_chips, (num_symbols, 1))
    local_code_samples = local_code_matrix.reshape(-1)

    tx_samples = (local_code_matrix * bits[:, None]).reshape(-1)

    t_symbol_s = float(sig_cfg.t_symbol_ms) * 1e-3
    chip_rate_hz = float(Ns) / t_symbol_s

    return e1b_chips, local_code_samples, tx_samples, Ns, chip_rate_hz, bits


def _ser_wipeoff(bits_true: np.ndarray, bhat: np.ndarray) -> float:
    """
    Symbol error rate (SER) of wipe-off decisions.

    Parameters
    ----------
    bits_true:
        True per-symbol bits in {+1, -1}.
    bhat:
        Estimated per-symbol decisions (real-valued). Non-negative maps to +1.

    Returns
    -------
    float
        SER = mean(bhat_hard != bits_true).
    """
    bits_true = np.asarray(bits_true, dtype=float)
    bhat = np.asarray(bhat, dtype=float)

    bhat_hard = np.where(bhat >= 0.0, 1.0, -1.0)
    return float(np.mean(bhat_hard != bits_true))


# =============================================================================
# Trial runner
# =============================================================================

def run_single_trial(
    scenario: Scenario,
    cn0_dbhz: float,
    nb_list: List[int],
    cfg: ExperimentConfig,
    sig_cfg: SignalConfig,
    sp_cfg: SpooferConfig,
    trial_index: int,
) -> Dict[str, Any]:
    """
    Run a single Monte Carlo trial for a given scenario and C/N0 point.

    Returns
    -------
    dict with keys:
      - cn0_dbhz, scenario, trial_index
      - metrics: list of dicts, one per Nb: {"Nb":..., "R1":..., "R2":..., "R3":...}
      - SER_wipeoff: SER computed using correlator bhat vs true bits
      - spoofer_BER: only for SCER (None for BASELINE)
      - stab_stats: reserved for future use (currently None)
    """
    # Deterministic seeds per trial (keep simple and explicit)
    base = int(cfg.master_seed + 1_000 * int(trial_index))
    seed_bits = base + 1
    seed_noise = base + 2
    seed_spoofer = base + int(sp_cfg.seed_offset) + 3

    # Build signals for this trial
    e1b_chips, local_code_samples, tx_samples, Ns, chip_rate_hz, bits = _build_trial_signals(
        sig_cfg=sig_cfg,
        num_symbols=int(cfg.num_symbols_total),
        seed_bits=seed_bits,
    )

    correlator = PartialCorrelator(window_fraction=float(cfg.window_fraction_corr))

    # Scenario simulation
    if scenario == Scenario.BASELINE:
        # Victim receiver sees authentic signal + AWGN at cn0_dbhz.
        chan_rx = AWGNChannel(cn0_db_hz=float(cn0_dbhz), chip_rate_hz=chip_rate_hz, seed=seed_noise)
        recv_samples = chan_rx.add_noise(tx_samples)

        spoofer_ber = None
        stab_stats = None

    elif scenario == Scenario.SCER:
        # Current implementation assumes same CN0 at spoofer input and at the victim receiver.
        cn0_spoofer = float(cn0_dbhz)
        cn0_rx = float(cn0_dbhz)

        # Spoofer input channel
        chan_spoofer = AWGNChannel(cn0_db_hz=cn0_spoofer, chip_rate_hz=chip_rate_hz, seed=seed_noise)
        rx_spoofer = chan_spoofer.add_noise(tx_samples)

        # Spoofer replay generation
        spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=float(sp_cfg.lock_gamma)), seed=seed_spoofer)
        spoofed, bhat_final, _lock_idx = spoofer.spoof_from_rx(
            rx_samples=rx_spoofer,
            local_code=e1b_chips,
            samples_per_symbol=Ns,
            return_debug=False,
        )

        # Victim receiver channel (independent noise realization, same CN0)
        chan_rx = AWGNChannel(cn0_db_hz=cn0_rx, chip_rate_hz=chip_rate_hz, seed=seed_noise + 7)
        recv_samples = chan_rx.add_noise(spoofed)

        # Spoofer BER against true bits
        bhat_hard = np.where(np.asarray(bhat_final) >= 0.0, 1.0, -1.0)
        spoofer_ber = float(np.mean(bhat_hard != bits))

        stab_stats = None

    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    # Partial correlations + aligned metrics at the victim receiver
    Bbeg, Bend, bhat = correlator.partial_correlations(
        auth_samples=local_code_samples,  # code-only replica
        recv_samples=recv_samples,
        samples_per_symbol=Ns,
    )

    ser = _ser_wipeoff(bits, bhat)

    metrics_per_nb: List[Dict[str, Any]] = []
    for Nb in nb_list:
        Nb_i = int(Nb)
        if Nb_i > Bbeg.size:
            raise ValueError(f"Nb={Nb_i} exceeds available symbols={Bbeg.size}.")
        met = correlator.r_metrics(Bbeg[:Nb_i], Bend[:Nb_i])
        metrics_per_nb.append({"Nb": Nb_i, **met})

    return {
        "cn0_dbhz": float(cn0_dbhz),
        "scenario": str(scenario.value),
        "trial_index": int(trial_index),
        "SER_wipeoff": float(ser),
        "spoofer_BER": spoofer_ber,
        "metrics": metrics_per_nb,
        "stab_stats": stab_stats,
    }


# =============================================================================
# Grid runner
# =============================================================================

def run_experiment_grid(
    cfg: ExperimentConfig,
    sig_cfg: SignalConfig,
    sp_cfg: SpooferConfig,
    scenarios: List[Scenario],
) -> List[Dict[str, Any]]:
    """
    Run the full experiment grid.

    For each cn0 in cfg.cn0_list_dbhz, each scenario, and each trial index,
    simulate a trial and store a record.

    Returns
    -------
    List[Dict[str, Any]]
        A list of trial records (nested metrics per Nb).
    """
    records: List[Dict[str, Any]] = []
    total = len(cfg.cn0_list_dbhz) * len(scenarios) * int(cfg.n_trials)
    done = 0

    for cn0 in cfg.cn0_list_dbhz:
        for scenario in scenarios:
            for ti in range(int(cfg.n_trials)):
                rec = run_single_trial(
                    scenario=scenario,
                    cn0_dbhz=float(cn0),
                    nb_list=list(cfg.nb_list),
                    cfg=cfg,
                    sig_cfg=sig_cfg,
                    sp_cfg=sp_cfg,
                    trial_index=ti,
                )
                records.append(rec)

                done += 1
                if done % 100 == 0 or done == total:
                    print(
                        "[run_experiment_grid] "
                        f"{done}/{total} | CN0={float(cn0):.1f} | scenario={scenario.value} | "
                        f"trial={ti + 1}/{int(cfg.n_trials)}"
                    )

    return records


# =============================================================================
# Flattening and summaries
# =============================================================================

def flatten_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert nested trial records into flat rows.

    Output rows have one row per (cn0, scenario, trial_index, Nb).
    This format is suitable for CSV storage and offline analysis.
    """
    rows: List[Dict[str, Any]] = []

    for r in records:
        for m in r["metrics"]:
            rows.append(
                {
                    "cn0_dbhz": float(r["cn0_dbhz"]),
                    "scenario": str(r["scenario"]),
                    "trial_index": int(r["trial_index"]),
                    "Nb": int(m["Nb"]),
                    "R1": float(m["R1"]),
                    "R2": float(m["R2"]),
                    "R3": float(m["R3"]),
                    "SER_wipeoff": float(r["SER_wipeoff"]),
                    "spoofer_BER": (None if r["spoofer_BER"] is None else float(r["spoofer_BER"])),
                }
            )

    return rows


def save_flattened_records_csv(path: str, records: List[Dict[str, Any]]) -> None:
    """
    Save flattened per-trial, per-Nb metrics to CSV.

    Storing the flattened rows allows post-processing (thresholds, ROC, plots)
    to be repeated without re-running simulations.
    """
    rows = flatten_records(records)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "cn0_dbhz",
        "scenario",
        "trial_index",
        "Nb",
        "R1",
        "R2",
        "R3",
        "SER_wipeoff",
        "spoofer_BER",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def load_flattened_records_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load flattened rows previously saved by save_flattened_records_csv().

    Returns
    -------
    List[Dict[str, Any]]
        Flat rows with parsed numeric types.
    """
    out: List[Dict[str, Any]] = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "cn0_dbhz": float(row["cn0_dbhz"]),
                    "scenario": str(row["scenario"]),
                    "trial_index": int(row["trial_index"]),
                    "Nb": int(row["Nb"]),
                    "R1": float(row["R1"]),
                    "R2": float(row["R2"]),
                    "R3": float(row["R3"]),
                    "SER_wipeoff": float(row["SER_wipeoff"]),
                    "spoofer_BER": (None if row["spoofer_BER"] in ("", "None", None) else float(row["spoofer_BER"])),
                }
            )

    return out


def summarize_records(records: List[Dict[str, Any]]) -> Dict[Tuple[float, str, int], Dict[str, float]]:
    """
    Compute summary statistics keyed by (cn0_dbhz, scenario, Nb).

    Returns:
      - R2 mean, p90, p99
      - R3 mean
      - SER mean
      - n (number of trials)
    """
    rows = flatten_records(records)

    out: Dict[Tuple[float, str, int], Dict[str, float]] = {}
    keys = {(row["cn0_dbhz"], row["scenario"], row["Nb"]) for row in rows}

    for key in sorted(keys):
        cn0, sc, Nb = key
        grp = [row for row in rows if row["cn0_dbhz"] == cn0 and row["scenario"] == sc and row["Nb"] == Nb]

        R2 = np.array([g["R2"] for g in grp], dtype=float)
        R3 = np.array([g["R3"] for g in grp], dtype=float)
        SER = np.array([g["SER_wipeoff"] for g in grp], dtype=float)

        out[key] = {
            "R2_mean": float(np.mean(R2)),
            "R2_p90": float(np.percentile(R2, 90)),
            "R2_p99": float(np.percentile(R2, 99)),
            "R3_mean": float(np.mean(R3)),
            "SER_mean": float(np.mean(SER)),
            "n": float(len(grp)),
        }

    return out


def print_summary_table(summary: Dict[Tuple[float, str, int], Dict[str, float]]) -> None:
    """Print a compact summary table for quick inspection."""
    print("\n=== Experiment summary (no thresholds yet) ===")
    print("CN0 | scenario  | Nb   | n  | R2_mean | R2_p99 | R3_mean | SER_mean")
    print("-" * 78)

    for (cn0, sc, Nb), s in summary.items():
        print(
            f"{cn0:4.1f} | {sc:9} | {Nb:4d} | {int(s['n']):3d} |"
            f" {s['R2_mean']:7.3f} | {s['R2_p99']:6.3f} | {s['R3_mean']:7.1f} | {s['SER_mean']:7.3f}"
        )


# =============================================================================
# Thresholding + detection metrics
# =============================================================================

def _quantile_threshold(values: np.ndarray, pfa_target: float) -> float:
    """
    Choose gamma such that P(metric > gamma) ≈ pfa_target under baseline (H0).

    With the convention "attack if score > gamma", we set:
        gamma = quantile_{1 - pfa_target}(values).
    """
    if not (0.0 < pfa_target < 1.0):
        raise ValueError("pfa_target must be in (0, 1).")

    q = 1.0 - float(pfa_target)
    return float(np.quantile(values, q))


def compute_thresholds_and_detection(
    records: List[Dict[str, Any]],
    metric: str = "R2",
    pfa_targets: Iterable[float] = (1e-2, 1e-3),
    baseline_name: str = "baseline",
    attack_name: str = "scer",
) -> List[Dict[str, Any]]:
    """
    For each (CN0, Nb), compute:
      - gamma from the BASELINE quantile (calibration on all baseline samples)
      - achieved Pfa on BASELINE (sanity check)
      - Pd on SCER

    Returns
    -------
    List[Dict[str, Any]]
        Rows suitable for printing or saving.
    """
    if metric not in ("R1", "R2", "R3"):
        raise ValueError("metric must be one of: R1, R2, R3.")

    rows = flatten_records(records)

    pairs = sorted({(r["cn0_dbhz"], r["Nb"]) for r in rows})
    out_rows: List[Dict[str, Any]] = []

    for cn0, Nb in pairs:
        base_vals = np.array(
            [r[metric] for r in rows if r["cn0_dbhz"] == cn0 and r["Nb"] == Nb and r["scenario"] == baseline_name],
            dtype=float,
        )
        atk_vals = np.array(
            [r[metric] for r in rows if r["cn0_dbhz"] == cn0 and r["Nb"] == Nb and r["scenario"] == attack_name],
            dtype=float,
        )

        if base_vals.size == 0 or atk_vals.size == 0:
            continue

        for pfa in pfa_targets:
            gamma = _quantile_threshold(base_vals, pfa_target=float(pfa))
            pfa_hat = float(np.mean(base_vals > gamma))
            pd_hat = float(np.mean(atk_vals > gamma))

            out_rows.append(
                {
                    "cn0_dbhz": float(cn0),
                    "Nb": int(Nb),
                    "metric": metric,
                    "pfa_target": float(pfa),
                    "gamma": float(gamma),
                    "pfa_hat": float(pfa_hat),
                    "pd_hat": float(pd_hat),
                    "n_baseline": int(base_vals.size),
                    "n_attack": int(atk_vals.size),
                }
            )

    return out_rows


def compute_thresholds_and_detection_split_from_rows(
    rows: List[Dict[str, Any]],
    metric: str = "R2",
    pfa_targets: Iterable[float] = (1e-2,),
    baseline_name: str = "baseline",
    attack_name: str = "scer",
    n_cal: int = 2500,
    n_test: int = 2500,
) -> List[Dict[str, Any]]:
    """
    Split-based threshold evaluation using already-flattened rows.

    - gamma is calibrated on a baseline calibration subset (size n_cal)
    - Pfa_hat is evaluated on an independent baseline test subset (size n_test)
    - Pd_hat is evaluated on an independent attack test subset (size n_test)
    """
    if metric not in ("R1", "R2", "R3"):
        raise ValueError("metric must be one of: R1, R2, R3.")

    pairs = sorted({(r["cn0_dbhz"], r["Nb"]) for r in rows})
    out_rows: List[Dict[str, Any]] = []

    rng = np.random.default_rng(0)

    for cn0, Nb in pairs:
        base_vals = np.array(
            [r[metric] for r in rows if r["cn0_dbhz"] == cn0 and r["Nb"] == Nb and r["scenario"] == baseline_name],
            dtype=float,
        )
        atk_vals = np.array(
            [r[metric] for r in rows if r["cn0_dbhz"] == cn0 and r["Nb"] == Nb and r["scenario"] == attack_name],
            dtype=float,
        )

        if base_vals.size < (n_cal + n_test) or atk_vals.size < n_test:
            continue

        base_vals = base_vals.copy()
        atk_vals = atk_vals.copy()
        rng.shuffle(base_vals)
        rng.shuffle(atk_vals)

        base_cal = base_vals[:n_cal]
        base_test = base_vals[n_cal:n_cal + n_test]
        atk_test = atk_vals[:n_test]

        for pfa in pfa_targets:
            gamma = _quantile_threshold(base_cal, pfa_target=float(pfa))
            pfa_hat = float(np.mean(base_test > gamma))
            pd_hat = float(np.mean(atk_test > gamma))

            out_rows.append(
                {
                    "cn0_dbhz": float(cn0),
                    "Nb": int(Nb),
                    "metric": metric,
                    "pfa_target": float(pfa),
                    "gamma": float(gamma),
                    "pfa_hat": float(pfa_hat),
                    "pd_hat": float(pd_hat),
                    "n_cal": int(n_cal),
                    "n_test": int(n_test),
                }
            )

    return out_rows


def compute_thresholds_and_detection_split(
    records: List[Dict[str, Any]],
    metric: str = "R2",
    pfa_targets: Iterable[float] = (1e-2,),
    baseline_name: str = "baseline",
    attack_name: str = "scer",
    n_cal: int = 2500,
    n_test: int = 2500,
) -> List[Dict[str, Any]]:
    """
    Split-based threshold evaluation (from nested trial records).

    Same as compute_thresholds_and_detection(), but:
      - gamma is calibrated on a baseline calibration subset
      - Pfa_hat is evaluated on an independent baseline test subset
      - Pd_hat is evaluated on an independent attack test subset
    """
    if metric not in ("R1", "R2", "R3"):
        raise ValueError("metric must be one of: R1, R2, R3.")

    rows = flatten_records(records)
    return compute_thresholds_and_detection_split_from_rows(
        rows=rows,
        metric=metric,
        pfa_targets=pfa_targets,
        baseline_name=baseline_name,
        attack_name=attack_name,
        n_cal=n_cal,
        n_test=n_test,
    )


# =============================================================================
# ROC curves (split-based)
# =============================================================================

def compute_roc_curves_split_from_rows(
    rows: List[Dict[str, Any]],
    metrics: Iterable[str] = ("R2", "R3"),
    baseline_name: str = "baseline",
    attack_name: str = "scer",
    n_cal: int = 2000,
    n_test: int = 2000,
    max_thresholds: Optional[int] = 2000,
) -> List[Dict[str, Any]]:
    """
    Compute split-based ROC points using already-flattened rows.

    ROC uses baseline TEST (size n_test) for Pfa and attack TEST (size n_test) for Pd.
    The threshold sweep is built from unique pooled TEST scores, optionally downsampled.
    """
    for metric in metrics:
        if metric not in ("R1", "R2", "R3"):
            raise ValueError("metrics must be chosen from: R1, R2, R3.")

    pairs = sorted({(r["cn0_dbhz"], r["Nb"]) for r in rows})
    rng = np.random.default_rng(0)
    out_rows: List[Dict[str, Any]] = []

    for metric in metrics:
        for cn0, Nb in pairs:
            base_vals = np.array(
                [r[metric] for r in rows
                 if r["cn0_dbhz"] == cn0 and r["Nb"] == Nb and r["scenario"] == baseline_name],
                dtype=float,
            )
            atk_vals = np.array(
                [r[metric] for r in rows
                 if r["cn0_dbhz"] == cn0 and r["Nb"] == Nb and r["scenario"] == attack_name],
                dtype=float,
            )

            if base_vals.size < (n_cal + n_test) or atk_vals.size < n_test:
                continue

            base_vals = base_vals.copy()
            atk_vals = atk_vals.copy()
            rng.shuffle(base_vals)
            rng.shuffle(atk_vals)

            base_test = base_vals[n_cal:n_cal + n_test]
            atk_test = atk_vals[:n_test]

            thr = np.unique(np.concatenate([base_test, atk_test]))

            if (max_thresholds is not None) and (thr.size > max_thresholds):
                q = np.linspace(0.0, 1.0, int(max_thresholds))
                thr = np.quantile(np.concatenate([base_test, atk_test]), q)

            for gamma in thr:
                pfa_hat = float(np.mean(base_test > gamma))
                pd_hat = float(np.mean(atk_test > gamma))
                out_rows.append(
                    {
                        "cn0_dbhz": float(cn0),
                        "Nb": int(Nb),
                        "metric": str(metric),
                        "gamma": float(gamma),
                        "pfa_hat": float(pfa_hat),
                        "pd_hat": float(pd_hat),
                        "n_cal": int(n_cal),
                        "n_test": int(n_test),
                        "n_thr": int(len(thr)),
                    }
                )

    return out_rows


def compute_roc_curves_split(
    records: List[Dict[str, Any]],
    metrics: Iterable[str] = ("R2", "R3"),
    baseline_name: str = "baseline",
    attack_name: str = "scer",
    n_cal: int = 2000,
    n_test: int = 2000,
    max_thresholds: Optional[int] = 2000,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: compute split-based ROC points from nested trial records.
    """
    rows = flatten_records(records)
    return compute_roc_curves_split_from_rows(
        rows=rows,
        metrics=metrics,
        baseline_name=baseline_name,
        attack_name=attack_name,
        n_cal=n_cal,
        n_test=n_test,
        max_thresholds=max_thresholds,
    )


def save_roc_rows_csv(path: str, roc_rows: List[Dict[str, Any]]) -> None:
    """Save ROC rows to CSV."""
    if not roc_rows:
        raise ValueError("No ROC rows to save.")

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(roc_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(roc_rows)


def print_detection_table(det_rows: List[Dict[str, Any]]) -> None:
    """
    Print detection results in a compact table.

    Supports both schemas:
      - Full calibration on all baseline samples: n_baseline / n_attack
      - Split evaluation: n_cal / n_test
    """
    if not det_rows:
        print("\n(No detection rows to print.)")
        return

    print("\n=== Thresholding + detection performance ===")
    print("CN0 | Nb   | metric | Pfa*     | gamma     | Pfa_hat  | Pd_hat   | n0  | n1")
    print("-" * 90)

    det_rows_sorted = sorted(det_rows, key=lambda d: (d["cn0_dbhz"], d["Nb"], d["pfa_target"]))

    for d in det_rows_sorted:
        n0 = d.get("n_baseline", d.get("n_cal", None))
        n1 = d.get("n_attack", d.get("n_test", None))

        n0_str = f"{int(n0):3d}" if n0 is not None else "  -"
        n1_str = f"{int(n1):3d}" if n1 is not None else "  -"

        print(
            f"{d['cn0_dbhz']:4.1f} | {d['Nb']:4d} | {d['metric']:>5} |"
            f" {d['pfa_target']:<8.1e} | {d['gamma']:9.4f} |"
            f" {d['pfa_hat']:7.4f} | {d['pd_hat']:7.4f} |"
            f" {n0_str} | {n1_str}"
        )
