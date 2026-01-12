# plot_roc_from_csv.py
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_roc_csv(path: str):
    """
    Expected columns (as produced by compute_roc_curves_split / save_roc_rows_csv):
      cn0_dbhz, Nb, metric, gamma, pfa_hat, pd_hat, n_cal, n_test, n_thr
    Extra columns are ignored.
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "cn0_dbhz": float(r["cn0_dbhz"]),
                "Nb": int(float(r["Nb"])),
                "metric": str(r["metric"]).strip(),
                "gamma": float(r["gamma"]),
                "pfa_hat": float(r["pfa_hat"]),
                "pd_hat": float(r["pd_hat"]),
                "n_cal": int(float(r.get("n_cal", 0) or 0)),
                "n_test": int(float(r.get("n_test", 0) or 0)),
                "n_thr": int(float(r.get("n_thr", 0) or 0)),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")
    return rows


def _group_curves(rows, metric: str):
    """
    Returns: dict[(cn0_dbhz, Nb)] -> (pfa_sorted, pd_sorted)
    """
    grouped = defaultdict(list)
    for r in rows:
        if r["metric"] != metric:
            continue
        key = (r["cn0_dbhz"], r["Nb"])
        grouped[key].append((r["pfa_hat"], r["pd_hat"]))

    curves = {}
    for key, pts in grouped.items():
        # sort by Pfa
        pts_sorted = sorted(pts, key=lambda t: t[0])
        pfa = np.array([p[0] for p in pts_sorted], dtype=float)
        pd = np.array([p[1] for p in pts_sorted], dtype=float)

        # de-duplicate identical pfa bins (keep max Pd for cleaner curves)
        # this can happen when multiple thresholds map to same empirical rate.
        if pfa.size > 1:
            uniq_pfa = []
            best_pd = []
            i = 0
            while i < pfa.size:
                j = i
                pd_max = pd[i]
                while j + 1 < pfa.size and np.isclose(pfa[j + 1], pfa[i], rtol=0.0, atol=0.0):
                    j += 1
                    pd_max = max(pd_max, pd[j])
                uniq_pfa.append(pfa[i])
                best_pd.append(pd_max)
                i = j + 1
            pfa = np.array(uniq_pfa, dtype=float)
            pd = np.array(best_pd, dtype=float)

        curves[key] = (pfa, pd)

    return curves


def _auc_trapz(pfa: np.ndarray, pd: np.ndarray) -> float:
    """
    AUC over the empirical ROC using trapezoidal integration in Pfa-domain.
    Note: empirical curves may not start at 0 or end at 1 depending on resolution.
    """
    if pfa.size < 2:
        return float("nan")
    # ensure sorted
    idx = np.argsort(pfa)
    return float(np.trapz(pd[idx], pfa[idx]))


def plot_roc_per_cn0(rows, out_dir: str, metric: str, x_min: float = 5e-4):
    """
    One figure per C/N0, with curves for each Nb.
    x-axis is log-scaled Pfa.
    """
    os.makedirs(out_dir, exist_ok=True)

    curves = _group_curves(rows, metric=metric)
    cn0_values = sorted({cn0 for (cn0, _nb) in curves.keys()})
    if not cn0_values:
        raise ValueError(f"No curves found for metric={metric}")

    for cn0 in cn0_values:
        plt.figure(figsize=(9, 5))

        nb_values = sorted({nb for (c, nb) in curves.keys() if c == cn0})
        for nb in nb_values:
            pfa, pd = curves[(cn0, nb)]

            # clip to display range (avoid log(0))
            mask = pfa >= x_min
            if np.any(mask):
                pfa_plot = pfa[mask]
                pd_plot = pd[mask]
            else:
                # if everything is below x_min, plot the last point (best-effort)
                pfa_plot = pfa[-1:]
                pd_plot = pd[-1:]

            auc = _auc_trapz(pfa, pd)
            plt.plot(pfa_plot, pd_plot, label=rf"$N_b={nb}$ (AUC={auc:.3f})")

        plt.xscale("log")
        plt.xlim(x_min, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"Empirical $\hat{P}_{FA}$")
        plt.ylabel(r"Empirical $\hat{P}_D$")
        plt.title(rf"ROC under SCER ({metric}) at $C/N_0={cn0:g}$ dB-Hz")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"roc_{metric}_cn0_{int(round(cn0))}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_roc_grid_one_metric(rows, out_path: str, metric: str, x_min: float = 5e-4):
    """
    Single figure per metric: overlay curves for all (cn0, Nb).
    Useful as a quick sanity check but can be visually dense.
    """
    curves = _group_curves(rows, metric=metric)
    if not curves:
        raise ValueError(f"No curves found for metric={metric}")

    plt.figure(figsize=(9, 6))
    for (cn0, nb), (pfa, pd) in sorted(curves.items(), key=lambda k: (k[0][0], k[0][1])):
        mask = pfa >= x_min
        pfa_plot = pfa[mask] if np.any(mask) else pfa[-1:]
        pd_plot = pd[mask] if np.any(mask) else pd[-1:]
        plt.plot(pfa_plot, pd_plot, label=rf"$C/N_0={cn0:g}$, $N_b={nb}$")

    plt.xscale("log")
    plt.xlim(x_min, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel(r"Empirical $\hat{P}_{FA}$")
    plt.ylabel(r"Empirical $\hat{P}_D$")
    plt.title(rf"ROC under SCER ({metric})")
    plt.grid(True, which="both")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    csv_path = "results/roc_R2_R3.csv"
    out_dir = "results/roc"

    rows = load_roc_csv(csv_path)

    metrics = sorted({r["metric"] for r in rows})
    if not metrics:
        raise ValueError("No metrics found in ROC CSV.")

    # Determine the smallest non-zero Pfa step from data (useful for log x-axis lower bound)
    # If you used n_test=2000, this will typically be 5e-4.
    pfa_vals = np.array([r["pfa_hat"] for r in rows], dtype=float)
    pfa_pos = pfa_vals[pfa_vals > 0.0]
    x_min = float(np.min(pfa_pos)) if pfa_pos.size else 1e-4

    for metric in metrics:
        # One plot per CN0 (recommended for readability)
        plot_roc_per_cn0(rows, out_dir=out_dir, metric=metric, x_min=x_min)

        # Optional: one dense plot per metric
        # plot_roc_grid_one_metric(
        #     rows,
        #     out_path=os.path.join(out_dir, f"roc_{metric}_all.png"),
        #     metric=metric,
        #     x_min=x_min,
        # )

    print(f"Saved ROC plots under {out_dir}/")


if __name__ == "__main__":
    main()
