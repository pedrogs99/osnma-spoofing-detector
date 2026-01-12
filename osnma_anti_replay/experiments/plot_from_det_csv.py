# plot_from_det_csv.py
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_det_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields robustly
            row = {
                "cn0_dbhz": float(r["cn0_dbhz"]),
                "Nb": int(float(r["Nb"])),
                "metric": str(r["metric"]).strip(),
                "pfa_target": float(r["pfa_target"]),
                "gamma": float(r["gamma"]),
                "pfa_hat": float(r["pfa_hat"]),
                "pd_hat": float(r["pd_hat"]),
                "n_cal": int(float(r.get("n_cal", 0) or 0)),
                "n_test": int(float(r.get("n_test", 0) or 0)),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")
    return rows


def _series_by_nb(rows, field: str, metric: str, pfa_target: float):
    # returns dict Nb -> (x_cn0_sorted, y_sorted)
    grouped = defaultdict(list)
    for r in rows:
        if r["metric"] != metric:
            continue
        if abs(r["pfa_target"] - pfa_target) > 1e-12:
            continue
        grouped[r["Nb"]].append((r["cn0_dbhz"], r[field]))

    out = {}
    for Nb, pts in grouped.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        x = np.array([p[0] for p in pts_sorted], dtype=float)
        y = np.array([p[1] for p in pts_sorted], dtype=float)
        out[Nb] = (x, y)
    return out


def plot_pfa(rows, out_path: str, metric: str, pfa_target: float):
    series = _series_by_nb(rows, "pfa_hat", metric, pfa_target)

    plt.figure(figsize=(9, 5))
    for Nb in sorted(series.keys()):
        x, y = series[Nb]
        plt.plot(x, y, marker="o", label=rf"$N_b={Nb}$")

    # target line
    plt.axhline(pfa_target, linestyle="--", linewidth=1.0, label=rf"target $P_{{FA}}^\star={pfa_target:g}$")

    plt.yscale("log")  # calibration is best seen on log scale
    plt.xlabel(r"$C/N_0$ (dB-Hz)")
    plt.ylabel(r"Empirical $\hat{P}_{FA}$")
    plt.title(rf"False-alarm calibration vs $C/N_0$ ({metric}, $P_{{FA}}^\star={pfa_target:g}$)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pd(rows, out_path: str, metric: str, pfa_target: float):
    series = _series_by_nb(rows, "pd_hat", metric, pfa_target)

    plt.figure(figsize=(9, 5))
    for Nb in sorted(series.keys()):
        x, y = series[Nb]
        plt.plot(x, y, marker="o", label=rf"$N_b={Nb}$")

    plt.ylim(-0.02, 1.02)
    plt.xlabel(r"$C/N_0$ (dB-Hz)")
    plt.ylabel(r"Empirical $\hat{P}_D$")
    plt.title(rf"Detection probability vs $C/N_0$ ({metric}, threshold at $P_{{FA}}^\star={pfa_target:g}$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gamma(rows, out_path: str, metric: str, pfa_target: float, logy: bool = False):
    series = _series_by_nb(rows, "gamma", metric, pfa_target)

    plt.figure(figsize=(9, 5))
    for Nb in sorted(series.keys()):
        x, y = series[Nb]
        plt.plot(x, y, marker="o", label=rf"$N_b={Nb}$")

    if logy:
        plt.yscale("log")

    plt.xlabel(r"$C/N_0$ (dB-Hz)")
    plt.ylabel(r"Threshold $\gamma$")
    plt.title(rf"Quantile threshold vs $C/N_0$ ({metric}, $P_{{FA}}^\star={pfa_target:g}$)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    csv_path = "results/pfa_calibration_R3.csv"  # <-- change to your path

    rows = load_det_csv(csv_path)

    # auto-detect (assumes single metric and single pfa_target in file)
    metrics = sorted({r["metric"] for r in rows})
    pfas = sorted({r["pfa_target"] for r in rows})
    if len(metrics) != 1 or len(pfas) != 1:
        raise ValueError(f"Expected single metric and single pfa_target. Found metrics={metrics}, pfas={pfas}")

    metric = metrics[0]
    pfa_target = pfas[0]

    plot_pfa(rows, f"results/pfa_hat_vs_cn0_{metric}_pfa{pfa_target:.0e}.png", metric, pfa_target)
    plot_pd(rows, f"results/pd_hat_vs_cn0_{metric}_pfa{pfa_target:.0e}.png", metric, pfa_target)

    # For R3, gamma spans 30..340 here; linear is fine, logy optional.
    plot_gamma(rows, f"results/gamma_vs_cn0_{metric}_pfa{pfa_target:.0e}.png", metric, pfa_target, logy=False)

    print("Saved plots under results/.")


if __name__ == "__main__":
    main()
