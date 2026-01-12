# OSNMA E1-B Partial-Correlation Detector (SCER / Replay)

Research prototype developed for a Master’s Thesis on software-only detection of Galileo OSNMA replay (SCER-like) spoofing using sample-level partial-correlation statistics computed over Galileo E1-B I/NAV symbols.

The implementation includes:
- A chip-rate signal model (one sample per chip)
- A C/N0-driven AWGN channel
- A SCER-style spoofer that estimates per-symbol signs via running correlation and replays a synthesized signal
- A partial-correlator that computes early/late partial correlations per symbol and aggregates them into R1/R2/R3
- A Monte Carlo experiment framework with threshold calibration and ROC generation

Note: This repository is intended for research and reproducibility, not for operational GNSS receiver integration.

---

## 1. Core idea

For each symbol, the receiver forms partial correlations over an early and late window inside the symbol:
- Bbeg: correlation over the first window
- Bend: correlation over the last window

A realistic wipe-off alignment is applied by estimating the symbol sign (bhat) from the full-symbol correlation, and using it to align both partial correlations.

Over Nb symbols, the detector aggregates these per-symbol values into global decision statistics:
- R1, R2, R3 (Seco-Granados-style definitions)

Detection is performed via a threshold test:
- Declare attack if metric > gamma

---

## 2. Repository layout

.
├── main.py
├── awgn_channel.py
├── spoofer.py
├── partial_correlator.py
├── e1b_code_helpers.py
└── experiments/
    ├── configs.py
    └── runner.py

---

## 3. Module overview

- main.py
  End-to-end entry point for:
  - single-symbol illustrative plots
  - many-symbol statistics and convergence plots
  - experiment sweeps and CSV export (via experiments/)

- awgn_channel.py
  Real-valued AWGN channel model specified by C/N0 (dB-Hz) and chip rate.

- spoofer.py
  SCER-style spoofer:
  - estimates running sign via cumulative correlation
  - optionally uses a lock heuristic (lock_gamma) and a pre-lock mode
  - generates replay s[n] = b_hat[n] · c[n]

- partial_correlator.py
  Per-symbol partial correlations (Bbeg, Bend) and aggregation metrics (R1, R2, R3).
  bhat is estimated from the full-symbol correlation.

- e1b_code_helpers.py
  Helpers to convert the Galileo E1-B code from ICD hex format to ±1 chips.

- experiments/configs.py
  Dataclasses and enums for sweep configuration and scenario selection:
  - Scenario.BASELINE (H0)
  - Scenario.SCER (H1)

- experiments/runner.py
  Monte Carlo runner and post-processing utilities:
  - grid sweeps over C/N0 and Nb
  - flattening to CSV
  - summary tables (means/percentiles)
  - quantile threshold calibration
  - split-based detection evaluation (cal/test separation)
  - split-based ROC generation

---

## 4. Requirements

- Python 3.11+
- Dependencies:
  - numpy
  - matplotlib (for plots in main.py)

Install via:
python -m pip install -r requirements.txt

---

## 5. Quick start

### 5.1 Clone and set up a virtual environment

git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

### 5.2 Configure the E1-B code (ICD hex)

In main.py, set E1B_CODE_HEX to the Galileo E1-B code in ICD hex format.

### 5.3 Run the main script

python main.py

By default, main.py is typically configured to run the many-symbol statistics path. You can enable additional runs by uncommenting blocks under:
if __name__ == "__main__":

---

## 6. Typical workflows

### 6.1 Single-symbol illustrative figure (5 panels)

In main.py, enable:
run_single_symbol_plot()

This produces a multi-panel figure showing:
- authentic TX symbol
- spoofer input (TX + AWGN)
- spoofed output
- cumulative correlation
- normalized cumulative correlation

### 6.2 Many-symbol run (statistics + plots)

In main.py, enable:
run = run_many_symbols_stats()

This produces:
- stabilization-time distribution plots (based on spoofer running decisions)
- per-symbol partial correlation distributions
- scatter plots linking stabilization to partial-correlation gaps
- convergence plots for R1, R2, R3 vs number of symbols

It also returns a dictionary containing the “source of truth” arrays for the run (tx_samples, rx_spoofer, spoofed, rx_receiver, Bbeg, Bend, etc.).

### 6.3 Monte Carlo sweeps (CSV-ready results)

Use the experiments package (example pattern is included in main.py):
- build an ExperimentConfig
- run run_experiment_grid(...)
- save flattened rows to CSV
- compute thresholds and detection performance
- compute ROC curves (split-based)

Outputs are typically written under a results/ directory (user-defined).

---

## 7. Reproducibility

The experiment framework derives deterministic seeds from a single master_seed:
- one RNG stream for symbol bits
- one RNG stream for channel noise
- one RNG stream for the spoofer

This ensures runs are stable and repeatable when configuration parameters are unchanged.

---

## 8. Notes and limitations

- The signal model is chip-rate and real-valued, intended for controlled experiments.
- Receiver tracking loops, code delay estimation, and front-end filtering are out of scope.
- The SCER spoofer is a simplified model based on running-correlation sign estimation, not a full waveform generator.

---

## 9. License

MIT License (see LICENSE).
