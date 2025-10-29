# OSNMA E1-B Partial-Correlation Spoofing Detector

A lightweight research prototype—built for my Master’s Thesis—designed to detect spoofing attacks on the Galileo E1-B navigation signal using **partial-correlation metrics**. The project progresses from a noise-free toy model to a full Monte-Carlo simulation with ROC analysis.

> **Status**: 🚧 *phase 0* — baseline implementation without noise or active spoofer.

---

## ✨ Key Features (current & planned)

| Phase | Module                   | Purpose                                                                                                      |
| ----- | ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| 0     | **`E1BSignalGenerator`** | Generates +1/-1 symbol streams and expands them to discrete samples (4 ms per symbol, 8 samples per symbol). |
| 0     | **`PartialCorrelator`**  | Computes correlation between the first and last 0.5 ms of each symbol; averages over the sequence.           |
| 0     | **`Spoofer`** *(stub)*   | Flips symbols with configurable probability `error_prob` (initially 0).                                      |
| 1     | **AWGN Noise**           | Additive white Gaussian noise with controllable SNR.                                                         |
| 2     | **ROC & Thresholding**   | Automate Monte-Carlo runs to estimate ${P_D}$ / ${P_{FA}}$ and plot ROC curves.                              |
| 3     | **CLI & Plots**          | Command-line interface, Matplotlib plots, exportable CSV results.                                            |

---

## 📂 Project Structure

```
project/
│  e1b_spoofing_detector.py      # Main skeleton (run this first)
│  requirements.txt              # Python dependencies
│  README.md                     # You are here
└─ .vscode/                      # VS Code launch & settings (not essential)
```

---

## 🚀 Getting Started

### Prerequisites

* Python ≥ 3.11 (Windows, macOS, or Linux)
* Git (to clone the repository)

### Installation (Windows PowerShell example)

```powershell
# Clone the repo
> git clone https://github.com/<your-user>/osnma-spoofing-detector.git
> cd osnma-spoofing-detector

# Create & activate virtual env
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1    # Cmd: .venv\Scripts\activate.bat ; Linux: source .venv/bin/activate

# Install dependencies
(.venv) > python -m pip install --upgrade pip
(.venv) > pip install -r requirements.txt   # currently only numpy; grows later
```

### First Run

```powershell
(.venv) > python e1b_spoofing_detector.py
Métrica de correlación parcial (sin spoofer): 1.00
```

You should see a correlation metric ≈ +1, confirming the baseline “no-spoof” case.

---

## 📈 Roadmap

* **Phase 0** ✔️ — verify ideal correlation metric.
* **Phase 1** ➡️ — integrate AWGN noise model.
* **Phase 2** 📊 — threshold search and ROC curve generation.
* **Phase 3** 📈 — parameter sweeps, performance plots, and report figures.

---

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for more information.

