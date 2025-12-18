# visualization.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def running_correlation(auth_sym: np.ndarray, recv_sym: np.ndarray) -> np.ndarray:
    """
    Correlación acumulada normalizada muestra a muestra para UN símbolo.

    r[m] = (sum_{n=0}^m auth_sym[n] * recv_sym[n]) / sum_{n=0}^{N-1} auth_sym[n] * recv_sym[n]

    De esta forma:
      - r[-1] = 1.0
      - El resto de valores quedan ~entre -1 y 1.
    """
    if auth_sym.shape != recv_sym.shape:
        raise ValueError("auth_sym and recv_sym must have the same length")

    prods  = auth_sym * recv_sym
    cumsum = np.cumsum(prods)
    total  = cumsum[-1]

    if total == 0:
        # caso extremo: correlación total nula
        return cumsum

    return cumsum / total

def plot_noisy_vs_local_symbol(auth_samples: np.ndarray,
                               noisy_samples: np.ndarray,
                               samples_per_symbol: int,
                               symbol_idx: int,
                               T_symbol_ms: float) -> None:
    """
    Plotea en DOS PANELES la réplica local y la señal recibida para UN símbolo.

    Panel superior: réplica local (código E1B limpio)
    Panel inferior: señal recibida (spoof + ruido)
    """
    start = symbol_idx * samples_per_symbol
    end   = start + samples_per_symbol

    auth_sym  = auth_samples[start:end]
    noisy_sym = noisy_samples[start:end]

    # Eje temporal dentro del símbolo (ms)
    t_ms = np.linspace(0.0, T_symbol_ms, samples_per_symbol, endpoint=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    # Panel superior: réplica local
    ax1.step(t_ms, auth_sym, where="post", label="réplica local (código E1B limpio)")
    ax1.set_ylabel("Amplitud")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title(f"Símbolo {symbol_idx}: réplica local")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    # Panel inferior: señal recibida (spoof + ruido)
    ax2.step(t_ms, noisy_sym, where="post", label="señal recibida (spoof + ruido)")
    ax2.set_xlabel("Tiempo dentro del símbolo (ms)")
    ax2.set_ylabel("Amplitud")
    ax2.grid(True)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
