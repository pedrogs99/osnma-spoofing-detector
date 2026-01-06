"""
main.py – Simulador simple de spoofing sobre código Galileo E1-B

Hace dos cosas:
1) Usa UN símbolo (4092 chips) para generar una figura en 2 paneles:
   - Arriba: señal spoofeada (sin ruido)
   - Abajo: correlación acumulada normalizada auténtica–spoofeada

2) Usa MUCHOS símbolos (el código E1B repetido) para:
   - Aplicar spoofing + AWGN
   - Calcular R1, R2, R3 sobre N símbolos
   - Ver cómo R2 y R3 se estabilizan al aumentar el número de símbolos
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from spoofer import Spoofer, SpooferCfg
from awgn_channel import AWGNChannel
from partial_correlator import PartialCorrelator
from visualization import running_correlation, plot_noisy_vs_local_symbol
from e1b_code_helpers import e1b_hex_to_chips


# ======================================================================
# 1. Parámetros de simulación
# ======================================================================

# Pega aquí TU código E1B en hexadecimal tal y como sale del ICD (Annex C)
E1B_CODE_HEX = """
F5D710130573541B9DBD4FD9E9B20A0D59D144C54BC7935539D2E75810FB51E494093A0A19DD79C70C5A98E5657AA578097777E86BCC4651CC72F2F974DC766E07AEA3D0B557EF42FF57E6A58E805358CE9257669133B18F80FDBDFB38C5524C7FB1DE079842482990DF58F72321D9201F8979EAB159B2679C9E95AA6D53456C0DF75C2B4316D1E2309216882854253A1FA60CA2C94ECE013E2A8C943341E7D9E5A8464B3AD407E0AE465C3E3DD1BE60A8C3D50F831536401E776BE02A6042FC4A27AF653F0CFC4D4D013F115310788D68CAEAD3ECCCC5330587EB3C22A1459FC8E6FCCE9CDE849A5205E70C6D66D125814D698DD0EEBFEAE52CC65C5C84EEDF207379000E169D318426516AC5D1C31F2E18A65E07AE6E33FDD724B13098B3A444688389EFBBB5EEAB588742BB083B679D42FB26FF77919EAB21DE0389D9997498F967AE05AF0F4C7E177416E18C4D5E6987ED3590690AD127D872F14A8F4903A12329732A9768F82F295BEE391879293E3A97D51435A7F03ED7FBE275F102A83202DC3DE94AF4C712E9D006D182693E9632933E6EB773880CF147B922E74539E4582F79E39723B4C80E42EDCE4C08A8D02221BAE6D17734817D5B531C0D3C1AE723911F3FFF6AAC02E97FEA69E376AF4761E6451CA61FDB2F9187642EFCD63A09AAB680770C1593EEDD4FF4293BFFD6DD2C3367E85B14A654C834B6699421A
"""

# Duración del símbolo E1-B (4 ms → 4092 chips)
T_SYMBOL_MS = 4.0

# C/N0 en la entrada del spoofer (controla su performance)
CN0_SPOOFER_DBHZ = 20.0   # prueba valores 30–50

# C/N0 en el receptor final (opcional; puedes igualarlo al del spoofer)
CN0_RX_DBHZ = 20.0

# Partial correlator: ventana early/late para las métricas R1/R2/R3
WINDOW_FRACTION_CORR = 0.125   # ~0.5 ms de una ventana de 4 ms

# C/N0 para el escenario "realista" (stats con muchos símbolos)
CN0_DBHZ = 50.0

# Nº de símbolos para el escenario de estadísticas (muchos símbolos)
NUM_SYMBOLS_STATS = 1000    # por ejemplo, 200 símbolos de 4 ms cada uno


# ======================================================================
# 2. Construcción de la señal auténtica a partir del código E1B
# ======================================================================

def build_signals(num_symbols: int, seed_bits: int = 1):
    # 1) Código E1B en polar +/-1 (4092 chips)
    e1b_chips = e1b_hex_to_chips(E1B_CODE_HEX)  # (4092,)
    samples_per_symbol = e1b_chips.size

    # 2) Bits impredecibles por símbolo (±1)
    rng = np.random.default_rng(seed_bits)
    bits = rng.choice([-1.0, 1.0], size=num_symbols)

    # 3) Réplica local (solo código, repetido)
    local_code_matrix = np.tile(e1b_chips, (num_symbols, 1))
    local_code_samples = local_code_matrix.reshape(-1)

    # 4) Señal auténtica transmitida (código * bit por símbolo)
    tx_matrix = local_code_matrix * bits[:, None]
    tx_samples = tx_matrix.reshape(-1)

    # 5) Tasa de chip
    T_symbol = T_SYMBOL_MS * 1e-3
    chip_rate_hz = samples_per_symbol / T_symbol

    return (e1b_chips, local_code_samples, tx_samples,
            samples_per_symbol, chip_rate_hz, bits)


# ======================================================================
# 3. Escenario 1: UN símbolo – figura en 3 paneles
# ======================================================================

def run_single_symbol_plot():
    num_symbols_single = 1

    (e1b_chips, local_code_samples, tx_samples,
     samples_per_symbol, chip_rate_hz, bits) = build_signals(num_symbols_single, seed_bits=50)

    tx_sym = tx_samples[:samples_per_symbol]  # símbolo real transmitido: b*c
    b0 = float(bits[0])

    # 1) Lo que recibe el spoofer (TX + AWGN_spoofer)
    chan_spoofer = AWGNChannel(cn0_db_hz=CN0_SPOOFER_DBHZ, chip_rate_hz=chip_rate_hz, seed=123)
    rx_spoofer_sym = chan_spoofer.add_noise(tx_sym)

    # 2) Spoofer estima y genera señal spoofeada (pedimos debug para bhat_running)
    spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=42)
    spoofed_sym, bhat_final, _lock_idx, bhat_running = spoofer.spoof_from_rx(
        rx_samples=rx_spoofer_sym,
        local_code=e1b_chips,
        samples_per_symbol=samples_per_symbol,
        return_debug=True
    )

    # 3) Lo que recibe el receptor (spoof + AWGN_rx)
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=200)
    rx_receiver_sym = chan_rx.add_noise(spoofed_sym)

    # Réplica local para correlación (símbolo "verificado": b*c)
    local_replica_sym = b0 * e1b_chips

    # Correlación acumulada (total y normalizada por nº muestras)
    prod = local_replica_sym * rx_receiver_sym
    corr_cum = np.cumsum(prod)
    n = np.arange(1, samples_per_symbol + 1, dtype=float)
    corr_norm = corr_cum / n

    # Eje temporal
    t_ms = np.linspace(0.0, T_SYMBOL_MS, samples_per_symbol, endpoint=False)

    # Línea vertical: fin de ventana early del correlador
    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)
    t_early_end = (wlen / samples_per_symbol) * T_SYMBOL_MS

    # Línea vertical: instante de estabilización del spoofer (según stabilization_idx)
    # bhat_running debería ser shape (1, Ns) para un símbolo; si te viniera 1D, lo arreglamos:
    bh = np.asarray(bhat_running)
    if bh.ndim == 1:
        bh = bh.reshape(1, -1)

    stab_idx = stabilization_idx(bh, min_run=32)[0]  # índice en chips (0..Ns)
    stab_valid = (stab_idx < samples_per_symbol)
    t_stab_ms = (stab_idx / samples_per_symbol) * T_SYMBOL_MS if stab_valid else None

    # Plot 5 paneles
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(11, 10), sharex=True)

    def add_vlines(ax):
        ax.axvline(t_early_end, linestyle="--", linewidth=0.8)
        if stab_valid:
            ax.axvline(t_stab_ms, linestyle=":", linewidth=1.2)

    # 1) Símbolo real
    ax1.step(t_ms, tx_sym, where="post")
    ax1.set_ylabel("Amp")
    ax1.set_ylim(-3, 3)
    ax1.set_title("1) Símbolo real (TX auténtica): b · c[n]")
    ax1.grid(True)
    add_vlines(ax1)

    # 2) Entrada al spoofer
    ax2.step(t_ms, rx_spoofer_sym, where="post")
    ax2.set_ylabel("Amp")
    ax2.set_title(f"2) Señal que recibe el spoofer: TX + AWGN (C/N0={CN0_SPOOFER_DBHZ:.1f} dB-Hz)")
    ax2.grid(True)
    add_vlines(ax2)

    # 3) Salida del spoofer
    ax3.step(t_ms, spoofed_sym, where="post")
    ax3.set_ylabel("Amp")
    ax3.set_title("3) Señal spoofeada generada por el spoofer (salida atacante)")
    ax3.grid(True)
    add_vlines(ax3)

    # 4) Correlación acumulada total
    ax4.plot(t_ms, corr_cum)
    ax4.set_ylabel("Sum corr")
    ax4.set_title("4) Correlación acumulada total (receptor): Σ local[i]·rx[i]")
    ax4.grid(True)
    add_vlines(ax4)

    # 5) Correlación acumulada normalizada
    ax5.plot(t_ms, corr_norm)
    ax5.set_xlabel("Tiempo dentro del símbolo (ms)")
    ax5.set_ylabel("Mean corr")
    ax5.set_title("5) Correlación acumulada normalizada: (Σ local[i]·rx[i]) / N")
    ax5.grid(True)
    add_vlines(ax5)

    # (Opcional) anotación textual para que quede autoexplicativo
    if stab_valid:
        ax5.text(0.01, 0.05,
                 f"stabilization_idx={stab_idx} chips ({t_stab_ms:.3f} ms)",
                 transform=ax5.transAxes)

    plt.tight_layout()
    plt.show()

# ======================================================================
# 4. Escenario 2: MUCHOS símbolos – stats de R1, R2, R3
# ======================================================================

def run_many_symbols_stats() -> tuple[int, int]:
    (e1b_chips, local_code_samples, tx_samples,
     samples_per_symbol, chip_rate_hz, bits) = build_signals(NUM_SYMBOLS_STATS, seed_bits=11)

    # Entrada del spoofer
    chan_spoofer = AWGNChannel(cn0_db_hz=CN0_SPOOFER_DBHZ, chip_rate_hz=chip_rate_hz, seed=100)
    rx_spoofer = chan_spoofer.add_noise(tx_samples)

    # Spoof (con debug para bhat_running)
    spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=101)
    spoofed, bhat_final, _lock_idx, bhat_running = spoofer.spoof_from_rx(
        rx_samples=rx_spoofer,
        local_code=e1b_chips,
        samples_per_symbol=samples_per_symbol,
        return_debug=True
    )

    stab_idx = stabilization_idx(bhat_running, min_run=32)
    
    # k_rep = pick_representative_symbol(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR, bits, desired_bit=+1.0)
    # print(f"Representative (+1) symbol k={k_rep}, stab_idx={stab_idx[k_rep]} chips, b={bits[k_rep]:+.0f}")

    k_pos = pick_representative_symbol(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR, bits, desired_bit=+1.0)
    k_neg = pick_representative_symbol(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR, bits, desired_bit=-1.0)

    print(f"k_pos={k_pos} stab={stab_idx[k_pos]} b={bits[k_pos]:+.0f}")
    print(f"k_neg={k_neg} stab={stab_idx[k_neg]} b={bits[k_neg]:+.0f}")


    
    stab_stats = summarize_lock_idx(stab_idx, samples_per_symbol)

    # Canal al receptor
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=200)
    noisy_recv = chan_rx.add_noise(spoofed)

    correlator = PartialCorrelator(window_fraction=WINDOW_FRACTION_CORR)
    Bbeg, Bend, bhat = correlator.partial_correlations(
        local_code_samples,
        noisy_recv,
        samples_per_symbol,
    )

    ber = spoofer_ber(bits, bhat_final)
    sym_feats = compute_symbol_features(Bbeg, Bend)

    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)
    frac_stable_early = float(np.mean(stab_idx <= wlen))

    corr_stats = {
        "corr_stab_absBbeg": pearson_corr(stab_idx, np.abs(Bbeg)),
        "corr_stab_absBend": pearson_corr(stab_idx, np.abs(Bend)),
        "corr_stab_absdB":   pearson_corr(stab_idx, np.abs(Bend - Bbeg)),
    }

    print_indicator_report(
        title=f"Indicators (CN0_spoofer={CN0_SPOOFER_DBHZ:.1f} dB-Hz, CN0_rx={CN0_RX_DBHZ:.1f} dB-Hz)",
        ber=ber,
        stab_stats=stab_stats,
        frac_stable_early=frac_stable_early,
        sym_feats=sym_feats,
        corr_stats=corr_stats,
    )

    # ===== PLOTS REPRESENTATIVOS (muchos símbolos) =====
    plot_stabilization_distribution(stab_idx, samples_per_symbol, WINDOW_FRACTION_CORR)
    plot_scatter_stab_vs_gap(stab_idx, Bbeg, Bend)
    plot_beg_end_distributions(Bbeg, Bend, samples_per_symbol, WINDOW_FRACTION_CORR)

    # Convergencia: usa la señal de ataque que ya tienes (noisy_recv)
    plot_metrics_convergence(
        correlator=correlator,
        local_code_samples=local_code_samples,
        tx_samples=tx_samples,
        noisy_recv_attack=noisy_recv,
        bits=bits,
        samples_per_symbol=samples_per_symbol,
        cn0_rx_dbhz=CN0_RX_DBHZ,
        chip_rate_hz=chip_rate_hz,
    )


    return k_pos, k_neg



    # metrics_all = correlator.r_metrics(Bbeg, Bend)
    # print("\n=== Métricas R1, R2, R3 con todos los símbolos ===")
    # print(f"  Num símbolos: {NUM_SYMBOLS_STATS}")
    # print(f"  R1 = {metrics_all['R1']:.6f}")
    # print(f"  R2 = {metrics_all['R2']:.6f}")
    # print(f"  R3 = {metrics_all['R3']:.6f}")

    # # 5) Estadísticas como función del nº de símbolos (prefix)
    # R1_list = []
    # R2_list = []
    # R3_list = []

    # for k in range(1, NUM_SYMBOLS_STATS + 1):
    #     metrics_k = correlator.r_metrics(Bbeg[:k], Bend[:k])
    #     R1_list.append(metrics_k["R1"])
    #     R2_list.append(metrics_k["R2"])
    #     R3_list.append(metrics_k["R3"])

    # R1_arr = np.array(R1_list)
    # R2_arr = np.array(R2_list)
    # R3_arr = np.array(R3_list)

    # # Cómo se diferencian los símbolos:
    # # - auth_samples y noisy_recv son vectores 1D con (NUM_SYMBOLS_STATS * samples_per_symbol) chips
    # # - el símbolo k-esimo es el bloque:
    # #     [k*samples_per_symbol : (k+1)*samples_per_symbol]
    # # - PartialCorrelator hace internamente justo este reshape:
    # #     (num_symbols, samples_per_symbol)

    # # 6) Plot de R2 y R3 vs número de símbolos utilizados
    # x = np.arange(1, NUM_SYMBOLS_STATS + 1)

    # plt.figure(figsize=(8, 4))
    # plt.plot(x, R2_arr, label="R2 (prefix)")
    # plt.plot(x, R3_arr, label="R3 (prefix)")
    # plt.xlabel("Número de símbolos usados")
    # plt.ylabel("Valor del estadístico")
    # plt.title(f"Evolución de R2 y R3 con el número de símbolos\n"
    #           f"(C/N0={CN0_DBHZ} dB-Hz, error_prob={ERROR_PROB}, "
    #           f"ventana spoof={WINDOW_FRACTION_SPOOF:.2f}, "
    #           f"ventana corr={WINDOW_FRACTION_CORR:.2f})")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

def summarize_lock_idx(lock_idx: np.ndarray, samples_per_symbol: int) -> dict:
    lock_idx = np.asarray(lock_idx, dtype=int)
    never_lock = np.mean(lock_idx >= samples_per_symbol)
    valid = lock_idx[lock_idx < samples_per_symbol]

    out = {
        "lock_never_frac": float(never_lock),
        "lock_mean": float(np.mean(valid)) if valid.size else float("nan"),
        "lock_median": float(np.median(valid)) if valid.size else float("nan"),
        "lock_p10": float(np.percentile(valid, 10)) if valid.size else float("nan"),
        "lock_p90": float(np.percentile(valid, 90)) if valid.size else float("nan"),
    }
    return out

def spoofer_ber(bits_true: np.ndarray, bhat_final: np.ndarray) -> float:
    bits_true = np.asarray(bits_true, dtype=float)
    bhat_final = np.asarray(bhat_final, dtype=float)

    # Trata 0 como error (si usas prelock_mode="zero")
    bhat_hard = np.where(bhat_final >= 0, 1.0, -1.0)
    ber = float(np.mean(bhat_hard != bits_true))
    return ber

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def compute_symbol_features(Bbeg: np.ndarray, Bend: np.ndarray) -> dict:
    Bbeg = np.asarray(Bbeg)
    Bend = np.asarray(Bend)

    # Usa magnitud para ser robusto a complejo
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

def stabilization_idx(bhat_running: np.ndarray, min_run: int = 32) -> np.ndarray:
    """
    Devuelve, para cada símbolo, el primer índice n tal que:
      - desde n hasta el final, la decisión coincide con la decisión final, y
      - al menos hay min_run muestras disponibles (robustez).
    Si no se cumple, devuelve Ns (meaning: no stabilization).
    """
    bh = np.asarray(bhat_running, dtype=float)
    Nsym, Ns = bh.shape
    final = bh[:, -1]
    idx = np.full(Nsym, Ns, dtype=int)

    for k in range(Nsym):
        target = final[k]
        ok = (bh[k] == target)

        # buscamos el primer n donde ok[n:] es todo True
        # pero además exigimos que queden min_run muestras
        for n in range(0, Ns - min_run + 1):
            if ok[n:].all():
                idx[k] = n
                break

    return idx

def print_indicator_report(title: str,
                           ber: float,
                           stab_stats: dict,
                           frac_stable_early: float,
                           sym_feats: dict,
                           corr_stats: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  Spoofer BER (bhat_final vs bits): {ber:.4f}")

    print("  Stabilization index stats (chips):")
    print(f"    never-stable frac: {stab_stats['lock_never_frac']:.3f}")
    print(f"    mean/median      : {stab_stats['lock_mean']:.1f} / {stab_stats['lock_median']:.1f}")
    print(f"    p10 / p90        : {stab_stats['lock_p10']:.1f} / {stab_stats['lock_p90']:.1f}")
    print(f"    frac stable in early window: {frac_stable_early:.3f}")

    print("  Partial-corr per-symbol magnitudes:")
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
    wlen = int(samples_per_symbol * window_fraction_corr)
    target = wlen

    stab_idx = np.asarray(stab_idx, dtype=int)
    bits = np.asarray(bits, dtype=float)

    valid = np.where((stab_idx < samples_per_symbol) & (bits == desired_bit))[0]
    if valid.size == 0:
        # fallback: ignore bit constraint
        valid = np.where(stab_idx < samples_per_symbol)[0]
        if valid.size == 0:
            return 0

    k = int(valid[np.argmin(np.abs(stab_idx[valid] - target))])
    return k

def run_cn0_spoofer_sweep(cn0_list_dbhz: list[float]):
    (e1b_chips, local_code_samples, tx_samples,
     samples_per_symbol, chip_rate_hz, bits) = build_signals(NUM_SYMBOLS_STATS, seed_bits=11)

    correlator = PartialCorrelator(window_fraction=WINDOW_FRACTION_CORR)
    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)

    # Baseline sin ataque: receptor recibe TX auténtica con su ruido
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=200)
    baseline_recv = chan_rx.add_noise(tx_samples)

    Bbeg0, Bend0, bhat0 = correlator.partial_correlations(local_code_samples, baseline_recv, samples_per_symbol)
    met0 = correlator.r_metrics(Bbeg0, Bend0)

    print("\n=== CN0 sweep (spoofer input) ===")
    print("CN0_spoofer | BER   | stab_mean | stab_p90 | frac_stab_early | R1     | R2     | R3")
    print("-"*98)
    print(f"{'BASELINE':>10} | {0.000:5.3f} | {'-':>9} | {'-':>8} | {'-':>14} |"
          f" {met0['R1']:6.3f} | {met0['R2']:6.3f} | {met0['R3']:6.1f}")

    for cn0s in cn0_list_dbhz:
        # Entrada del spoofer
        chan_spoofer = AWGNChannel(cn0_db_hz=cn0s, chip_rate_hz=chip_rate_hz, seed=100)
        rx_spoofer = chan_spoofer.add_noise(tx_samples)

        # Spoofer: importante pedir bhat_running para stabilization_idx
        spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=101)
        spoofed, bhat_final, _lock_idx, bhat_running = spoofer.spoof_from_rx(
            rx_samples=rx_spoofer,
            local_code=e1b_chips,
            samples_per_symbol=samples_per_symbol,
            return_debug=True
        )

        # Receptor
        noisy_recv = chan_rx.add_noise(spoofed)

        Bbeg, Bend, bhat = correlator.partial_correlations(local_code_samples, noisy_recv, samples_per_symbol)
        met = correlator.r_metrics(Bbeg, Bend)

        ber = spoofer_ber(bits, bhat_final)
        stab_idx = stabilization_idx(bhat_running, min_run=32)
        stab_stats = summarize_lock_idx(stab_idx, samples_per_symbol)
        frac_stable_early = float(np.mean(stab_idx <= wlen))

        print(f"{cn0s:10.1f} | {ber:5.3f} | {stab_stats['lock_mean']:9.1f} | {stab_stats['lock_p90']:8.1f} |"
              f" {frac_stable_early:14.3f} | {met['R1']:6.3f} | {met['R2']:6.3f} | {met['R3']:6.1f}")

def plot_symbol_k(e1b_chips, tx_samples, bits, chip_rate_hz, samples_per_symbol, k: int):
    # Extrae símbolo k
    start = k * samples_per_symbol
    end = (k + 1) * samples_per_symbol
    tx_sym = tx_samples[start:end]
    b0 = float(bits[k])

    # Canal spoofer
    chan_spoofer = AWGNChannel(cn0_db_hz=CN0_SPOOFER_DBHZ, chip_rate_hz=chip_rate_hz, seed=123)
    rx_spoofer_sym = chan_spoofer.add_noise(tx_sym)

    # Spoof con debug
    spoofer = Spoofer(cfg=SpooferCfg(lock_gamma=0.0), seed=42)
    spoofed_sym, bhat_final, _lock_idx, bhat_running_1 = spoofer.spoof_from_rx(
        rx_samples=rx_spoofer_sym,
        local_code=e1b_chips,
        samples_per_symbol=samples_per_symbol,
        return_debug=True
    )

    # Canal receptor
    chan_rx = AWGNChannel(cn0_db_hz=CN0_RX_DBHZ, chip_rate_hz=chip_rate_hz, seed=200)
    rx_receiver_sym = chan_rx.add_noise(spoofed_sym)

    # Replica local: elige una de las dos
    # 1) Solo código -> corr puede tender a ±1 según b
    local_replica_sym = e1b_chips

    # 2) Símbolo verificado -> corr tiende a +1 siempre
    # local_replica_sym = b0 * e1b_chips

    # Correlaciones
    prod = local_replica_sym * rx_receiver_sym
    corr_cum = np.cumsum(prod)
    n = np.arange(1, samples_per_symbol + 1, dtype=float)
    corr_norm = corr_cum / n

    # Tiempo
    t_ms = np.linspace(0.0, T_SYMBOL_MS, samples_per_symbol, endpoint=False)
    wlen = int(samples_per_symbol * WINDOW_FRACTION_CORR)
    t_early_end = (wlen / samples_per_symbol) * T_SYMBOL_MS

    # Stabilization line (sobre 1 símbolo)
    bh = np.asarray(bhat_running_1)
    if bh.ndim == 1:
        bh = bh.reshape(1, -1)
    stab = stabilization_idx(bh, min_run=32)[0]
    t_stab_ms = (stab / samples_per_symbol) * T_SYMBOL_MS if stab < samples_per_symbol else None

    # Plot 5 paneles
    fig, axs = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
    ax1, ax2, ax3, ax4, ax5 = axs

    def add_vlines(ax):
        ax.axvline(t_early_end, linestyle="--", linewidth=0.8)
        if t_stab_ms is not None:
            ax.axvline(t_stab_ms, linestyle=":", linewidth=1.2)

    # 1) símbolo real (misma escala -1..1)
    ax1.step(t_ms, tx_sym, where="post")
    ax1.set_ylabel("Amp")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title(f"1) Símbolo real (TX): b·c[n] (b={b0:+.0f})")
    ax1.grid(True); add_vlines(ax1)

    # 2) entrada spoofer
    ax2.step(t_ms, rx_spoofer_sym, where="post")
    ax2.set_ylabel("Amp")
    ax2.set_title(f"2) Entrada al spoofer (C/N0={CN0_SPOOFER_DBHZ:.1f} dB-Hz)")
    ax2.grid(True); add_vlines(ax2)

    # 3) salida spoofer
    ax3.step(t_ms, spoofed_sym, where="post")
    ax3.set_ylabel("Amp")
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_title("3) Señal spoofeada (salida atacante)")
    ax3.grid(True); add_vlines(ax3)

    # 4) correlación acumulada total
    ax4.plot(t_ms, corr_cum)
    ax4.set_ylabel("Sum corr")
    ax4.set_title("4) Correlación acumulada total: Σ local[i]·rx[i]")
    ax4.grid(True); add_vlines(ax4)

    # 5) correlación acumulada normalizada
    ax5.plot(t_ms, corr_norm)
    ax5.set_title("5) Correlación acumulada normalizada: (Σ local[i]·rx[i]) / N")
    ax5.set_xlabel("Tiempo dentro del símbolo (ms)")
    ax5.set_ylabel("Mean corr")
    ax5.grid(True); add_vlines(ax5)

    plt.tight_layout()
    plt.show()

def plot_stabilization_distribution(stab_idx: np.ndarray, samples_per_symbol: int, window_fraction_corr: float):
    stab_idx = np.asarray(stab_idx, dtype=int)
    wlen = int(samples_per_symbol * window_fraction_corr)

    valid = stab_idx[stab_idx < samples_per_symbol]
    if valid.size == 0:
        print("No valid stabilization indices to plot.")
        return

    # Histograma
    plt.figure(figsize=(9, 4))
    plt.hist(valid, bins=60)
    plt.axvline(wlen, linestyle="--", linewidth=1.2, label=f"wlen (early) = {wlen} chips")
    plt.xlabel("stabilization_idx (chips)")
    plt.ylabel("count")
    plt.title("Spoofer stabilization time distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # CDF
    xs = np.sort(valid)
    ys = np.arange(1, xs.size + 1) / xs.size

    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys)
    plt.axvline(wlen, linestyle="--", linewidth=1.2, label=f"wlen (early) = {wlen} chips")
    plt.xlabel("stabilization_idx (chips)")
    plt.ylabel("CDF")
    plt.title("CDF of spoofer stabilization time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_scatter_stab_vs_gap(stab_idx: np.ndarray, Bbeg: np.ndarray, Bend: np.ndarray):
    stab = np.asarray(stab_idx, dtype=float)
    gap = np.abs(np.asarray(Bend) - np.asarray(Bbeg))

    plt.figure(figsize=(8, 4))
    plt.scatter(stab, gap, s=8, alpha=0.5)
    plt.xlabel("stabilization_idx (chips)")
    plt.ylabel("|Bend - Bbeg|")
    plt.title("Per-symbol gap vs spoofer stabilization time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_beg_end_distributions(Bbeg: np.ndarray, Bend: np.ndarray, samples_per_symbol: int, window_fraction_corr: float):
    wlen = int(samples_per_symbol * window_fraction_corr)
    absBbeg = np.abs(np.asarray(Bbeg))
    absBend = np.abs(np.asarray(Bend))

    plt.figure(figsize=(9, 4))
    plt.hist(absBbeg, bins=60, alpha=0.6, label="|Bbeg|")
    plt.hist(absBend, bins=60, alpha=0.6, label="|Bend|")
    plt.axvline(wlen, linestyle="--", linewidth=1.2, label=f"wlen={wlen} (ideal, no noise)")
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
                             bits: np.ndarray,
                             samples_per_symbol: int,
                             cn0_rx_dbhz: float,
                             chip_rate_hz: float):
    # Baseline: receptor recibe tx auténtica + AWGN_rx
    chan_rx = AWGNChannel(cn0_db_hz=cn0_rx_dbhz, chip_rate_hz=chip_rate_hz, seed=999)
    noisy_recv_base = chan_rx.add_noise(tx_samples)

    Bbeg_a, Bend_a, bhat_a = correlator.partial_correlations(local_code_samples, noisy_recv_attack, samples_per_symbol)
    Bbeg_b, Bend_b, bhat_b = correlator.partial_correlations(local_code_samples, noisy_recv_base, samples_per_symbol)

    N = Bbeg_a.size
    x = np.arange(1, N + 1)

    R1a = np.zeros(N); R2a = np.zeros(N); R3a = np.zeros(N)
    R1b = np.zeros(N); R2b = np.zeros(N); R3b = np.zeros(N)

    for k in range(1, N + 1):
        ma = correlator.r_metrics(Bbeg_a[:k], Bend_a[:k])
        mb = correlator.r_metrics(Bbeg_b[:k], Bend_b[:k])
        R1a[k-1], R2a[k-1], R3a[k-1] = ma["R1"], ma["R2"], ma["R3"]
        R1b[k-1], R2b[k-1], R3b[k-1] = mb["R1"], mb["R2"], mb["R3"]

    # R2
    plt.figure(figsize=(9, 4))
    plt.plot(x, R2a, label="R2 attack")
    plt.plot(x, R2b, label="R2 baseline")
    plt.xlabel("Number of symbols (N)")
    plt.ylabel("R2")
    plt.title("Convergence of R2 with N (attack vs baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # R3
    plt.figure(figsize=(9, 4))
    plt.plot(x, R3a, label="R3 attack")
    plt.plot(x, R3b, label="R3 baseline")
    plt.xlabel("Number of symbols (N)")
    plt.ylabel("R3")
    plt.title("Convergence of R3 with N (attack vs baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # R1 (opcional; suele ser redundante con R2)
    plt.figure(figsize=(9, 4))
    plt.plot(x, R1a, label="R1 attack")
    plt.plot(x, R1b, label="R1 baseline")
    plt.xlabel("Number of symbols (N)")
    plt.ylabel("R1")
    plt.title("Convergence of R1 with N (attack vs baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================
# 5. main
# ======================================================================

if __name__ == "__main__":
    # 1) Calcula estadísticas y selecciona un símbolo representativo
    k_pos, k_neg = run_many_symbols_stats()

    # 2) Reconstruye señales (mismo seed_bits que en run_many_symbols_stats)
    # (e1b_chips, local_code_samples, tx_samples,
    #  samples_per_symbol, chip_rate_hz, bits) = build_signals(NUM_SYMBOLS_STATS, seed_bits=11)

    # 3) Plotea el símbolo representativo
    #plot_symbol_k(e1b_chips, tx_samples, bits, chip_rate_hz, samples_per_symbol, k_pos)
    # plot_symbol_k(e1b_chips, tx_samples, bits, chip_rate_hz, samples_per_symbol, k_neg)

    # 4) Barrido de C/N0 del spoofer
    run_cn0_spoofer_sweep(cn0_list_dbhz=[30.0, 35.0, 40.0, 45.0, 50.0])