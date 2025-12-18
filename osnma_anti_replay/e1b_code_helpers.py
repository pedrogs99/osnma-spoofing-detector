# e1b_code_helpers.py
from __future__ import annotations
import numpy as np


def hex_to_bits_be(hex_str: str, n_bits: int) -> np.ndarray:
    """
    Convierte una cadena hex en un array de bits {0,1}, MSB primero.

    - hex_str: cadena como viene del ICD (puede llevar espacios).
    - n_bits: longitud total deseada (para E1B: 4092 bits).

    Devuelve un array np.uint8 de tamaño n_bits con valores 0/1.
    """
    # limpia espacios y posibles prefijos
    s = hex_str.replace(" ", "").replace("\n", "").replace("\t", "")
    s = s.replace("0x", "").replace("0X", "")

    # valor entero
    value = int(s, 16)

    # cadena binaria sin '0b', con zero-pad a n_bits
    bits_str = bin(value)[2:].zfill(n_bits)

    # pásalo a array de enteros 0/1
    bits = np.fromiter(bits_str, dtype="U1") == "1"
    return bits.astype(np.uint8)


def bits_to_polar(bits: np.ndarray) -> np.ndarray:
    """
    Mapea 0 -> +1, 1 -> -1 (como quieres).

    Devuelve un array float64 con valores +1.0 / -1.0.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    return np.where(bits == 0, 1.0, -1.0)


def e1b_hex_to_chips(hex_str: str) -> np.ndarray:
    """
    Convierte un código E1B en hex (4092 chips) a una secuencia polar +/-1.
    """
    N_CHIPS = 4092
    bits = hex_to_bits_be(hex_str, n_bits=N_CHIPS)
    chips = bits_to_polar(bits)
    return chips
