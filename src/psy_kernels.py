"""Jadra obliczeniowe rekurencyjnych testow pierwiastka jednostkowego PSY.

Wydzielone z modulu validation, bo to jedyny kod projektu zalezny od GPU:
budowa sum prefiksowych idzie wielowatkowo na CPU, a jadro BSADF wymaga
urzadzenia CUDA.

Regresja ADF(1) dy_t = b0 + b1*y_{t-1} + b2*dy_{t-1} + u_t liczona jest dla
kazdego dopuszczalnego okna. Liczenie jej niezaleznie powtarzaloby te same
sumy dla silnie nachodzacych na siebie okien, wiec dziewiec iloczynow
krzyzowych z rownan normalnych akumulujemy raz jako sumy prefiksowe; suma dla
dowolnego okna jest wtedy roznica dwoch wpisow.
"""
import math

import numpy as np
from numba import cuda, jit, prange

# Liczba wierszy sum prefiksowych; kolejnosc akumulatorow w build_prefix_sums.
N_TERMS = 9

# Zwracane, gdy okna nie da sie oszacowac; na tyle male, by nigdy nie wygralo.
SKIP = -1e12


@jit(nopython=True, parallel=True, fastmath=True)
def build_prefix_sums(series_matrix):
    """Akumuluje iloczyny krzyzowe ADF dla kazdego szeregu.

    series_matrix ma ksztalt [N, T], wynik [9, T, N]. Uklad transponowany
    stawia kolejne szeregi obok siebie w pamieci, czyli dokladnie tak, jak
    czyta je jadro GPU w jednym dostepie scalonym.
    """
    n_series, n_time = series_matrix.shape
    out = np.zeros((N_TERMS, n_time, n_series), dtype=np.float64)

    for i in prange(n_series):
        acc_y = acc_dy = acc_y2 = acc_dy2 = acc_ydy = 0.0
        acc_t = acc_yt = acc_dyt = acc_tt = 0.0

        # Od 2, bo regresja potrzebuje y_{t-1} oraz y_{t-2}.
        for k in range(2, n_time):
            y_prev = series_matrix[i, k - 1]
            y_prev2 = series_matrix[i, k - 2]

            target = series_matrix[i, k] - y_prev      # dy_t
            val_y = y_prev                              # y_{t-1}
            val_dy = y_prev - y_prev2                   # dy_{t-1}

            acc_y += val_y
            acc_dy += val_dy
            acc_y2 += val_y * val_y
            acc_dy2 += val_dy * val_dy
            acc_ydy += val_y * val_dy
            acc_t += target
            acc_yt += val_y * target
            acc_dyt += val_dy * target
            acc_tt += target * target

            out[0, k, i] = acc_y
            out[1, k, i] = acc_dy
            out[2, k, i] = acc_y2
            out[3, k, i] = acc_dy2
            out[4, k, i] = acc_ydy
            out[5, k, i] = acc_t
            out[6, k, i] = acc_yt
            out[7, k, i] = acc_dyt
            out[8, k, i] = acc_tt

    return out


@cuda.jit(device=True, inline=True)
def adf_tstat(prefix, idx, start_idx, end_idx):
    """Statystyka t przy y_{t-1} dla jednego okna albo SKIP, gdy sie nie da.

    Uklad rownan normalnych 3x3 rozwiazuja wzory Cramera, nie odwracanie
    macierzy, co utrzymuje cala regresje w rejestrach.
    """
    if start_idx >= end_idx:
        return SKIP

    n_obs = float(end_idx - start_idx + 1)
    if n_obs < 4.0:
        return SKIP

    prev = start_idx - 1

    s_1 = n_obs
    s_y = prefix[0, end_idx, idx] - prefix[0, prev, idx]
    s_dy = prefix[1, end_idx, idx] - prefix[1, prev, idx]
    s_y2 = prefix[2, end_idx, idx] - prefix[2, prev, idx]
    s_dy2 = prefix[3, end_idx, idx] - prefix[3, prev, idx]
    s_ydy = prefix[4, end_idx, idx] - prefix[4, prev, idx]
    s_t = prefix[5, end_idx, idx] - prefix[5, prev, idx]
    s_yt = prefix[6, end_idx, idx] - prefix[6, prev, idx]
    s_dyt = prefix[7, end_idx, idx] - prefix[7, prev, idx]
    s_tt = prefix[8, end_idx, idx] - prefix[8, prev, idx]

    det = (s_1 * (s_y2 * s_dy2 - s_ydy * s_ydy)
           - s_y * (s_y * s_dy2 - s_ydy * s_dy)
           + s_dy * (s_y * s_ydy - s_y2 * s_dy))
    if abs(det) < 1e-12:
        return SKIP

    det_b0 = (s_t * (s_y2 * s_dy2 - s_ydy * s_ydy)
              - s_y * (s_yt * s_dy2 - s_ydy * s_dyt)
              + s_dy * (s_yt * s_ydy - s_y2 * s_dyt))
    det_b1 = (s_1 * (s_yt * s_dy2 - s_ydy * s_dyt)
              - s_t * (s_y * s_dy2 - s_ydy * s_dy)
              + s_dy * (s_y * s_dyt - s_yt * s_dy))
    det_b2 = (s_1 * (s_y2 * s_dyt - s_yt * s_ydy)
              - s_y * (s_y * s_dyt - s_yt * s_dy)
              + s_t * (s_y * s_ydy - s_y2 * s_dy))

    beta_0 = det_b0 / det
    beta_1 = det_b1 / det
    beta_2 = det_b2 / det

    sse = s_tt - (beta_0 * s_t + beta_1 * s_yt + beta_2 * s_dyt)
    dof = n_obs - 3.0
    if sse <= 1e-12 or dof <= 0.0:
        return SKIP

    cov_11 = (s_1 * s_dy2 - s_dy * s_dy) / det
    if cov_11 < 1e-12:
        return SKIP

    return beta_1 / math.sqrt((sse / dof) * cov_11)


@cuda.jit
def bsadf_kernel(prefix, min_window, out_sequence):
    """Sekwencja Backward Sup ADF, jeden watek na szereg.

    Dla kazdego punktu koncowego r2 brane jest supremum po wszystkich
    dopuszczalnych poczatkach r1. Maksimum powstalej sekwencji jest statystyka
    GSADF, wiec osobne jadro GSADF nie jest potrzebne.

    Pozycje przed min_window zostaja nietkniete i nie wolno ich czytac.
    """
    idx = cuda.grid(1)
    if idx >= prefix.shape[2]:
        return

    n_time = prefix.shape[1]

    for r2 in range(min_window, n_time):
        best = SKIP
        for r1 in range(0, r2 - min_window + 1):
            value = adf_tstat(prefix, idx, r1 + 2, r2)
            if value > best:
                best = value
        out_sequence[idx, r2] = best
