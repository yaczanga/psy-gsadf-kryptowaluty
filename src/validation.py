"""Test PSY: statystyka GSADF, rozklad zerowy z wild bootstrapu, datowanie.

Implementacja Phillipsa, Shi i Yu (2015) z korekta na heteroskedastycznosc
wedlug Hafnera (2020): rozklad zerowy generuje wild bootstrap odtwarzajacy
strukture zmiennosci szeregu zrodlowego.

Cztery parametry rozkladu zerowego ida za opisem metody w pracy i roznia sie
od pierwszej zatwierdzonej wersji notebooka; szczegoly w README.
"""
import numpy as np
import pandas as pd

from src import paths
from src.data_load import CRYPTO_LIST, date_column, load_interim
# cuda pochodzi z psy_kernels, bo tam stoi zaslepka na wypadek braku sterownika.
from src.psy_kernels import bsadf_kernel, build_prefix_sums, cuda

# Minimalne okno regresji ADF, w obserwacjach.
MIN_WINDOW = 200

# Liczba sciezek bootstrapu generujacych rozklad zerowy.
N_BOOTSTRAP = 10_000

# Gorny kwantyl rozkladu symulowanego, w ktorym zakotwiczona jest krzywa
# wartosci krytycznych; 99 odpowiada poziomowi 1% raportowanemu w pracy.
CV_QUANTILE = 99

# Poziom istotnosci, przy ktorym epizod nazywamy banka.
BUBBLE_LEVEL = 0.01

# Rozmiar bloku CUDA.
THREADS_PER_BLOCK = 256

# Bootstrap jest zaziarnowany, zeby raportowane p-wartosci byly odtwarzalne.
SEED = 2026


def gpu_available():
    """Prawda, gdy jest urzadzenie CUDA; etap BSADF go wymaga.

    Zwraca falsz takze wtedy, gdy samego numba.cuda nie dalo sie
    zaimportowac, bo na maszynie nie ma biblioteki runtime'u CUDA.
    """
    return cuda.is_available()


def bsadf_sequence(series_matrix, min_window=MIN_WINDOW):
    """Sekwencja BSADF dla kazdego wiersza series_matrix o ksztalcie [N, T].

    Pozycje przed min_window nie sa liczone i wracaja jako NaN, zeby pozniejsze
    maksimum nie wciagnelo niezainicjowanej wartosci.
    """
    prefix = build_prefix_sums(series_matrix)
    n_series, n_time = series_matrix.shape

    device_prefix = cuda.to_device(prefix)
    device_out = cuda.to_device(np.full((n_series, n_time), np.nan))

    blocks = (n_series + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    bsadf_kernel[blocks, THREADS_PER_BLOCK](device_prefix, min_window, device_out)
    cuda.synchronize()

    return device_out.copy_to_host()


def mammen_weights(shape, rng):
    """Losowania z dwupunktowego rozkladu Mammena: E(v)=0, E(v^2)=E(v^3)=1."""
    sqrt5 = np.sqrt(5.0)
    probability = (sqrt5 + 1.0) / (2.0 * sqrt5)
    low = (1.0 - sqrt5) / 2.0
    high = (1.0 + sqrt5) / 2.0
    return np.where(rng.random(size=shape) <= probability, low, high)


def wild_bootstrap_paths(std_residuals, g_t, log_price, n_paths, rng):
    """Buduje sciezki zerowe o dlugosci rownej dlugosci danych.

    Przy hipotezie zerowej logarytm ceny jest bladzeniem losowym z dryfem.
    Dryf to mediana historycznych przyrostow logarytmicznych cen, co oddziela
    trend wzrostowy od eksplozji. Reszty standaryzowane zaburzamy wagami
    Mammena i mnozymy z powrotem przez g_t - jest to odchylenie standardowe,
    wiec czynnikiem jest samo g_t, nie jego pierwiastek.
    """
    n_time = len(std_residuals)
    drift = float(np.median(np.diff(log_price)))

    weights = mammen_weights((n_paths, n_time - 1), rng)
    increments = std_residuals[:n_time - 1] * weights * g_t[:n_time - 1] + drift

    null_paths = np.empty((n_paths, n_time), dtype=np.float64)
    null_paths[:, 0] = log_price[0]
    null_paths[:, 1:] = log_price[0] + np.cumsum(increments, axis=1)
    return null_paths, drift


def calibrate_cv_scale(boot_sequence, n_time, quantile=CV_QUANTILE):
    """Parametr skalujacy c krzywej CV(t) = log(log(t)) / c.

    Krzywa jest zakotwiczona w koncu proby, gdzie jej wartosc odpowiada
    zadanemu gornemu kwantylowi rozkladu symulowanego.
    """
    end_quantile = float(np.percentile(boot_sequence[:, -1], quantile))

    if n_time <= 10 or end_quantile <= 1e-5:
        return 1.0, end_quantile

    return float(np.log(np.log(n_time)) / end_quantile), end_quantile


def critical_value_curve(n_time, scale):
    """CV(t) = log(log(t)) / c dla t = 1..n_time, z wyzerowanym poczatkiem."""
    steps = np.arange(1, n_time + 1, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        curve = np.log(np.log(steps)) / scale
    return np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)


def date_episodes(dates, statistic, critical, min_duration):
    """Datuje przedzialy, w ktorych statystyka BSADF przekracza wartosc krytyczna.

    Epizod zaczyna sie tam, gdzie sekwencja po raz pierwszy przecina krzywa,
    i konczy tam, gdzie wraca ponizej. min_duration nie filtruje wyniku; kazdy
    wiersz niesie informacje, czy osiaga te dlugosc, wiec zalecany prog
    pozostaje widoczny bez usuwania krotszych przedzialow.
    """
    above = np.asarray(statistic) > np.asarray(critical)
    spans = []
    start = None

    for position, flag in enumerate(above):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            spans.append((start, position - 1))
            start = None
    if start is not None:
        spans.append((start, len(above) - 1))

    rows = []
    for first, last in spans:
        begin = pd.Timestamp(dates[first])
        finish = pd.Timestamp(dates[last])
        duration = int((finish - begin).days)
        rows.append({
            'start': begin.date(),
            'end': finish.date(),
            'duration_days': duration,
            'max_bsadf': float(np.max(statistic[first:last + 1])),
            'meets_min_duration': duration >= min_duration,
        })
    return rows


def test_series(name, interim_dir=None, n_bootstrap=N_BOOTSTRAP,
                min_window=MIN_WINDOW, seed=SEED):
    """Przeprowadza test PSY dla jednego szeregu.

    Zwraca (podsumowanie, epizody, dane do wykresu), gdzie ostatni element
    niesie daty, sekwencje BSADF i krzywa wartosci krytycznych, przyciete do
    przedzialu, w ktorym obie sa okreslone.
    """
    frame = load_interim(name, folder=interim_dir)
    col_date = date_column(frame)

    log_price = frame['log_price'].to_numpy()
    g_t = frame['g_t'].to_numpy()
    std_residuals = frame['std_residuals'].to_numpy()
    n_time = len(log_price)

    real_sequence = bsadf_sequence(log_price.reshape(1, -1), min_window)[0]
    gsadf = float(np.nanmax(real_sequence))

    rng = np.random.default_rng(seed)
    null_paths, drift = wild_bootstrap_paths(
        std_residuals, g_t, log_price, n_bootstrap, rng
    )
    boot_sequence = bsadf_sequence(null_paths, min_window)

    # Empiryczna p-wartosc: udzial sciezek zerowych o GSADF wyzszym od danych.
    boot_gsadf = np.nanmax(boot_sequence, axis=1)
    p_value = float(np.mean(boot_gsadf > gsadf))

    scale, end_quantile = calibrate_cv_scale(boot_sequence, n_time)
    curve = critical_value_curve(n_time, scale)

    # Sekwencja nie jest okreslona przed min_window; przycinamy obie tablice.
    dates = frame[col_date].to_numpy()[min_window:]
    statistic = real_sequence[min_window:]
    critical = curve[min_window:]

    min_duration = int(np.log(n_time))
    episodes = date_episodes(dates, statistic, critical, min_duration)

    summary = {
        'crypto': name,
        'n_observations': n_time,
        'gsadf': gsadf,
        'p_value': p_value,
        'bubble': p_value < BUBBLE_LEVEL,
        'cv_scale_c': scale,
        'cv_end_quantile': end_quantile,
        'drift': drift,
        'min_duration_days': min_duration,
        'n_episodes': len(episodes),
    }
    plot_data = {'dates': dates, 'statistic': statistic, 'critical': critical}
    return summary, episodes, plot_data


def run(crypto_list=None, interim_dir=None, n_bootstrap=N_BOOTSTRAP,
        min_window=MIN_WINDOW, seed=SEED, make_plots=True):
    """Przeprowadza test PSY dla kazdego szeregu, zapisuje tabele i wykresy."""
    if not gpu_available():
        raise RuntimeError(
            'Etap BSADF wymaga karty NVIDIA. Wczesniejsze etapy dzialaja bez '
            'niej; patrz README, sekcja Uruchomienie.'
        )

    crypto_list = CRYPTO_LIST if crypto_list is None else crypto_list
    paths.ensure_directories()

    if make_plots:
        # Zapis wlacza sie jawnie; bez tego funkcje rysujace nic nie zapisuja.
        from src.visualisation import plot_bsadf, set_figures_dir
        set_figures_dir(paths.FIGURES)

    summaries, all_episodes = [], []
    for name in crypto_list:
        summary, episodes, plot_data = test_series(
            name, interim_dir=interim_dir, n_bootstrap=n_bootstrap,
            min_window=min_window, seed=seed,
        )
        summaries.append(summary)
        for episode in episodes:
            all_episodes.append({'crypto': name, **episode})

        print(
            f"{name}: GSADF={summary['gsadf']:.4f}, p={summary['p_value']:.4f}, "
            f"c={summary['cv_scale_c']:.4f}, episodes={summary['n_episodes']}"
        )

        if make_plots:
            plot_bsadf(name, plot_data, summary)

    summary_table = pd.DataFrame(summaries)
    episode_table = pd.DataFrame(all_episodes)
    summary_table.to_csv(paths.REPORTS / 'gsadf_results.csv', index=False)
    episode_table.to_csv(paths.REPORTS / 'bubble_episodes.csv', index=False)
    return summary_table, episode_table
