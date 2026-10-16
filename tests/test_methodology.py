"""Testy gwarancji, na ktorych stoi metoda.

Kazdy test poza dwoma oznaczonymi gpu dziala bez urzadzenia CUDA, wiec zestaw
da sie uruchomic na maszynie, ktora w ogole nie udzwignie etapu PSY.
"""
import numpy as np
import pandas as pd
import pytest

from src import validation
from src.psy_kernels import N_TERMS, build_prefix_sums


# --- Wild bootstrap ------------------------------------------------------

def test_mammen_weights_have_unit_second_and_third_moments():
    """Rozklad Mammena musi spelniac E(v)=0, E(v^2)=1, E(v^3)=1.

    Te trzy momenty sprawiaja, ze wild bootstrap odtwarza skosnosc stop zwrotu
    o niestalej wariancji; bledna para punktow nosnika albo prawdopodobienstw
    zepsulaby rozklad zerowy, nie zglaszajac zadnego bledu.
    """
    rng = np.random.default_rng(7)
    draws = validation.mammen_weights(2_000_000, rng)

    assert abs(draws.mean()) < 0.005
    assert abs((draws ** 2).mean() - 1.0) < 0.005
    assert abs((draws ** 3).mean() - 1.0) < 0.01


def test_mammen_support_is_the_two_published_points():
    rng = np.random.default_rng(7)
    draws = validation.mammen_weights(10_000, rng)

    sqrt5 = np.sqrt(5.0)
    expected = sorted([(1.0 - sqrt5) / 2.0, (1.0 + sqrt5) / 2.0])
    assert sorted(np.unique(draws)) == pytest.approx(expected)


def test_bootstrap_increments_keep_the_scale_of_the_residuals():
    """Ponowne wprowadzenie trendu musi uzyc g_t, nie jego pierwiastka.

    Reszte standaryzowana otrzymano przez dzielenie przez g_t, wiec mnozenie
    przez g_t przywraca pierwotna skale. Mnozenie przez sqrt(g_t) zostawia
    sciezki zerowe z bledna zmiennoscia i po cichu przesuwa kazda p-wartosc;
    jedynym widocznym objawem jest niezgodnosc skali, ktora mierzy ten test.
    """
    rng = np.random.default_rng(11)
    n = 4000
    g_t = np.linspace(0.02, 0.08, n)
    residuals = rng.normal(scale=g_t)
    std_residuals = residuals / g_t
    log_price = np.cumsum(np.r_[0.0, residuals]) [:n] + 9.0

    null_paths, _ = validation.wild_bootstrap_paths(
        std_residuals, g_t, log_price, n_paths=200, rng=rng
    )
    increments = np.diff(null_paths, axis=1)

    assert increments.std() == pytest.approx(residuals.std(), rel=0.1)


def test_bootstrap_drift_is_the_median_log_price_increment():
    rng = np.random.default_rng(3)
    n = 500
    log_price = np.cumsum(rng.normal(loc=0.004, scale=0.01, size=n)) + 9.0
    g_t = np.full(n, 0.01)
    std_residuals = rng.normal(size=n)

    null_paths, drift = validation.wild_bootstrap_paths(
        std_residuals, g_t, log_price, n_paths=400, rng=rng
    )

    assert drift == pytest.approx(np.median(np.diff(log_price)))

    # Dryf musi trafic do sciezek, nie tylko zostac zwrocony: bez niego rozklad
    # zerowy jest bladzeniem bez dryfu, a kazda p-wartosc wychodzi za mala.
    assert np.diff(null_paths, axis=1).mean() == pytest.approx(drift, abs=2e-4)


def test_null_paths_have_the_same_length_as_the_data():
    """Krzywa jest zakotwiczona w ostatniej obserwacji, wiec dlugosci musza sie zgadzac."""
    rng = np.random.default_rng(5)
    n = 300
    null_paths, _ = validation.wild_bootstrap_paths(
        rng.normal(size=n), np.full(n, 0.01), np.linspace(9.0, 10.0, n),
        n_paths=8, rng=rng,
    )
    assert null_paths.shape == (8, n)


# --- Krzywa wartosci krytycznych -----------------------------------------

def test_critical_value_curve_rises_with_the_sample():
    """CV(t) = log(log(t))/c musi rosnac; to kontroluje blad I rodzaju."""
    curve = validation.critical_value_curve(3000, scale=1.5)
    tail = curve[10:]
    assert np.all(np.diff(tail) > 0)


def test_calibration_anchors_the_curve_at_the_requested_quantile():
    rng = np.random.default_rng(13)
    n_time = 2920
    boot = rng.normal(loc=1.2, scale=0.4, size=(5000, n_time))

    scale, end_quantile = validation.calibrate_cv_scale(boot, n_time, quantile=99)
    curve = validation.critical_value_curve(n_time, scale)

    assert end_quantile == pytest.approx(np.percentile(boot[:, -1], 99))
    assert curve[-1] == pytest.approx(end_quantile, rel=1e-9)


def test_calibration_uses_the_upper_one_percent_not_five():
    rng = np.random.default_rng(17)
    boot = rng.normal(loc=1.0, scale=0.3, size=(20_000, 500))

    _, at_one = validation.calibrate_cv_scale(boot, 500, quantile=99)
    _, at_five = validation.calibrate_cv_scale(boot, 500, quantile=95)

    assert validation.CV_QUANTILE == 99
    assert at_one > at_five


# --- Datowanie epizodow --------------------------------------------------

def _dates(n):
    return pd.date_range('2020-01-01', periods=n, freq='D').to_numpy()


def test_episode_boundaries_are_the_crossing_points():
    statistic = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0])
    critical = np.ones(7)

    episodes = validation.date_episodes(_dates(7), statistic, critical, min_duration=1)

    assert len(episodes) == 1
    assert str(episodes[0]['start']) == '2020-01-03'
    assert str(episodes[0]['end']) == '2020-01-05'
    assert episodes[0]['duration_days'] == 2


def test_two_runs_separated_by_one_observation_stay_separate():
    statistic = np.array([2.0, 2.0, 0.0, 2.0, 2.0])
    episodes = validation.date_episodes(_dates(5), statistic, np.ones(5), min_duration=1)
    assert len(episodes) == 2


def test_run_touching_the_end_of_the_sample_is_closed():
    statistic = np.array([0.0, 2.0, 2.0])
    episodes = validation.date_episodes(_dates(3), statistic, np.ones(3), min_duration=1)
    assert len(episodes) == 1
    assert str(episodes[0]['end']) == '2020-01-03'


def test_no_exceedance_gives_no_episodes():
    episodes = validation.date_episodes(
        _dates(50), np.zeros(50), np.ones(50), min_duration=1
    )
    assert episodes == []


def test_min_duration_is_reported_and_does_not_filter():
    statistic = np.array([0.0, 2.0, 2.0, 0.0])
    episodes = validation.date_episodes(_dates(4), statistic, np.ones(4), min_duration=8)

    assert len(episodes) == 1
    assert episodes[0]['meets_min_duration'] is False


# --- Sumy prefiksowe -----------------------------------------------------

def _direct_sums(series, start_idx, end_idx):
    """Iloczyny krzyzowe regresji ADF(1) policzone bez sum prefiksowych."""
    target, y_lag, dy_lag = [], [], []
    for k in range(start_idx, end_idx + 1):
        target.append(series[k] - series[k - 1])
        y_lag.append(series[k - 1])
        dy_lag.append(series[k - 1] - series[k - 2])

    target = np.array(target)
    y_lag = np.array(y_lag)
    dy_lag = np.array(dy_lag)

    return np.array([
        y_lag.sum(), dy_lag.sum(), (y_lag ** 2).sum(), (dy_lag ** 2).sum(),
        (y_lag * dy_lag).sum(), target.sum(), (y_lag * target).sum(),
        (dy_lag * target).sum(), (target ** 2).sum(),
    ])


@pytest.mark.parametrize('start_idx, end_idx', [(2, 60), (15, 99), (40, 119)])
def test_prefix_sums_reproduce_the_window_cross_products(start_idx, end_idx):
    """Sumy okna musza rownac sie roznicy dwoch wpisow prefiksowych.

    Na tej optymalizacji stoi jadro GPU: przy blednej kolejnosci akumulacji
    albo ukladzie wierszy kazda statystyka ADF bylaby bledna, a kod i tak by
    sie wykonal.
    """
    rng = np.random.default_rng(23)
    series = np.cumsum(rng.normal(size=120)) + 9.0

    prefix = build_prefix_sums(series.reshape(1, -1))
    from_prefix = np.array([
        prefix[term, end_idx, 0] - prefix[term, start_idx - 1, 0]
        for term in range(N_TERMS)
    ])

    np.testing.assert_allclose(
        from_prefix, _direct_sums(series, start_idx, end_idx), rtol=1e-10
    )


def test_prefix_sums_are_zero_before_the_third_observation():
    rng = np.random.default_rng(29)
    prefix = build_prefix_sums(np.cumsum(rng.normal(size=50)).reshape(1, -1))
    assert np.all(prefix[:, :2, :] == 0.0)


# --- Jadro GPU -----------------------------------------------------------

gpu = pytest.mark.skipif(
    not validation.gpu_available(), reason='Etap PSY wymaga karty NVIDIA.'
)


@gpu
def test_bsadf_statistic_matches_an_independent_ols():
    """Statystyka t z jadra musi rownac sie tej z jawnej regresji.

    Jadro rozwiazuje rownania normalne wzorami Cramera na sumach prefiksowych;
    ten test porownuje wynik z dopasowaniem najmniejszych kwadratow na tym
    samym oknie.
    """
    rng = np.random.default_rng(31)
    series = np.cumsum(rng.normal(size=260)) + 9.0
    min_window = 250

    sequence = validation.bsadf_sequence(series.reshape(1, -1), min_window)[0]

    # Przy min_window = 250 i r2 = 250 dopuszczalne jest jedno okno, wiec
    # supremum jest statystyka t tego okna.
    end_idx = 250
    target = series[2:end_idx + 1] - series[1:end_idx]
    design = np.column_stack([
        np.ones(end_idx - 1),
        series[1:end_idx],
        series[1:end_idx] - series[0:end_idx - 1],
    ])

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    dof = len(target) - 3
    covariance = np.linalg.inv(design.T @ design) * (residuals @ residuals) / dof
    expected = coefficients[1] / np.sqrt(covariance[1, 1])

    assert sequence[end_idx] == pytest.approx(expected, rel=1e-8)


@gpu
def test_positions_before_the_minimum_window_are_not_reported():
    """Pozycje nietkniete musza byc NaN, nigdy niezainicjowana pamiecia GPU."""
    rng = np.random.default_rng(37)
    series = np.cumsum(rng.normal(size=300)) + 9.0

    sequence = validation.bsadf_sequence(series.reshape(1, -1), min_window=200)[0]

    assert np.all(np.isnan(sequence[:200]))
    assert not np.any(np.isnan(sequence[200:]))
