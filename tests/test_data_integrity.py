"""Testy plikow zapisywanych przez potok.

Dzialaja na danych z repozytorium, wiec zapala sie, gdy ktorys etap zostanie
puszczony ze zmieniona specyfikacja i warstwy przestana sie zgadzac.
"""
import numpy as np
import pytest

from src import paths
from src.data_load import CRYPTO_LIST, load_interim, load_processed, load_raw

RAW_COLUMNS = {'Date', 'Close'}
INTERIM_COLUMNS = {'Date', 'Close', 'log_price', 'residuals', 'g_t', 'std_residuals'}
PROCESSED_COLUMNS = INTERIM_COLUMNS | {'short_term_h', 'total_volatility', 'model_type'}

MIN_OBSERVATIONS = 2900


@pytest.fixture(scope='module', params=CRYPTO_LIST)
def crypto(request):
    return request.param


def test_every_series_has_all_three_layers(crypto):
    for folder in (paths.DATA_RAW, paths.DATA_INTERIM, paths.DATA_PROCESSED):
        assert (folder / f'{crypto}.csv').is_file(), f'{crypto} missing in {folder.name}'


def test_raw_columns(crypto):
    assert RAW_COLUMNS <= set(load_raw(crypto).columns)


def test_interim_columns(crypto):
    assert INTERIM_COLUMNS <= set(load_interim(crypto).columns)


def test_processed_columns(crypto):
    assert PROCESSED_COLUMNS <= set(load_processed(crypto).columns)


def test_sample_covers_the_declared_range(crypto):
    frame = load_raw(crypto)
    assert len(frame) >= MIN_OBSERVATIONS
    assert str(frame['Date'].min().date()) == '2018-01-01'
    assert str(frame['Date'].max().date()) == '2025-12-30'


def test_dates_are_unique_and_sorted(crypto):
    dates = load_interim(crypto)['Date']
    assert dates.is_monotonic_increasing
    assert not dates.duplicated().any()


def test_volatility_trend_is_strictly_positive(crypto):
    """g(t/T) jest odchyleniem standardowym; wartosc niedodatnia oznaczalaby
    dzielenie przez zero w standaryzacji i bezsensowna skale bootstrapu."""
    g_t = load_interim(crypto)['g_t'].to_numpy()
    assert np.all(np.isfinite(g_t))
    assert np.all(g_t > 0)


def test_standardisation_is_invertible(crypto):
    """Pomnozenie reszty standaryzowanej przez g_t musi odtworzyc reszte."""
    frame = load_interim(crypto)
    recovered = frame['std_residuals'].to_numpy() * frame['g_t'].to_numpy()
    np.testing.assert_allclose(recovered, frame['residuals'].to_numpy(), rtol=1e-9)


def test_no_missing_values_in_the_modelling_columns(crypto):
    frame = load_interim(crypto)
    for column in ('log_price', 'residuals', 'g_t', 'std_residuals'):
        assert frame[column].notna().all(), f'{crypto}: NaN in {column}'


def test_total_volatility_is_the_product_of_both_components(crypto):
    """Zmiennosc calkowita to g(t/T) razy odchylenie krotkookresowe."""
    frame = load_processed(crypto)
    short_run = np.sqrt(frame['short_term_h'].to_numpy())
    expected = frame['g_t'].to_numpy() * short_run
    np.testing.assert_allclose(
        frame['total_volatility'].to_numpy(), expected, rtol=1e-9
    )


def test_model_label_is_one_of_the_two_specifications(crypto):
    labels = set(load_processed(crypto)['model_type'].unique())
    assert len(labels) == 1
    assert labels <= {'GARCH(1,1)', 'GJR-GARCH(1,1,1)'}


def test_interim_and_processed_describe_the_same_observations(crypto):
    interim = load_interim(crypto)
    processed = load_processed(crypto)
    assert len(interim) == len(processed)
    np.testing.assert_array_equal(
        interim['Date'].to_numpy(), processed['Date'].to_numpy()
    )


def test_log_price_matches_the_closing_price(crypto):
    frame = load_interim(crypto)
    np.testing.assert_allclose(
        frame['log_price'].to_numpy(), np.log(frame['Close'].to_numpy()), rtol=1e-12
    )
