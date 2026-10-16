"""Warunkowa wartosc oczekiwana i deterministyczna czesc struktury bledu.

Za Hafnerem (2020) reszta AR rozklada sie jako eps_t = g(t/T) * sqrt(h_t) * eta_t,
gdzie g(t/T) jest deterministyczna funkcja odchylenia standardowego w czasie
przeskalowanym. Modul estymuje g(t/T) lokalna jadrowa regresja liniowa na
kwadratach reszt i standaryzuje szereg przez te funkcje; skladnik stochastyczny
h_t zostaje dla modulu model.
"""
import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from src import paths
from src.data_load import CRYPTO_LIST, date_column, load_raw

MAX_LAGS = 30
MIN_OBSERVATIONS = 50


def fit_ar_bic(returns, max_lags=MAX_LAGS):
    """Dopasowuje AR(p) do stop zwrotu, rzad wybiera BIC. Zwraca (reszty, p)."""
    selection = ar_select_order(returns, maxlag=max_lags, ic='bic', trend='c', glob=False)
    lags = selection.ar_lags

    if lags is None or len(lags) == 0:
        # BIC wskazuje p=0: brak autokorelacji liniowej, model to sama stala.
        return AutoReg(returns, lags=0, trend='c').fit().resid, 0

    return selection.model.fit().resid, max(lags)


def estimate_variance_trend(residuals):
    """Estymuje g(t/T) lokalna jadrowa regresja liniowa na kwadratach reszt.

    Obserwowalne sa kwadraty reszt, wiec regresja estymuje lokalna wariancje,
    a g(t/T) jest odchyleniem standardowym, stad pierwiastek.
    Zwraca (g_t, trend wariancji, szerokosc pasma).
    """
    rescaled_time = np.linspace(0, 1, len(residuals))

    regression = KernelReg(
        endog=residuals ** 2,
        exog=rescaled_time,
        var_type='c',
        reg_type='ll',
        bw='cv_ls',
        ckertype='gaussian',
    )
    variance_trend, _ = regression.fit(rescaled_time)

    # Regresja lokalnie liniowa nie jest ograniczona do wartosci nieujemnych.
    variance_trend = np.maximum(variance_trend, 1e-10)

    bandwidth = regression.bw
    bandwidth = float(bandwidth[0]) if np.ndim(bandwidth) else float(bandwidth)

    return np.sqrt(variance_trend), variance_trend, bandwidth


def prepare_series(name, raw_dir=None, out_dir=None):
    """Przeprowadza etap sredniej warunkowej i trendu dla jednego szeregu.

    Zapisuje data/interim/<name>.csv i zwraca (ramka, diagnostyka).
    """
    out_dir = paths.DATA_INTERIM if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_raw(name, folder=raw_dir)
    col_date = date_column(frame)

    if len(frame) < MIN_OBSERVATIONS:
        raise ValueError(f'{name}: {len(frame)} obserwacji, mniej niz {MIN_OBSERVATIONS}.')

    frame['log_price'] = np.log(frame['Close'])
    frame['log_ret'] = frame['log_price'].diff()

    model_frame = frame.dropna(subset=['log_ret']).copy()
    residuals, order = fit_ar_bic(model_frame['log_ret'].to_numpy())

    # AutoReg odrzuca pierwsze p obserwacji; dopasowujemy ramke do reszt.
    aligned = model_frame.iloc[-len(residuals):].copy()
    g_t, variance_trend, bandwidth = estimate_variance_trend(residuals)

    aligned['residuals'] = residuals
    aligned['g_t'] = g_t
    aligned['variance_trend'] = variance_trend
    aligned['std_residuals'] = residuals / g_t

    columns = [col_date, 'Close', 'log_price', 'residuals', 'g_t', 'std_residuals']
    aligned[columns].to_csv(out_dir / f'{name}.csv', index=False)

    diagnostics = {
        'crypto': name,
        'n_observations': len(aligned),
        'ar_order': order,
        'bandwidth': bandwidth,
    }
    return aligned[columns], diagnostics


def run(crypto_list=None, raw_dir=None, out_dir=None):
    """Przygotowuje kazdy szereg i zwraca tabele diagnostyczna."""
    crypto_list = CRYPTO_LIST if crypto_list is None else crypto_list

    rows = []
    for name in crypto_list:
        _, diagnostics = prepare_series(name, raw_dir=raw_dir, out_dir=out_dir)
        rows.append(diagnostics)
        print(
            f"{name}: n={diagnostics['n_observations']}, "
            f"AR({diagnostics['ar_order']}), pasmo={diagnostics['bandwidth']:.4f}"
        )

    return pd.DataFrame(rows)
