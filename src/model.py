"""Stochastyczna czesc struktury bledu: GARCH(1,1) i GJR-GARCH(1,1,1).

Reszty standaryzowane z modulu pre_processing niosa grupowanie zmiennosci,
ktorego g(t/T) nie pochlania. Dla kazdego szeregu estymowane sa obie
specyfikacje; asymetryczna zostaje tylko tam, gdzie parametr dzwigni jest
istotny, a kryterium informacyjne nizsze.
"""
import numpy as np
import pandas as pd
from arch import arch_model

from src import paths
from src.data_load import CRYPTO_LIST, date_column, load_interim

# Optymalizator w pakiecie arch zbiega pewniej na szeregach w skali procentowej.
SCALE = 100.0

# Poziom istotnosci dla parametru dzwigni gamma.
GAMMA_LEVEL = 0.05


def fit_garch_pair(residuals_scaled):
    """Estymuje GARCH(1,1) i GJR-GARCH(1,1,1) na tym samym szeregu."""
    symmetric = arch_model(
        residuals_scaled, vol='Garch', p=1, o=0, q=1, dist='Normal'
    ).fit(disp='off', show_warning=False)

    asymmetric = arch_model(
        residuals_scaled, vol='Garch', p=1, o=1, q=1, dist='Normal'
    ).fit(disp='off', show_warning=False)

    return symmetric, asymmetric


def select_model(symmetric, asymmetric, level=GAMMA_LEVEL):
    """Zostawia GJR-GARCH tylko przy istotnej asymetrii i nizszym BIC.

    Zwraca (wynik, etykieta, p-wartosc gammy).
    """
    gamma_pvalue = float(asymmetric.pvalues.get('gamma[1]', 1.0))

    if gamma_pvalue < level and asymmetric.bic < symmetric.bic:
        return asymmetric, 'GJR-GARCH(1,1,1)', gamma_pvalue

    return symmetric, 'GARCH(1,1)', gamma_pvalue


def estimate_series(name, interim_dir=None, out_dir=None):
    """Modeluje zmiennosc krotkookresowa jednego szeregu i laczy ja z trendem.

    Zapisuje data/processed/<name>.csv i zwraca (ramka, parametry).
    """
    out_dir = paths.DATA_PROCESSED if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_interim(name, folder=interim_dir)
    col_date = date_column(frame)

    residuals_scaled = frame['std_residuals'].to_numpy() * SCALE
    symmetric, asymmetric = fit_garch_pair(residuals_scaled)
    selected, label, gamma_pvalue = select_model(symmetric, asymmetric)

    # Cofamy skalowanie; conditional_volatility jest odchyleniem standardowym.
    short_run_sigma = selected.conditional_volatility / SCALE

    frame['short_term_h'] = short_run_sigma ** 2
    frame['total_volatility'] = frame['g_t'] * short_run_sigma
    frame['model_type'] = label

    columns = [
        col_date, 'Close', 'log_price', 'residuals', 'g_t', 'std_residuals',
        'short_term_h', 'total_volatility', 'model_type',
    ]
    frame[columns].to_csv(out_dir / f'{name}.csv', index=False)

    parameters = {'crypto': name, 'model': label}
    for key in ('omega', 'alpha[1]', 'beta[1]', 'gamma[1]'):
        value = selected.params.get(key, np.nan)
        parameters[key.replace('[1]', '')] = float(value)
    parameters['gamma_pvalue'] = gamma_pvalue
    parameters['bic'] = float(selected.bic)

    return frame[columns], parameters


def run(crypto_list=None, interim_dir=None, out_dir=None, table_path=None):
    """Modeluje kazdy szereg; zapisuje tabele parametrow i zwraca ja."""
    crypto_list = CRYPTO_LIST if crypto_list is None else crypto_list
    table_path = (paths.REPORTS / 'garch_parameters.csv') if table_path is None else table_path

    rows = []
    for name in crypto_list:
        _, parameters = estimate_series(name, interim_dir=interim_dir, out_dir=out_dir)
        rows.append(parameters)
        print(
            f"{name}: {parameters['model']}, "
            f"p-wartosc gammy={parameters['gamma_pvalue']:.3f}"
        )

    table = pd.DataFrame(rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    return table
