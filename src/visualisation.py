"""Rysunki projektu: sekwencja BSADF i dekompozycja zmiennosci.

Barwy pochodza wylacznie z tokenow modulu plot_style i sa wybierane przez
role, jaka seria pelni w danych: cynober to przedmiot uwagi, blekit to
odniesienie. Sekwencja BSADF i obszar przekroczenia nosza RISK, krzywa
wartosci krytycznych jako norma nosi REFERENCE, a modul reszt AR i zmiennosc
calkowita biora pare tozsamosci SERIES_1 i SERIES_2.

Napisy na rysunku maja diakrytyki i przecinek dziesietny, kod i komentarze
nie. Zapis do plikow jest wylaczony, dopoki set_figures_dir nie wskaze
katalogu.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend nieinteraktywny: rysunki ida do plikow, nigdy na ekran.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from src import plot_style as style
from src.data_load import date_column, display_name, load_processed

style.apply_style()

FIGURES_DIR: Path | None = None


def set_figures_dir(path: str | Path | None) -> None:
    """Ustawia katalog zapisu raz, dla wszystkich funkcji rysujacych."""
    global FIGURES_DIR
    FIGURES_DIR = None if path is None else Path(path)
    if FIGURES_DIR is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(figure, name: str | None) -> Path | None:
    """Zapisuje figure, jesli katalog zostal ustawiony i podano nazwe."""
    if FIGURES_DIR is None or not name:
        return None
    target = FIGURES_DIR / f'{name}.png'
    figure.savefig(target)
    return target


def plot_bsadf(crypto, plot_data, summary, name: str | None = None):
    """Sekwencja BSADF wobec krzywej wartosci krytycznych, z epizodami.

    Obszar miedzy sekwencja a krzywa jest cieniowany tam, gdzie statystyka
    przekracza norme. Epizody krotsze niz mniej wiecej dwa tygodnie zostawiaja
    na skali osmiu lat kreske szerokosci wlosa - daty sa w tabeli epizodow.
    """
    name = f'bsadf_{crypto.lower()}' if name is None else name

    dates = plot_data['dates']
    statistic = plot_data['statistic']
    critical = plot_data['critical']

    figure, ax = plt.subplots(figsize=style.WIDE)

    ax.plot(dates, critical, color=style.REFERENCE, linestyle='--',
            linewidth=1.4,
            label=f"wartość krytyczna 1% (c={_number(summary['cv_scale_c'], 2)})")
    ax.plot(dates, statistic, color=style.RISK, linewidth=1.2,
            label='statystyka BSADF')
    ax.fill_between(dates, statistic, critical, where=(statistic > critical),
                    color=style.RISK, alpha=0.25, linewidth=0,
                    label='okres wybuchowy')

    ax.set_title(
        f"{display_name(crypto)}: statystyka BSADF wobec wartości krytycznej, "
        f"p-wartość GSADF {_number(summary['p_value'])}")
    ax.set_ylabel('statystyka ADF')
    style.decimal_comma(ax, axis='y', places=1)
    ax.legend(loc='upper left')
    figure.tight_layout()

    target = _save(figure, name)
    plt.close(figure)
    return target


def plot_garch_decomposition(crypto, frame=None, name: str | None = None):
    """Zmiennosc calkowita na tle modulu reszt AR."""
    name = f'garch_{crypto.lower()}' if name is None else name

    frame = load_processed(crypto) if frame is None else frame
    col_date = date_column(frame)
    model = frame['model_type'].iloc[0]

    figure, ax = plt.subplots(figsize=style.WIDE)

    ax.plot(frame[col_date], np.abs(frame['residuals']), color=style.SERIES_1,
            alpha=0.35, linewidth=0.5, label='reszty AR (moduł)')
    ax.plot(frame[col_date], frame['total_volatility'], color=style.SERIES_2,
            linewidth=1.4, label='zmienność całkowita')

    ax.set_title(f'{display_name(crypto)}: dekompozycja zmienności, model {model}')
    ax.set_ylabel('odchylenie standardowe')
    style.decimal_comma(ax, axis='y', places=2)
    ax.legend(loc='upper right')
    figure.tight_layout()

    target = _save(figure, name)
    plt.close(figure)
    return target


def _number(value, places: int = 4) -> str:
    """Formatuje liczbe do napisu na rysunku: przecinek, stala liczba miejsc."""
    return f'{value:.{places}f}'.replace('.', ',')
