"""System wizualny projektu - tokeny barw, typografia i ustawienia matplotlib.

Jedyne zrodlo barw i proporcji. Funkcje rysujace siegaja po token nazwany
ROLA, jaka kolor pelni w danych, nie po barwe.

Zasada generujaca: CYNOBER to przedmiot uwagi, BLEKIT to odniesienie.
Napisy na rysunku maja diakrytyki, kod i komentarze nie.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# --- Tozsamosc serii i stan --------------------------------------------
BLUE = '#006c98'
VERMILION = '#c64e31'

SERIES_1 = BLUE          # odniesienie, norma, grupa kontrolna
SERIES_2 = VERMILION     # przedmiot uwagi, kategoria badana
RISK = VERMILION         # zdarzenie negatywne
REFERENCE = BLUE         # stan bazowy

# --- Akcent i masa bez tozsamosci --------------------------------------
ACCENT = VERMILION
MUTED = '#7b8f9e'        # nigdy nie trafia do legendy jako nazwana kategoria

# --- Tusz nie-danych ---------------------------------------------------
TEXT = '#22292e'
TEXT_SECONDARY = '#5c6b78'
AXES = '#90a0ad'
GRID = '#dde2e6'
BACKGROUND = '#ffffff'

# --- Kategorie uporzadkowane: kwartyle, przedzialy, etapy --------------
ORDINAL_RAMP = ['#8fb4c7', '#6b9db5', '#3a7694', '#1a5572', '#06324a']

# --- Wiele kategorii rownorzednych - ODSTEPSTWO od limitu dwoch barw ---
# Tylko gdy wszystkie kategorie musza byc w jednym obiekcie graficznym
# (slupek skumulowany, mapa, diagram przeplywu). Zielen i purpura nie moga
# sasiadowac: separacja przy deuteranopii 7,6 wobec progu 8.
CATEGORICAL = ['#009e73', '#e69f00', '#0072b2', '#56b4e9', '#f0e442', '#cc79a7']


# --- Rampy ciagle ------------------------------------------------------
# Sekwencyjna: wielkosc. Dywergencyjna: biegunowosc wokol zera.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'project_sequential',
    ['#ffffff', '#c3d6e0', '#7fa8bd', '#3a7694', '#06324a'])

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'project_diverging',
    ['#006c98', '#b6cad6', '#f2f2f2', '#e8cdc3', '#c64e31'])

# --- Proporcje figur - zamknieta lista ---------------------------------
SMALL = (6.5, 4.0)       # jedna seria, prosty komunikat
STANDARD = (9.0, 5.5)    # domyslny
WIDE = (12.0, 6.0)       # wiele kategorii na osi poziomej, szeregi
SQUARE = (7.5, 7.0)      # macierze

SAVE_DPI = 150


def apply_style() -> None:
    """Ustawia rcParams. Wywolac raz, przed pierwszym rysunkiem."""
    mpl.rcParams.update({
        'figure.figsize': STANDARD,
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'savefig.facecolor': BACKGROUND,
        'savefig.dpi': SAVE_DPI,
        'savefig.bbox': 'tight',

        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titlelocation': 'left',
        'axes.titlepad': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        'text.color': TEXT,
        'axes.titlecolor': TEXT,
        'axes.labelcolor': TEXT_SECONDARY,
        'xtick.color': TEXT_SECONDARY,
        'ytick.color': TEXT_SECONDARY,

        # Ramke nosi tylko os odnoszaca.
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.edgecolor': AXES,
        'xtick.major.size': 0,
        'ytick.major.size': 0,

        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,

        'legend.frameon': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.prop_cycle': mpl.cycler(color=[BLUE, VERMILION]),
    })


def decimal_comma(ax, axis: str = 'y', places: int = 2) -> None:
    """Zamienia kropke dziesietna na przecinek na podanych osiach."""
    formatter = FuncFormatter(
        lambda value, _: f'{value:.{places}f}'.replace('.', ','))
    if 'x' in axis:
        ax.xaxis.set_major_formatter(formatter)
    if 'y' in axis:
        ax.yaxis.set_major_formatter(formatter)
