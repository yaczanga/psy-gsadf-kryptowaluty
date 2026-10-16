"""
System wizualny projektu - tokeny barw, typografia i ustawienia matplotlib.

Modul jest jedynym zrodlem barw i proporcji dla wszystkich wykresow. Funkcje
rysujace nie dobieraja koloru samodzielnie - siegaja po token opisujacy ROLE,
jaka kolor pelni w danych (tozsamosc serii, wielkosc, biegunowosc, stan).

Zasada generujaca systemu:

    CYNOBER to przedmiot uwagi, BLEKIT to odniesienie.

Ryzyko, zdarzenie negatywne, element wyrozniony i kategoria bedaca przedmiotem
badania sa "przedmiotem uwagi". Grupa kontrolna, norma i tlo sa "odniesieniem".
Przydzial barwy do serii jest wiec wnioskiem z pytania badawczego, nie decyzja
estetyczna.

Palety sprawdzono walidatorem separacji barw: para blekit-cynober osiaga
delta E 15,6 przy protanopii i 27,4 przy widzeniu pelnobarwnym (progi 8 i 15)
oraz ponad 3:1 kontrastu wobec bieli. Informacja siedzi takze w jasnosci, wiec
rysunki pozostaja czytelne w druku czarno-bialym.

Tekst widoczny NA rysunku zapisujemy pelna polszczyzna z diakrytykami (domyslny
font DejaVu Sans je obsluguje). Komentarze, docstringi i nazwy w kodzie
pozostaja bez ogonkow, zgodnie z reszta projektu.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# Rola: tozsamosc serii
# --------------------------------------------------------------------------
# System niesie tozsamosc DWIEMA barwami. Trzecia seria nie jest kolejnym
# odcieniem - patrz series_colors() nizej.
BLUE = '#006c98'
VERMILION = '#c64e31'

SERIES_1 = BLUE          # odniesienie, norma, grupa kontrolna
SERIES_2 = VERMILION     # przedmiot uwagi, kategoria badana

# --------------------------------------------------------------------------
# Rola: stan (wartosciowanie)
# --------------------------------------------------------------------------
RISK = VERMILION         # upadlosc, blad, zdarzenie negatywne
REFERENCE = BLUE         # firma zdrowa, trafienie, stan bazowy

# --------------------------------------------------------------------------
# Rola: akcent (jeden wyrozniony element na tle masy)
# --------------------------------------------------------------------------
ACCENT = VERMILION
MUTED = '#7b8f9e'        # masa elementow BEZ tozsamosci; nigdy nie trafia
                         # do legendy jako nazwana kategoria

# --------------------------------------------------------------------------
# Rola: neutralna (tusz nie-danych)
# --------------------------------------------------------------------------
TEXT = '#22292e'            # tytul, etykiety osi             14,74:1
TEXT_SECONDARY = '#5c6b78'  # etykiety wartosci, znaczniki     5,48:1
AXES = '#90a0ad'
GRID = '#dde2e6'
BACKGROUND = '#ffffff'

# --------------------------------------------------------------------------
# Rola: porzadkowa (kategorie uporzadkowane: kwartyle, przedzialy, etapy)
# --------------------------------------------------------------------------
# Jeden odcien, monotoniczna jasnosc. Jasny koniec ma kontrast 2,21:1 wobec
# bieli, wiec pierwszy stopien nie znika na tle.
ORDINAL_RAMP = ['#8fb4c7', '#6b9db5', '#3a7694', '#1a5572', '#06324a']

# --------------------------------------------------------------------------
# Rola: sekwencyjna (wielkosc: macierz pomylek, waznosc cech)
# --------------------------------------------------------------------------
# Jasny koniec siega bieli celowo: wartosc zerowa ma czytac sie jako puste tlo,
# a nie jako najnizszy stopien skali. To rampa ciagla, nie porzadkowa.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'project_sequential',
    ['#ffffff', '#c3d6e0', '#7fa8bd', '#3a7694', '#06324a'])

# --------------------------------------------------------------------------
# Rola: dywergencyjna (biegunowosc wokol zera: korelacje, wartosci SHAP)
# --------------------------------------------------------------------------
# Blekit = wartosci ujemne, cynober = dodatnie, szary neutralny w zerze.
# Dla SHAP zgadza sie to z zasada generujaca: wplyw podnoszacy ryzyko jest
# przedmiotem uwagi, wiec dostaje cynober.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'project_diverging',
    ['#006c98', '#b6cad6', '#f2f2f2', '#e8cdc3', '#c64e31'])

# --------------------------------------------------------------------------
# Proporcje figur - zamknieta lista
# --------------------------------------------------------------------------
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

        # Ramke nosi tylko os odnoszaca - gorna i prawa nie niosa informacji.
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


def apply_shap_style() -> None:
    """Nadpisuje palete biblioteki shap barwami systemu.

    Biblioteki rysujace wlasne figury (shap, optuna.visualization, yellowbrick)
    nie wiedza nic o systemie projektu i uzywaja swoich domyslnych barw. Nie
    zostawiamy ich w tym stanie - przykrywamy palete przed pierwszym
    wywolaniem, bo rysunek z obca kolorystyka psuje spojnosc calego zestawu
    mocniej niz jakakolwiek pojedyncza decyzja wewnatrz wykresu.

    UWAGA na sposob podmiany. Podstawienie pod shap.plots.colors.* nie dziala
    dla czesci wykresow: summary_plot ma cmap=colors.red_blue jako WARTOSC
    DOMYSLNA argumentu, wiazana przy imporcie modulu, wiec pozniejsza zmiana
    atrybutu nie ma skutku - mape trzeba podac jawnie przy wywolaniu (robia to
    funkcje plot_shap_* w module visualisation). Wykresy oparte o nowy
    mechanizm stylu shap (waterfall) ustawia sie przez set_style, ktory zyje
    w shap.plots._style i nie jest wystawiony w shap.plots. Slownik labels jest
    zwyklym slownikiem wspoldzielonym przez moduly, wiec modyfikacja w miejscu
    dziala i tlumaczy opisy osi.
    """
    from matplotlib.colors import to_rgb
    import shap
    from shap.plots._labels import labels

    shap.plots.colors.red_blue = DIVERGING_CMAP
    shap.plots.colors.red_rgb = np.array(to_rgb(VERMILION))
    shap.plots.colors.blue_rgb = np.array(to_rgb(BLUE))
    shap.plots.colors.light_red_rgb = np.array(to_rgb('#e8cdc3'))
    shap.plots.colors.light_blue_rgb = np.array(to_rgb('#b6cad6'))

    set_style = (getattr(shap.plots, 'set_style', None)
                 or getattr(shap.plots._style, 'set_style', None))
    if set_style is not None:
        set_style(
            primary_color_positive=VERMILION,
            primary_color_negative=BLUE,
            secondary_color_positive='#e8cdc3',
            secondary_color_negative='#b6cad6',
            hlines_color=GRID,
            vlines_color=AXES,
            tick_labels_color=TEXT_SECONDARY,
        )

    labels.update({
        'VALUE': 'Wartość SHAP (wpływ na wynik modelu)',
        'GLOBAL_VALUE': 'Średni bezwzględny wpływ na wynik modelu',
        'FEATURE_VALUE': 'Wartość zmiennej',
        'FEATURE_VALUE_LOW': 'niska',
        'FEATURE_VALUE_HIGH': 'wysoka',
        'MODEL_OUTPUT': 'Wynik modelu',
        'FEATURE': 'Zmienna %s',
    })


def series_colors(n: int) -> list[str]:
    """Barwy dla n serii - lub blad, gdy n przekracza pojemnosc systemu.

    System niesie tozsamosc dwiema barwami. Trzeciej i dalszych NIE tworzymy
    przez dobranie kolejnego odcienia, bo to rozmontowuje rozpoznawalnosc
    i psuje separacje przy daltonizmie. Zamiast tego:

    - kategorie UPORZADKOWANE (kwartyle, przedzialy, etapy) -> ORDINAL_RAMP
    - jedna seria wyrozniona na tle reszty -> ACCENT na tle MUTED
    - kategorie rownorzedne, ktorych jest wiecej niz dwie -> osobne panele
      (small multiples) zamiast jednego wykresu z wieloma barwami
    """
    if n <= 0:
        raise ValueError('Liczba serii musi byc dodatnia.')
    if n == 1:
        return [BLUE]
    if n == 2:
        return [BLUE, VERMILION]
    raise ValueError(
        f'System niesie tozsamosc dwiema barwami, zadano {n}. Uzyj '
        f'ORDINAL_RAMP (kategorie uporzadkowane), wzorca ACCENT na tle MUTED '
        f'(jedna seria wyrozniona) albo rozbij na osobne panele.')


def decimal_comma(ax, axis: str = 'y', places: int = 2) -> None:
    """Zamienia kropke dziesietna na przecinek na podanych osiach.

    Etykiety sa po polsku, wiec separatorem dziesietnym jest przecinek.
    """
    formatter = FuncFormatter(
        lambda value, _: f'{value:.{places}f}'.replace('.', ','))
    if 'x' in axis:
        ax.xaxis.set_major_formatter(formatter)
    if 'y' in axis:
        ax.yaxis.set_major_formatter(formatter)


def value_labels(ax, places: int = 3, suffix: str = '', **kwargs) -> None:
    """Podpisuje slupki wartoscia, z przecinkiem dziesietnym.

    Etykieta nosi barwe tekstu, nie barwe serii - liczba ma uzupelniac dane,
    nie konkurowac z nimi o uwage. Separator dziesietny musi byc ten sam co na
    osiach, inaczej jeden rysunek miesza dwie konwencje zapisu liczby.

    Etykiety stosujemy wybiorczo - tam, gdzie liczba jest przedmiotem odczytu.
    Podpisanie kazdego punktu zamienia wykres w tabele i zasloni dane, gdy
    slupki sasiednich serii sa waskie.
    """
    for container in ax.containers:
        texts = [f'{value:.{places}f}'.replace('.', ',') + suffix
                 for value in container.datavalues]
        ax.bar_label(container, labels=texts, fontsize=9, padding=2,
                     color=TEXT_SECONDARY, **kwargs)


def upper_triangle_mask(matrix):
    """Zwraca maske ukrywajaca przekatna i gorny trojkat macierzy symetrycznej.

    Na macierzy korelacji przekatna zawsze wynosi 1,00, a oba trojkaty niosa te
    sama informacje. Bez maski najmocniejsza barwa systemu trafia na komorki
    bez tresci, co jest wprost sprzeczne z zasada "cynober to przedmiot uwagi".
    """
    return np.triu(np.ones_like(np.asarray(matrix), dtype=bool))
