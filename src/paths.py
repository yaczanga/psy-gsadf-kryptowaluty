"""Katalogi projektu, wyznaczane z polozenia tego pliku.

Moduly znajduja korzen przez __file__; notebook nie moze, wiec idzie w gore od
katalogu roboczego (patrz pierwsza komorka notebooka).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_INTERIM = PROJECT_ROOT / 'data' / 'interim'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

REPORTS = PROJECT_ROOT / 'reports'
FIGURES = REPORTS / 'figures'

_ALL = (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, REPORTS, FIGURES)


def ensure_directories():
    """Tworzy kazdy katalog, do ktorego zapisuje potok."""
    for directory in _ALL:
        directory.mkdir(parents=True, exist_ok=True)
