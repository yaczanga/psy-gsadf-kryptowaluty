"""Pobranie dziennych cen zamkniecia i wczytywanie plikow potoku.

Yahoo Finance odpytuje wylacznie download_prices; kazdy pozniejszy etap czyta
pliki CSV zapisane przez ten modul, wiec analiza dziala bez dostepu do sieci.
"""
import re

import pandas as pd

from src import paths

TICKERS = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'BinanceCoin': 'BNB-USD',
    'XRP': 'XRP-USD',
    'Cardano': 'ADA-USD',
    'Litecoin': 'LTC-USD',
    'Dogecoin': 'DOGE-USD',
    'BitcoinCash': 'BCH-USD',
    'NEM': 'XEM-USD',
    'Dash': 'DASH-USD',
    'Stellar': 'XLM-USD',
    'Monero': 'XMR-USD',
}

CRYPTO_LIST = list(TICKERS)

# Granice proby. Yahoo Finance podaje 10 z 12 szeregow dopiero od 2017-11-09,
# wiec 2018 to pierwszy pelny rok z kompletem notowan. Gorna granica jest
# w yfinance wylaczna, wiec ostatnia obserwacja to 2025-12-30.
START_DATE = '2018-01-01'
END_DATE = '2025-12-31'


def display_name(name):
    """Rozdziela klucz CamelCase na potrzeby napisu na rysunku.

    BinanceCoin -> Binance Coin, zgodnie z zapisem w README.
    """
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)


def date_column(frame):
    """Zwraca nazwe kolumny z data, niezaleznie od wielkosci liter."""
    for column in frame.columns:
        if 'date' in column.lower():
            return column
    raise KeyError(f'Brak kolumny z data wsrod {list(frame.columns)}.')


def download_prices(tickers=None, start=START_DATE, end=END_DATE, out_dir=None):
    """Pobiera ceny zamkniecia i zapisuje po jednym pliku CSV na szereg.

    Wymaga dostepu do sieci i pakietu yfinance; reszta potoku nie wymaga ani
    jednego, ani drugiego. Zwraca liste faktycznie zapisanych sciezek.
    """
    import yfinance as yf

    tickers = TICKERS if tickers is None else tickers
    out_dir = paths.DATA_RAW if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, symbol in tickers.items():
        frame = yf.download(symbol, start=start, end=end, progress=False)
        if frame.empty:
            print(f'{name} ({symbol}): brak danych.')
            continue

        # yfinance zwraca MultiIndex przy wielu symbolach, a w czesci wersji
        # zostawia go takze przy pojedynczym.
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        frame = frame.reset_index()[['Date', 'Close']]
        target = out_dir / f'{name}.csv'
        frame.to_csv(target, index=False)
        written.append(target)
        print(f'{name} ({symbol}): {len(frame)} obserwacji -> {target.name}')

    return written


def _read(path):
    frame = pd.read_csv(path)
    column = date_column(frame)
    frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values(column).reset_index(drop=True)


def load_raw(name, folder=None):
    """Wczytuje szereg surowy: kolumny z data i cena zamkniecia."""
    folder = paths.DATA_RAW if folder is None else folder
    return _read(folder / f'{name}.csv')


def load_interim(name, folder=None):
    """Wczytuje szereg po przygotowaniu: reszty, g(t/T), reszty standaryzowane."""
    folder = paths.DATA_INTERIM if folder is None else folder
    return _read(folder / f'{name}.csv')


def load_processed(name, folder=None):
    """Wczytuje szereg po modelowaniu zmiennosci: zmiennosc calkowita i model."""
    folder = paths.DATA_PROCESSED if folder is None else folder
    return _read(folder / f'{name}.csv')
