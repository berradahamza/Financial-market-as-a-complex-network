"""
Yahoo Finance Stock Price Collector
Project: Financial Market as a Complex Network

USA ticker universe:
- NASDAQ listed securities
- NYSE / AMEX / ARCA securities via NASDAQ Trader

Output:
- yahoo_close_prices_10y_USA.csv
- failed_tickers_USA.csv
- ticker_universe_USA.csv
"""

import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_FILE = "yahoo_close_prices_10y_USA.csv"
FAILED_FILE = "failed_tickers_USA.csv"
TICKER_UNIVERSE_FILE = "ticker_universe_USA.csv"

YEARS_BACK = 10
BATCH_SIZE = 80
SLEEP_BETWEEN_BATCHES = 2
MIN_VALID_RATIO = 0.80

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * YEARS_BACK)


# ============================================================
# 2. LOAD THOUSANDS OF USA TICKERS
# ============================================================

def load_tickers():
    """
    Loads US-listed tickers from NASDAQ Trader.

    Sources:
    - nasdaqlisted.txt: NASDAQ-listed securities
    - otherlisted.txt: NYSE, AMEX, ARCA, etc.

    Filters:
    - removes test issues
    - removes ETFs
    - removes invalid symbols
    - converts Yahoo format: BRK.B -> BRK-B
    """

    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    other_url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

    nasdaq = pd.read_csv(nasdaq_url, sep="|")
    other = pd.read_csv(other_url, sep="|")

    # Remove footer rows
    nasdaq = nasdaq[nasdaq["Symbol"].notna()]
    other = other[other["ACT Symbol"].notna()]

    # Remove test issues
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    other = other[other["Test Issue"] == "N"]

    # Remove ETFs
    nasdaq = nasdaq[nasdaq["ETF"] == "N"]
    other = other[other["ETF"] == "N"]

    # Keep useful metadata
    nasdaq_meta = nasdaq[["Symbol", "Security Name"]].copy()
    nasdaq_meta["Exchange"] = "NASDAQ"

    other_meta = other[["ACT Symbol", "Security Name", "Exchange"]].copy()
    other_meta = other_meta.rename(columns={"ACT Symbol": "Symbol"})

    universe = pd.concat([nasdaq_meta, other_meta], ignore_index=True)

    # Remove special symbols that often fail in Yahoo
    universe = universe[~universe["Symbol"].str.contains(r"\$", regex=True, na=False)]
    universe = universe[~universe["Symbol"].str.contains(r"\^", regex=True, na=False)]
    universe = universe[~universe["Symbol"].str.contains(r"/", regex=True, na=False)]

    # Convert to Yahoo Finance format
    universe["Yahoo Symbol"] = universe["Symbol"].str.replace(".", "-", regex=False)

    universe = universe.drop_duplicates(subset=["Yahoo Symbol"])
    universe = universe.sort_values("Yahoo Symbol")

    universe.to_csv(TICKER_UNIVERSE_FILE, index=False)

    tickers = universe["Yahoo Symbol"].tolist()

    print(f"Loaded ticker universe: {len(tickers)} tickers")
    print(f"Ticker universe saved to: {TICKER_UNIVERSE_FILE}")

    return tickers


# ============================================================
# 3. DOWNLOAD ONE BATCH
# ============================================================

def download_batch(tickers_batch):
    """
    Downloads close prices for one batch of tickers.
    """

    try:
        data = yf.download(
            tickers=tickers_batch,
            start=START_DATE.strftime("%Y-%m-%d"),
            end=END_DATE.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True
        )

        if data.empty:
            return pd.DataFrame(), tickers_batch

        if len(tickers_batch) == 1:
            ticker = tickers_batch[0]
            close_df = data[["Close"]].rename(columns={"Close": ticker})
        else:
            if "Close" not in data.columns.get_level_values(0):
                return pd.DataFrame(), tickers_batch
            close_df = data["Close"]

        close_df = close_df.dropna(axis=1, how="all")

        downloaded_tickers = set(close_df.columns)
        failed = [t for t in tickers_batch if t not in downloaded_tickers]

        return close_df, failed

    except Exception as e:
        print(f"Batch failed: {tickers_batch[:5]}... Error: {e}")
        return pd.DataFrame(), tickers_batch


# ============================================================
# 4. MAIN COLLECTION FUNCTION
# ============================================================

def collect_close_prices(tickers):
    """
    Downloads close prices batch by batch.
    """

    all_dataframes = []
    all_failed = []

    batches = [
        tickers[i:i + BATCH_SIZE]
        for i in range(0, len(tickers), BATCH_SIZE)
    ]

    print(f"\nNumber of tickers: {len(tickers)}")
    print(f"Number of batches: {len(batches)}")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")

    for batch_number, batch in enumerate(batches, start=1):
        print(f"\nDownloading batch {batch_number}/{len(batches)}...")

        close_df, failed = download_batch(batch)

        if not close_df.empty:
            all_dataframes.append(close_df)

        all_failed.extend(failed)

        print(f"Downloaded columns: {close_df.shape[1]}")
        print(f"Failed tickers in batch: {len(failed)}")

        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not all_dataframes:
        raise RuntimeError("No data was downloaded.")

    final_df = pd.concat(all_dataframes, axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    final_df = final_df.sort_index()

    return final_df, all_failed


# ============================================================
# 5. CLEANING
# ============================================================

def clean_price_matrix(df, min_valid_ratio=MIN_VALID_RATIO):
    """
    Keeps only tickers with enough available daily close values.
    """

    min_valid_values = int(len(df) * min_valid_ratio)
    cleaned_df = df.dropna(axis=1, thresh=min_valid_values)

    print("\nCleaning summary:")
    print(f"Initial shape: {df.shape}")
    print(f"After cleaning: {cleaned_df.shape}")
    print(f"Removed tickers: {df.shape[1] - cleaned_df.shape[1]}")

    return cleaned_df


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

def save_outputs(price_df, failed_tickers):
    price_df.to_csv(OUTPUT_FILE)

    failed_df = pd.DataFrame({"failed_ticker": sorted(set(failed_tickers))})
    failed_df.to_csv(FAILED_FILE, index=False)

    print("\nFiles saved:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {FAILED_FILE}")


# ============================================================
# 7. RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    tickers = load_tickers()

    raw_prices, failed_tickers = collect_close_prices(tickers)

    clean_prices = clean_price_matrix(raw_prices)

    save_outputs(clean_prices, failed_tickers)

    print("\nFinal dataset preview:")
    print(clean_prices.head())