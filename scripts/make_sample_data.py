"""Generate a fictitious brokerage export for the demo (data_sample/raw/data.csv).

Same 23-column format as the Trade Republic transaction export the pipeline
reads. Prices are real (Yahoo Finance, so the portfolio moves like the real
market), everything else is invented: a monthly savings plan into the example
funds from the notebook seeds, a few single-stock and crypto buys, dividends,
interest, salary, rent and everyday card payments at made-up merchants.

Deterministic (fixed random seed) apart from the downloaded prices.
Usage: python scripts/make_sample_data.py
"""

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT_PATH = Path(__file__).resolve().parents[1] / "data_sample" / "raw" / "data.csv"
START = "2020-01-01"
CASH_ACCOUNT_START = "2024-06-01"  # from here the account is also used for spending
SAVINGS_PLAN_EUR = 300.0
RNG = np.random.default_rng(42)

COLUMNS = [
    "datetime", "date", "account_type", "category", "type", "asset_class", "name",
    "symbol", "shares", "price", "amount", "fee", "tax", "currency", "original_amount",
    "original_currency", "fx_rate", "description", "transaction_id", "counterparty_name",
    "counterparty_iban", "payment_reference", "mcc_code",
]

# Symbols/tickers match the example seeds in notebooks 02 and 04.
ASSETS = {
    "VWCE.DE": {"symbol": "IE00EXAMPLE01", "name": "Vanguard FTSE All-World (Acc)", "asset_class": "FUND", "usd": False},
    "IEMA.AS": {"symbol": "IE00EXAMPLE02", "name": "iShares MSCI EM (Acc)", "asset_class": "FUND", "usd": False},
    "4GLD.DE": {"symbol": "IE00EXAMPLE03", "name": "Xetra-Gold", "asset_class": "FUND", "usd": False},
    "AAPL": {"symbol": "US0EXAMPLE04", "name": "Apple", "asset_class": "STOCK", "usd": True},
    "MSFT": {"symbol": "US0EXAMPLE05", "name": "Microsoft", "asset_class": "STOCK", "usd": True},
    "BTC-EUR": {"symbol": "XXEXAMPLE0006", "name": "Bitcoin", "asset_class": "CRYPTO", "usd": False},
}
SAVINGS_PLAN = {"VWCE.DE": 0.70, "IEMA.AS": 0.20, "4GLD.DE": 0.10}
ONE_OFF_BUYS = [  # (date, ticker, EUR amount)
    ("2021-03-15", "AAPL", 800), ("2022-06-20", "MSFT", 600), ("2023-01-10", "AAPL", 400),
    ("2023-11-06", "BTC-EUR", 300), ("2024-08-12", "MSFT", 500), ("2025-04-09", "BTC-EUR", 200),
]
ONE_OFF_SELLS = [("2024-03-18", "AAPL", 0.3)]  # (date, ticker, share of position)
DIVIDEND_USD_PER_SHARE = {"AAPL": 0.24, "MSFT": 0.75}  # per quarter, rough

# (merchant, MCC, visits per month, min EUR, max EUR)
CARD_MERCHANTS = [
    ("CITY SUPERMARKET", 5411, 8, 12, 70),
    ("CORNER BAKERY", 5462, 4, 3, 9),
    ("TRATTORIA ROMA", 5812, 2, 18, 55),
    ("COFFEE CORNER", 5814, 5, 3, 8),
    ("METRO TRANSIT", 4111, 3, 3, 12),
    ("CITY PHARMACY", 5912, 1, 5, 30),
    ("FASHION STORE", 5651, 0.5, 25, 120),
    ("TECH WORLD", 5732, 0.2, 30, 250),
    ("BOOK HOUSE", 5942, 0.4, 10, 35),
    ("FUEL STATION", 5541, 1, 40, 75),
    ("HOTEL SEASIDE", 7011, 0.15, 120, 400),
    ("SKYLINE AIRLINES", 4511, 0.15, 80, 300),
]
CARD_SUBSCRIPTIONS = [("STREAMFLIX", 4899, 12.99), ("MUSICBOX", 5815, 10.99)]
DIRECT_DEBITS = [("Example Immobilien GmbH", 750.0), ("Example Fitness Club", 29.90)]
SALARY_EUR = 2600.0


def _row(ts, **fields):
    row = {c: np.nan for c in COLUMNS}
    row.update(
        datetime=ts.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        date=ts.strftime("%Y-%m-%d"),
        account_type="DEFAULT",
        currency="EUR",
        transaction_id=str(uuid.UUID(int=int(RNG.integers(0, 2**63)) << 64 | int(RNG.integers(0, 2**63)))),
    )
    row.update(fields)
    return row


def load_prices():
    tickers = list(ASSETS) + ["EURUSD=X"]
    raw = yf.download(tickers, start=START, auto_adjust=False, progress=False)["Close"]
    prices = raw.ffill().dropna(how="all")
    for ticker, meta in ASSETS.items():
        if meta["usd"]:
            prices[ticker] = prices[ticker] / prices["EURUSD=X"]  # USD -> EUR
    return prices


def price_on(prices, ticker, date):
    return float(prices[ticker].loc[:pd.Timestamp(date)].dropna().iloc[-1])


def main():
    prices = load_prices()
    end = pd.Timestamp.today().normalize() - pd.offsets.MonthBegin(1)  # start of this month
    months = pd.date_range(START, end, freq="MS")
    rows, shares_held = [], {t: 0.0 for t in ASSETS}

    def trade(date, ticker, eur=None, share_of_position=None, fee=0.0, time="09:05"):
        ts = pd.Timestamp(f"{date} {time}")
        price = round(price_on(prices, ticker, ts), 4)
        meta = ASSETS[ticker]
        if share_of_position is None:
            shares = round(eur / price, 6)
            shares_held[ticker] += shares
            rows.append(_row(ts, category="TRADING", type="BUY", asset_class=meta["asset_class"],
                             name=meta["name"], symbol=meta["symbol"], shares=shares, price=price,
                             amount=round(-shares * price - fee, 2), fee=-fee))
        else:
            shares = round(shares_held[ticker] * share_of_position, 6)
            shares_held[ticker] -= shares
            rows.append(_row(ts, category="TRADING", type="SELL", asset_class=meta["asset_class"],
                             name=meta["name"], symbol=meta["symbol"], shares=shares, price=price,
                             amount=round(shares * price - fee, 2), fee=-fee))

    events = sorted(
        [(d, "buy", t, a) for d, t, a in ONE_OFF_BUYS] + [(d, "sell", t, s) for d, t, s in ONE_OFF_SELLS]
    )

    for month in months:
        cash_account = month >= pd.Timestamp(CASH_ACCOUNT_START)
        plan_day = (month + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

        # money in: deposits for the savings plan, later the salary
        if cash_account:
            rows.append(_row(month + pd.Timedelta(days=26, hours=7),
                             category="CASH", type="TRANSFER_INBOUND",
                             amount=round(SALARY_EUR + RNG.normal(0, 40), 2),
                             description="Incoming transfer from Example Employer GmbH"))
        else:
            rows.append(_row(month + pd.Timedelta(hours=8), category="CASH", type="CUSTOMER_INBOUND",
                             name="Sample User", amount=SAVINGS_PLAN_EUR, description="Deposit"))

        # savings plan
        for ticker, weight in SAVINGS_PLAN.items():
            trade(plan_day, ticker, eur=SAVINGS_PLAN_EUR * weight)

        # one-off trades in this month
        for date, kind, ticker, value in events:
            if pd.Timestamp(date).to_period("M") == month.to_period("M"):
                if kind == "buy":
                    trade(date, ticker, eur=value, fee=1.0, time="14:30")
                else:
                    trade(date, ticker, share_of_position=value, fee=1.0, time="15:10")

        # quarterly dividends on the US stocks
        if month.month in (2, 5, 8, 11):
            for ticker, per_share in DIVIDEND_USD_PER_SHARE.items():
                if shares_held[ticker] > 0:
                    fx = float(prices["EURUSD=X"].loc[:month + pd.Timedelta(days=14)].dropna().iloc[-1])
                    usd = round(shares_held[ticker] * per_share, 2)
                    rows.append(_row(month + pd.Timedelta(days=14, hours=10), category="CASH", type="DIVIDEND",
                                     asset_class="STOCK", name=ASSETS[ticker]["name"],
                                     symbol=ASSETS[ticker]["symbol"], shares=round(shares_held[ticker], 6),
                                     amount=round(usd / fx, 2), original_amount=usd,
                                     original_currency="USD", fx_rate=round(fx, 4)))

        if not cash_account:
            continue

        # interest on the cash balance
        rows.append(_row(month + pd.Timedelta(hours=6), category="CASH", type="INTEREST_PAYMENT",
                         amount=round(RNG.uniform(2, 6), 2), tax=0.0, description="Your interest payment"))

        # rent and gym as SEPA direct debits
        for creditor, eur in DIRECT_DEBITS:
            rows.append(_row(month + pd.Timedelta(days=int(RNG.integers(0, 4)), hours=11),
                             category="CASH", type="TRANSFER_DIRECT_DEBIT_INBOUND", amount=-eur,
                             description=f"SEPA Direct Debit transfer to {creditor} (DE00EXAMPLE0000000000)"))

        # everyday card payments
        month_end = month + pd.offsets.MonthEnd(0)
        for merchant, mcc, visits, lo, hi in CARD_MERCHANTS:
            for _ in range(RNG.poisson(visits)):
                day = month + pd.Timedelta(days=int(RNG.integers(0, month_end.day)),
                                           hours=int(RNG.integers(8, 21)), minutes=int(RNG.integers(0, 60)))
                rows.append(_row(day, category="CASH", type="CARD_TRANSACTION", name=merchant,
                                 description=merchant, amount=-round(RNG.uniform(lo, hi), 2), mcc_code=float(mcc)))
        for merchant, mcc, eur in CARD_SUBSCRIPTIONS:
            rows.append(_row(month + pd.Timedelta(days=4, hours=3), category="CASH", type="CARD_TRANSACTION",
                             name=merchant, description=merchant, amount=-eur, mcc_code=float(mcc)))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[pd.to_datetime(df["date"]) <= pd.Timestamp.today()]  # nothing in the future
    df = df.sort_values("datetime").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} transactions to {OUT_PATH}")
    print(df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
