import yfinance as yf
import pandas as pd
import numpy as np


tickers = [
    # Tech / Comm
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM',
    # Finance
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'AXP', 'USB', 'PNC', 'SCHW',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABT', 'TMO', 'AMGN', 'LLY', 'BMY', 'CVS',
    # Consumer
    'WMT', 'COST', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'HD', 'SBUX', 'TGT',
    # Industrials / Energy / Materials
    'XOM', 'CVX', 'COP', 'HON', 'UNP', 'MMM', 'BA', 'CAT', 'GE', 'LMT'
]


print(f"Downloading data for {len(tickers)} tickers...")
data = yf.download(tickers, start="2005-01-01", end="2026-03-01")['Close']


print("Calculating log returns...")
log_returns = np.log(data / data.shift(1))


log_returns = log_returns.dropna(axis=0, how='any')


output_file = "asset_log_returns.csv"
log_returns.to_csv(output_file)

print(f"Success! Data matrix saved to {output_file}")
print(f"Shape of the dataset: {log_returns.shape[0]} trading days, {log_returns.shape[1]} assets.")