import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, startDate = "2018-01-01", endDate="2025-3-15"):

  stockData = yf.download(tickers, start = startDate, end = endDate, auto_adjust = False)["Adj Close"]

  returns = stockData.pct_change().dropna()

  return stockData, returns
