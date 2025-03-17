import yfinance as yf
import pandas as pd

def fetchStockData(tickers, startDate = "2018-01-01", endDate="2025-3-15"):

  stockData = yf.download(tickers, start = startDate, end = endDate, auto_adjust = False)["Adj Close"]

  returns = stockData.pct_change().dropna()

  return stockData, returns

if __name__ == "__main__":

  tickers = ["GOOG"]

  df, returns = fetchStockData(tickers)

  print(df.head())
  print(returns.head())
