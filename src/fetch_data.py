import yfinance as yf
import pandas as pd
from datetime import date

today = date.today()
formatted_today = f"{today.year}-{today.month}-{today.day}"

def fetch_stock_data(tickers, startDate = "2000-01-01", endDate= formatted_today):

  stockData = yf.download(tickers, start = startDate, end = endDate, auto_adjust = False)["Adj Close"]

  returns = stockData.pct_change().dropna()

  return stockData, returns
