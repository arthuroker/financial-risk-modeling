from fetch_data import fetch_stock_data
from historical_var import calculate_historical_var
from monte_carlo_var import monte_carlo_var

tickers = ["GOOG"]
prices, returns = fetch_stock_data(tickers)


initial_price = float(prices.iloc[-1].iloc[0]) 

historical_var = calculate_historical_var(returns, confidence_level=95)
monte_carlo_var = monte_carlo_var(initial_price, returns["GOOG"], confidence_level=95, num_simulations=10000)

print(f"Historical VaR (95% Confidence): {historical_var:.4f}")
print(f"Monte Carlo VaR (95% Confidence): {monte_carlo_var:.4f}")