from fetch_data import fetch_stock_data
from historical_var import calculate_historical_var
from monte_carlo_var import monte_carlo_var
import matplotlib.pyplot as plt


tickers = ["GOOG"]
prices, returns = fetch_stock_data(tickers)


initial_price = float(prices.iloc[-1].iloc[0]) 

historical_var = calculate_historical_var(returns, confidence_level=95)
monte_carlo_var_value, simulated_losses = monte_carlo_var(initial_price, returns["GOOG"], confidence_level=95, num_simulations=10000, return_all = True)

# Print Values
print(f"Historical VaR (95% Confidence): {historical_var:.4f}")
print(f"Monte Carlo VaR (95% Confidence): {monte_carlo_var_value:.4f}")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(simulated_losses, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

# Mark Monte Carlo VaR and Histroical VaR
plt.axvline(monte_carlo_var_value, color="red", linestyle="--", linewidth=2, label=f'VaR (95%): {monte_carlo_var_value:.4f}')
plt.axvline(historical_var, color="orange", linestyle="--", linewidth=2, label=f'Historical VaR (95%): {historical_var:.4f}')

# Formatting
plt.title("Monte Carlo Simulated Returns with Historical & Monte Carlo VaR (95%)")
plt.xlabel("Simulated Daily Return (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()