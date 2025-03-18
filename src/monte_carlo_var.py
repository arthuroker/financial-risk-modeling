import pandas as pd
import numpy as np


def monte_carlo_var(initial_price, returns, confidence_level = 95, num_simulations = 10000):

  mean_return = returns.mean()

  standard_deviation = returns.std()

  print(f"Mean return: {mean_return:.6f}")
  print(f"Standard deviation: {standard_deviation:.6f}")
  print(f"Initial price: {initial_price:.2f}")

  delta_t = 1

  random_shocks = np.random.normal(0, 1, num_simulations)
  simulated_returns = mean_return - (.5 * (standard_deviation**2)) * delta_t + standard_deviation * random_shocks * np.sqrt(delta_t)

  simulated_prices = initial_price * np.exp(simulated_returns)

  print(f"Simulated prices (first 5): {simulated_prices[:5]}")

  simulated_losses = (simulated_prices - initial_price) / initial_price

  percentile = 100 - confidence_level
  var_mc = np.percentile(simulated_losses, percentile)

  return var_mc

