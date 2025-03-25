import pandas as pd
import numpy as np


def monte_carlo_var(initial_price, returns, confidence_level = 95, num_simulations = 10000, return_all = False):

  mean_return = returns.mean()

  standard_deviation = returns.std()
  
  delta_t = 1

  random_shocks = np.random.normal(0, 1, num_simulations)
  simulated_returns = mean_return - (.5 * (standard_deviation**2)) * delta_t + standard_deviation * random_shocks * np.sqrt(delta_t)

  simulated_prices = initial_price * np.exp(simulated_returns)

  simulated_losses = (simulated_prices - initial_price) / initial_price

  percentile = 100 - confidence_level
  var_mc = np.percentile(simulated_losses, percentile)

  if return_all:
    return var_mc, simulated_losses
  
  return var_mc

