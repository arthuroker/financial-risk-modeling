import numpy as np
import pandas as pd

def calculate_historical_var(returns, confidence_level = 95):

  percentile = 100 - confidence_level

  var_values = np.percentile(returns.dropna(), percentile)

  return var_values



