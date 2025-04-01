import numpy as np
import pandas as pd

def create_var_breach_dataset(returns: pd.Series, var_threshold: float, window_size=10):

    returns = returns.dropna().reset_index(drop=True)

    X = []
    y = []

    for i in range(window_size, len(returns) - 1):
        window = returns[i - window_size:i].values
        next_return = returns[i + 1]

        label = int(next_return < var_threshold)
        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)