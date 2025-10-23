import numpy as np

params = np.random.uniform(low=50, high=150, size=20)


def clamp(x, lower_bound, upper_bound):
    x[x < lower_bound] = lower_bound
    x[x > upper_bound] = upper_bound
