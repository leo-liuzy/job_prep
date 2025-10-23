import numpy as np

params = np.random.uniform(low=50, high=150, size=20)


def clamp(x, lower_bound, upper_bound):
    x[x < lower_bound] = lower_bound
    x[x > upper_bound] = upper_bound

    return x


def asymetric_quantization(params, bits):
    # calcualte the scale and offset to shift to zero
    alpha = np.max(params)
    beta = np.min(params)
    value_range = alpha - beta
    scale = value_range / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits - 1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound)
    return quantized, scale, zero
