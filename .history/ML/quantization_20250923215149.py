import numpy as np

params = np.random.uniform(low=50, high=150, size=20)


def clamp(x, lower_bound, upper_bound):
    x[x < lower_bound] = lower_bound
    x[x > upper_bound] = upper_bound

    return x


def asymmetric_quantization(params, bits):
    # calcualte the scale and offset to shift to zero
    alpha = np.max(params)
    beta = np.min(params)
    value_range = alpha - beta
    scale = value_range / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits - 1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero


def asymmetric_dequantization(params, scale, zero):
    return (params - zero) * scale


def symmetric_quantization(params, bits):
    # calcualte the scale and offset to shift to zero
    alpha = np.max(np.abs(params))
    value_range = alpha - beta
    # note `(bits-1)` instead of `bits`, we sacrifice one more bit for symmetry
    scale = alpha / (2 ** (bits - 1) - 1)
    lower_bound, upper_bound = -(2 ** (bits - 1) - 1), (2 ** (bits - 1) - 1)
    quantized = clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale


def symmetric_dequantization(params, scale):
    return params * scale


def quantization_error(params, params_q):
    return np.mean((params - params_q) ** 2)


# Range selection
def asymmetric_quantization_percentile(params, bits, percentile=99.99):
    # calcualte the scale and offset to shift to zero
    alpha = np.percentile(params, precentile)
    beta = np.percentile(params, 100 - precentile)
    value_range = alpha - beta
    scale = value_range / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits - 1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero


# Post-training quantization
class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedVerySimpleNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.linear1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.quant(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)
        return x
