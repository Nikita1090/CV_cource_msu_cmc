from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float64,
    r_max: np.float64,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    scale = np.float64((r_max - r_min) / (q_max - q_min))
    zero = np.int32(np.round((r_max * q_min - r_min * q_max)/(r_max - r_min)))
    return QuantizationParameters(q_min=q_min, q_max=q_max, scale=scale, zero_point=zero)
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return np.int8(np.clip(np.round(r/qp.scale + qp.zero_point), -128, 127))
    # your code goes here /\


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return np.float64((np.int32(q) - qp.zero_point) * qp.scale)
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float64).max
        self.max = np.finfo(np.float64).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        x = x.detach().cpu().numpy()
        self.min = np.float64(min(np.min(x), self.min))
        self.max = np.float64(max(np.max(x), self.max))
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    bound = np.float64(max(np.abs(np.max(weights)), np.abs(np.min(weights))))
    q = compute_quantization_params(r_min=-bound, r_max=bound, q_min=-127, q_max=127)
    return np.int8(np.clip(quantize(weights, q), -127, 127)), q
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/
    L = list()
    L1 = list()
    temp = np.zeros_like(weights, dtype=np.int8)
    for i in range(np.shape(weights)[0]):
        bound = np.float64(max(np.abs(np.max(weights[i,:,:])), np.abs(np.min(weights[i,:,:]))))
        q = compute_quantization_params(r_min=-bound, r_max=bound, q_min=-127, q_max=127)
        L.append(q)
        temp[i,:,:] = np.int8(np.clip(quantize(weights[i,:,:], q), -127, 127))
    return temp, L
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float64,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    return np.int32(np.round(bias/(scale_x * scale_w)))
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    n = 0
    while m < np.float64(0.5) or m >= np.float64(1):
        if m < np.float64(0.5):
            m *= 2
            n += 1
        else:
            m /= 2
            n -= 1
    res = np.int32(np.round(np.multiply(m, np.power(2, 31), dtype=np.float64)))
    return np.int32(n), res
    # your code goes here /\


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    p = np.multiply(accum, m0, dtype=np.int64)
    return np.int32(p >> 31 + n) + np.int32((p >> n + 30) & 1)
    # your code goes here /\
