import math
import cv2
import numpy as np


def transform(x):
    x[:x.shape[0] // 2] = np.flip(x[:x.shape[0] // 2], 0)
    x[x.shape[0] // 2:] = np.flip(x[x.shape[0] // 2:], 0)
    res = np.zeros((x.shape[0], x.shape[1] * 2)).astype('uint8')
    res[:, :x.shape[1]] = x
    res[:, :x.shape[1]] = np.flip(res[:, :x.shape[1]], 1)
    res[:, x.shape[1]:] = x
    return res


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def fft_to_image(img_fft):
    x = np.abs(img_fft).clip(0, 255).astype('uint8')
    return transform(adjust_gamma(x, 2.5))


def apply_filter_to_fft(x, kernel):
    k = kernel.copy()
    k[:k.shape[0] // 2] = np.flip(k[:k.shape[0] // 2], 0)
    k[k.shape[0] // 2:] = np.flip(k[k.shape[0] // 2:], 0)

    tr_x = x * k[:, -x.shape[1]:]
    return tr_x


def generate_faussian_filter(size=(5, 5), sigma=2):
    kernel = np.fromfunction(
        lambda x, y: \
            (1 / (2 * math.pi * sigma ** 2)) * math.e ** ((-1 * (
                        (x - (size[0] - 1) / 2) ** 2 + (
                            y - (size[1] - 1) / 2) ** 2)) / (2 * sigma ** 2)),
        (size[0], size[1])
    )
    return kernel / np.sum(kernel)


def generate_simple_filter(size=(5, 5), r=5):
    kernel = np.fromfunction(
        lambda x, y: \
            (x - size[0] // 2) ** 2 + (y - size[1] // 2) ** 2 <= r ** 2,
        (size[0], size[1])
    )
    return kernel


def generate_batt(size=(5, 5), d0=5, n=2):
    kernel = np.fromfunction(
        lambda x, y: \
            1 / (1 + (((x - size[0] // 2) ** 2 + (
                        y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
        (size[0], size[1])
    )
    return kernel