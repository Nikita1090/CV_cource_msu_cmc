import numpy as np
from scipy.fft import fft2, ifft2, ifftshift


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    limit = (size - 1) / (2*sigma)
    spread = np.linspace(-limit, limit, size)

    e_spread = np.exp(-0.5 * np.float_power(spread, 2))
    kernel = e_spread[None, :] * e_spread[:, None]
    
    return kernel / np.sum(kernel)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    a, b = shape[0] - np.shape(h)[0], shape[1] - np.shape(h)[1]
    pad_width = [((a + 1) // 2, a // 2), ((b + 1) // 2, b // 2)]
    h_padded = np.pad(h, pad_width, mode='constant', constant_values=0)
    kernel_unshifted = ifftshift(h_padded)
    return fft2(kernel_unshifted)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    return np.where(np.abs(H) <= threshold, 0, 1 / H)


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    F_blur = fft2(blurred_img)
    H_i = inverse_kernel(fourier_transform(h, np.shape(F_blur)), threshold=threshold)
    F_l = F_blur * H_i
    f = ifft2(F_l)
    return np.abs(f)


def wiener_filtering(blurred_img, h, K=0.00004):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    F_blur = fft2(blurred_img)
    H = fourier_transform(h, np.shape(F_blur))
    H_s = H.conj()
    F = (H_s / (H_s * H + K)) * F_blur
    return np.abs(ifft2(F))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    return 20 * np.log10(255 / np.sqrt(np.mean((img1 - img2) ** 2)))
