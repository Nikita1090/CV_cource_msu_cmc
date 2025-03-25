import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here
    matrix_f = matrix.astype('float64')
    # Отцентруем каждую строчку матрицы
    mean_values = np.mean(matrix_f, axis=1)
    matrix_centered = matrix_f - np.tile(mean_values[:, np.newaxis], np.shape(matrix)[1])
    # Найдем матрицу ковариации
    Cov_matrix = np.cov(matrix_centered)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(Cov_matrix)
    # Посчитаем количество найденных собственных векторов
    vec_count = np.shape(eig_vec)[1]
    # Сортируем собственные значения в порядке убывания
    eig_val_sorted_indx = np.argsort(eig_val)
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eig_vec_sorted = eig_vec[eig_val_sorted_indx][:, ::-1]
    # Оставляем только p собственных векторов
    eig_vec_sorted = eig_vec_sorted[:, :p]
    # Проекция данных на новое пространство
    new_sp = eig_vec_sorted.T @ matrix_centered

    return eig_vec_sorted, new_sp, mean_values


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        eig_vec_sorted, new_sp, mean_values = comp
        res = (eig_vec_sorted @ new_sp) + np.tile(mean_values[:, np.newaxis], np.shape(new_sp)[1])
        result_img.append(np.clip(res, 0, 255).astype('uint8'))
        
    return np.dstack(result_img)


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            processed = pca_compression(img[:, :, j], p)
            compressed.append(processed)
        decompressed = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(decompressed, interpolation='nearest')
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    # Your code here
    offset = np.array([[0], [128], [128]])
    matrix = np.array(
            [[0.299, 0.587, 0.114],
             [-0.1687, -0.3313, 0.5],
             [0.5, -0.4187, -0.0813]])
    img_correct = img.transpose(0, 2, 1).astype('float64')
    res = (matrix @ img_correct + offset).transpose(0, 2, 1).astype('uint8')
    return res


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    offset = np.array([[0], [-128], [-128]])
    matrix = np.array(
            [[1, 0, 1.402],
             [1, -0.34414, -0.71414],
             [1, 1.77, 0]])
    img_corr = img.transpose(0, 2, 1)
    return np.clip((matrix @ (img_corr + offset)).transpose(0, 2, 1), 0, 255).astype('uint8')


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, 1:] = gaussian_filter(ycbcr_img[:, 1:], sigma=10)
    rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(rgb_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, 0] = gaussian_filter(ycbcr_img[:, 0], sigma=10)
    rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(rgb_img)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    # Your code here
    return gaussian_filter(component, 10)[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    alpha = np.array([(1/np.sqrt(2)), 1, 1, 1, 1, 1, 1, 1])
    alpha = alpha[..., None]
    alpha_matrix = (alpha @ alpha.T)
    
    block_shape = np.shape(block)
    res = np.zeros(block_shape).astype('float64')

    for u in range(block_shape[0]):
        for v in range(block_shape[1]):

            cos_u = [np.cos((2 * i + 1) * u * np.pi / 16) for i in range(np.shape(res)[0])]
            cos_v = [np.cos((2 * i + 1) * v * np.pi / 16) for i in range(np.shape(res)[0])]

            for x in range(block_shape[0]):
                for y in range(block_shape[1]):
                    res[u, v] += 0.25 * alpha_matrix[u, v] * block[x, y] * cos_u[x] * cos_v[y]
    return res


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here
    scale_factor = 1
    if q < 50:
        scale_factor = 5000 / q
    elif (q >= 50) and (q <= 99):
        scale_factor = 200 - 2 * q

    recalc_matrix = np.floor((50 + scale_factor * default_quantization_matrix) / 100)
    recalc_matrix[recalc_matrix == 0] = 1
    return recalc_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    # Your code here
    size = np.shape(block)[0]
    flg = True
    res = []
    for i in range(2*size - 1):
        for j in range(i + 1):
            if (i - j < size) and (j < size):
                if flg: # "считываем змейкой"
                    res.append(block[i - j, j]) 
                else:
                    res.append(block[j, i - j])
        flg = flg ^ 1 
    return res

def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here
    count = 1
    zero = False
    res = []
    for elem in zigzag_list:
        if zero:
            if elem == 0:
                count += 1
            else:
                res.append(count)
                res.append(elem)
                zero = False
        else:
            if elem == 0:
                res.append(elem)
                count = 1
                zero = True
            else:
                res.append(elem)
        
    if zero:
        res.append(count)
    return res


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here

    # Переходим из RGB в YCbCr
    img_ycbcr = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    img_y = img_ycbcr[:, 0]
    colored_img = np.dstack((downsampling(img_ycbcr[:, 1]), downsampling(img_ycbcr[:, 2])))
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    res = [[], [], []]
    for y_h in range(0, np.shape(img_y)[0] - 8 + 1, 8): # блок размера 8
        for y_w in range(0, np.shape(img_y)[1] - 8 + 1, 8):
            block = img_y[y_h : y_h + 8, y_w : y_w + 8].astype('int32')
            block -= 128
            q_matrix = quantization_matrixes[0]
            block = compression(zigzag(quantization(dct(block), q_matrix)))
            res[0].append(block)

    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    for y_h in range(0, colored_img.shape[0] - 8 + 1, 8):
        for y_w in range(0, colored_img.shape[1] - 8 + 1, 8):
            for ch in range(2):
                block = colored_img[y_h : y_h + 8, y_w : y_w + 8, ch].astype('int32')
                block -= 128
                q_matrix = quantization_matrixes[1] 
                block = compression(zigzag(quantization(dct(block), q_matrix)))
                res[ch + 1].append(block)
    return res


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    # Your code here
    last_n = 0
    zeros = False
    res = []
    for elem in compressed_list:
        if not(zeros):
            if elem == 0:
                res.append(elem)
                zeros = True
            else:
                res.append(elem)
        else:
            last_n = elem
            for _ in range(elem - 1):
                res.append(0)
            zeros = False

    return res


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    # Your code here
    flg = True
    block = np.zeros((8, 8))
    inp_reversed = input[::-1]
    for i in range(2*8 - 1):
        for j in range(i + 1):
            if (i - j < 8) and (j < 8):
                if flg:
                    block[i - j, j] = inp_reversed.pop()
                else:
                    block[j, i - j] = inp_reversed.pop()
        flg = flg ^ 1 
    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    # Your code here
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    alpha = np.array([(1/np.sqrt(2)), 1, 1, 1, 1, 1, 1, 1])
    alpha = alpha[:, None]
    alpha_matrix = (alpha @ alpha.T)
    
    block_shape = np.shape(block)
    res = np.zeros(block_shape).astype('float64')
    res_shape = np.shape(res)
    for x in range(res_shape[0]):
        for y in range(res_shape[1]):
            for u in range(block_shape[0]):
                for v in range(block_shape[1]):
                    cos_1 = np.cos((2 * x + 1) * u * np.pi / 16)
                    cos_2 = np.cos((2 * y + 1) * v * np.pi / 16)
                    res[x, y] += 0.25 * alpha_matrix[u, v] * block[u, v] * cos_1 * cos_2
    res_rounded = np.round(res)
    return res_rounded


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    # Your code here
    upsampled = np.zeros((np.shape(component)[0] * 2, np.shape(component)[1] * 2), dtype=component.dtype)
    for i in range(np.shape(component)[0]):
        for j in range(np.shape(component)[1]):
            upsampled[2*i, 2*j] = component[i, j]
            upsampled[2*i, 2*j + 1] = component[i, j]
            upsampled[2*i + 1, 2*j] = component[i, j]
            upsampled[2*i + 1, 2*j + 1] = component[i, j]
    return upsampled


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here

    return ...


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        quantization_matrix = [own_quantization_matrix(y_quantization_matrix, p), own_quantization_matrix(color_quantization_matrix, p)]
        compressed = jpeg_compression(img, quantization_matrix)
        decompressed = jpeg_decompression(compressed, np.shape(img), quantization_matrix)
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
