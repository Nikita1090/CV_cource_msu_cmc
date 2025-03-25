import numpy as np


def get_bayer_masks(n_rows, n_cols):
    matrixR = np.zeros((n_rows, n_cols), dtype=bool)
    matrixG = np.zeros((n_rows, n_cols), dtype=bool)
    matrixB = np.zeros((n_rows, n_cols), dtype=bool)
    rows, cols = np.indices((n_rows, n_cols))
    matrixR[(rows % 2 == 0) & (cols % 2 == 1)] = True
    matrixB[(rows % 2 == 1) & (cols % 2 == 0)] = True
    matrixG = ~(matrixR | matrixB)
    res = np.dstack((matrixR, matrixG, matrixB))
    return res


def get_colored_img(raw_img):
    x = np.shape(raw_img)[0]
    y = np.shape(raw_img)[1]
    mask = get_bayer_masks(x, y)
    raw_img_r = np.zeros_like(raw_img)
    raw_img_g = np.zeros_like(raw_img)
    raw_img_b = np.zeros_like(raw_img)
    raw_img_r[mask[:, :, 0]] = raw_img[mask[:, :, 0]]
    raw_img_g[mask[:, :, 1]] = raw_img[mask[:, :, 1]]
    raw_img_b[mask[:, :, 2]] = raw_img[mask[:, :, 2]]
    return np.dstack((raw_img_r, raw_img_g, raw_img_b))


def get_raw_img(colored_img):
    raw_img = np.zeros_like(colored_img[:, :, 0])
    x = np.shape(colored_img[:, :, 0])[0]
    y = np.shape(colored_img[:, :, 0])[1]
    mask = get_bayer_masks(x, y)
    raw_img[mask[:, :, 0]] += (colored_img[:, :, 0])[mask[:, :, 0]]
    raw_img[mask[:, :, 1]] += (colored_img[:, :, 1])[mask[:, :, 1]]
    raw_img[mask[:, :, 2]] += (colored_img[:, :, 2])[mask[:, :, 2]]
    return raw_img



def bilinear_interpolation(raw_img):
    rows, cols = np.shape(raw_img)
    ch = 3
    colored_img = get_colored_img(raw_img)
    mask = get_bayer_masks(rows, cols)
    padded_image = np.pad(colored_img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    padded_mask = np.pad(mask, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    result = np.copy(colored_img)
    for c in range(ch):
   
        shifted_images = np.array([padded_image[i:rows + i, j:cols + j, c] for i in range(3) for j in range(3)])
        shifted_masks = np.array([padded_mask[i:rows + i, j:cols + j, c] for i in range(3) for j in range(3)])
   
        summed_values = np.sum(shifted_images * shifted_masks, axis=0)
        known_counts = np.sum(shifted_masks, axis=0)

        known_counts[known_counts == 0] = 1

        mean_values = summed_values // known_counts

        result[:, :, c] = np.where(((colored_img[:, :, c] == 0) & (mask[:, :, c] == 0)), mean_values, colored_img[:, :, c])

    return result


def improved_interpolation(raw_img):
    rows, cols = np.shape(raw_img)
    colored_img = get_colored_img(raw_img)
    padded_image = np.pad(np.copy(raw_img), ((2, 2), (2, 2)), mode='constant', constant_values=0)
    mask_1_2 = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  2,  0,  0],
        [-1,  2,  4,  2, -1],
        [ 0,  0,  2,  0,  0],
        [ 0,  0, -1,  0,  0]
    ])
    mask_3_6 = np.array([
        [ 0,  0, 1/2, 0,  0],
        [ 0, -1,  0, -1,  0],
        [-1,  4,  5,  4, -1],
        [ 0, -1,  0, -1,  0],
        [ 0,  0, 1/2, 0,  0],
    ])
    mask_4_7 = np.array([
        [   0,  0, -1,  0,  0  ],
        [   0, -1,  4, -1,  0  ],
        [ 1/2,  0,  5,  0,  1/2],
        [   0, -1,  4, -1,  0  ],
        [   0,  0, -1,  0,  0  ],
    ])
    mask_5_8 = np.array([
        [   0,  0,-3/2,  0, 0  ],
        [   0,  2,   0,  2, 0  ],
        [-3/2,  0,   6,  0,-3/2],
        [   0,  2,   0,  2, 0  ],
        [   0,  0,-3/2,  0, 0  ],
    ])
    mask = get_bayer_masks(rows, cols)
    Mrows, Mcols = np.indices((rows, cols))
    shifted_images = np.array([padded_image[i:rows + i, j:cols + j] for i in range(5) for j in range(5)])
    deep, shape_a, shape_b = np.shape(shifted_images)
    mask_1_2 = np.repeat(np.expand_dims(np.tile(mask_1_2.reshape(-1, 1), (1, shape_a)), axis=2), repeats=shape_b, axis=2)
    mask_3_6 = np.repeat(np.expand_dims(np.tile(mask_3_6.reshape(-1, 1), (1, shape_a)), axis=2), repeats=shape_b, axis=2)
    mask_4_7 = np.repeat(np.expand_dims(np.tile(mask_4_7.reshape(-1, 1), (1, shape_a)), axis=2), repeats=shape_b, axis=2)
    mask_5_8 = np.repeat(np.expand_dims(np.tile(mask_5_8.reshape(-1, 1), (1, shape_a)), axis=2), repeats=shape_b, axis=2)
    summed_values_R = colored_img[:, :, 0].astype(np.int16)
    summed_values_G = colored_img[:, :, 1].astype(np.int16)
    summed_values_B = colored_img[:, :, 2].astype(np.int16)
    conv_mask_1_2 = np.sum(shifted_images * mask_1_2, axis=0)
    conv_mask_3_6 = np.sum(shifted_images * mask_3_6, axis=0)
    conv_mask_4_7 = np.sum(shifted_images * mask_4_7, axis=0)
    conv_mask_5_8 = np.sum(shifted_images * mask_5_8, axis=0)
    summed_values_R[::2, ::2] = conv_mask_3_6[::2, ::2]
    summed_values_R[1::2, 1::2] = conv_mask_4_7[1::2, 1::2]
    summed_values_R[1::2, ::2] = conv_mask_5_8[1::2, ::2]
    summed_values_G[::2, 1::2] = conv_mask_1_2[::2, 1::2]
    summed_values_G[1::2, ::2] = conv_mask_1_2[1::2, ::2]
    summed_values_B[1::2, 1::2] = conv_mask_3_6[1::2, 1::2]
    summed_values_B[::2, ::2] = conv_mask_4_7[::2, ::2]
    summed_values_B[::2, 1::2] = conv_mask_5_8[::2, 1::2]
    summed_values_R[summed_values_R < 0] = 0
    summed_values_G[summed_values_G < 0] = 0
    summed_values_B[summed_values_B < 0] = 0
    summed_values_R[~mask[:, :, 0]] = np.clip(summed_values_R[~mask[:, :, 0]] // 8, 0, 255)
    summed_values_G[~mask[:, :, 1]] = np.clip(summed_values_G[~mask[:, :, 1]] // 8, 0, 255)
    summed_values_B[~mask[:, :, 2]] = np.clip(summed_values_B[~mask[:, :, 2]] // 8, 0, 255)

    return np.dstack((summed_values_R.astype(np.uint8), summed_values_G.astype(np.uint8), summed_values_B.astype(np.uint8)))


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    mse = np.mean((img_pred - img_gt) ** 2)
    if mse == 0:
        raise ValueError
    m = np.max(img_gt)
    res = 10 * np.log10(m ** 2/mse)
    return res
