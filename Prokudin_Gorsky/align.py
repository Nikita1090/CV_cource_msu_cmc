import numpy as np

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    raw_height = np.shape(raw_img)[0]
    ch_h = raw_height // 3
    image = raw_img
    if (raw_height % 3 == 1):
        image = raw_img[:ch_h * 3, :]
        raw_height = 3 * ch_h
    blue = image[:ch_h,:]
    green = image[ch_h:ch_h * 2,:]
    red = image[ch_h * 2:ch_h * 3]
    width = np.shape(image)[1]
    b_coords = [0, 0]
    g_coords = [ch_h, 0]
    r_coords = [2 * ch_h, 0]
    if crop:
        width = width // 10
        height = ch_h // 10
        blue = blue[height:-height, width:-width]
        green = green[height:-height, width:-width]
        red = red[height:-height, width:-width]
        b_coords = (height, width)
        g_coords = (ch_h + height, width)
        r_coords = (2 * ch_h + height, width)
    unaligned_rgb = (red, green, blue)
    coords = (np.array(r_coords), np.array(g_coords), np.array(b_coords))
    return unaligned_rgb, coords


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def correct_shift(img_a, img_b, v_sh, h_sh):
    return img_a[(-v_sh if v_sh < 0 else 0):(-v_sh if v_sh > 0 else None), (-h_sh if h_sh < 0 else 0):(-h_sh if h_sh > 0 else None)], img_b[(v_sh if v_sh > 0 else 0):(v_sh if v_sh < 0 else None), (h_sh if h_sh > 0 else 0):(h_sh if h_sh < 0 else None)]


def find_relative_shift_pyramid(img_a, img_b):
    max_sh = 15
    pyramid_a = [img_a]
    pyramid_b = [img_b]
    while (pyramid_a[-1].shape[0] > 500 or pyramid_a[-1].shape[1] > 500):
        lvl_a = pyramid_a[-1]
        lvl_b = pyramid_b[-1]
        pyramid_a.append(np.reshape(lvl_a[:lvl_a.shape[0] - lvl_a.shape[0] % 2, :lvl_a.shape[1] - lvl_a.shape[1] % 2], (lvl_a.shape[0] // 2, 2, lvl_a.shape[1] // 2, 2)).swapaxes(1, 2).mean(axis=(2, 3)))
        pyramid_b.append(np.reshape(lvl_b[:lvl_a.shape[0] - lvl_b.shape[0] % 2, :lvl_b.shape[1] - lvl_b.shape[1] % 2], (lvl_b.shape[0] // 2, 2, lvl_b.shape[1] // 2, 2)).swapaxes(1, 2).mean(axis=(2, 3)))
    a_to_b = np.array([0, 0])
    x1, x2, y1, y2 = max_sh, max_sh, max_sh, max_sh
    for level in reversed(range(len(pyramid_a))):
        best_Lshift = np.array([0, 0])
        best_mse_val = float('inf')
        for dx in range(-x1, x2 + 1):
            for dy in range(-y1, y2 + 1):
                val = mse(*correct_shift(pyramid_a[level], pyramid_b[level], dx, dy))
                if val < best_mse_val:
                    best_mse_val = val
                    best_Lshift = np.array([dx, dy])
        x1 = -(2 * best_Lshift[0] - 1)
        x2 = (2 * best_Lshift[0] + 1)
        y1 = -(2 * best_Lshift[1] - 1)
        y2 = (2 * best_Lshift[1] + 1)
        a_to_b = best_Lshift
    return a_to_b

def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    red, green, blue = crops
    coord_r, coord_g, coord_b = crop_coords
    rel_rg = find_relative_shift_fn(red, green)
    rel_bg = find_relative_shift_fn(blue, green)
    r_to_g = np.array(coord_g) - np.array(coord_r) + rel_rg
    b_to_g = np.array(coord_g) - np.array(coord_b) + rel_bg
    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    red, green, blue = channels
    coord_r, coord_g, coord_b = channel_coords
    coord_r = np.array(coord_r) + np.array(r_to_g)
    coord_b = np.array(coord_b) + np.array(b_to_g)

    top = max(coord_r[0], coord_g[0], coord_b[0])
    left = max(coord_r[1], coord_g[1], coord_b[1])
    bottom = min(coord_r[0] + red.shape[0], coord_g[0] + green.shape[0], coord_b[0] + blue.shape[0])
    right = min(coord_r[1] + red.shape[1], coord_g[1] + green.shape[1], coord_b[1] + blue.shape[1])

    top_r, left_r = top - coord_r[0], left - coord_r[1]
    top_g, left_g = top - coord_g[0], left - coord_g[1]
    top_b, left_b = top - coord_b[0], left - coord_b[1]

    red_cropped = red[top_r:top_r + (bottom - top), left_r:left_r + (right - left)]
    green_cropped = green[top_g:top_g + (bottom - top), left_g:left_g + (right - left)]
    blue_cropped = blue[top_b:top_b + (bottom - top), left_b:left_b + (right - left)]

    aligned_img = np.stack((red_cropped, green_cropped, blue_cropped), axis=-1)
    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    f_a = np.fft.ifft2(img_a)
    f_b = np.fft.ifft2(img_b)
    cross_corr = np.fft.ifft2(f_a * np.conj(f_b))
    max_idx = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
    a_to_b = np.array(max_idx)
    if a_to_b[0] > img_a.shape[0] // 2:
        a_to_b[0] -= img_a.shape[0]
    if a_to_b[1] > img_a.shape[1] // 2:
        a_to_b[1] -= img_a.shape[1]
    return a_to_b


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
