import numpy as np
import numpy.lib.stride_tricks as ns
import imageio
import skimage.color as skc
import skimage.filters as skf


def edge_detection(im):
    '''Implement DoG smooth edge detection (Eq. 6)'''

    print("Starting edge_detection()")

    s_e = skf.gaussian(image=im, sigma=sigma_e)

    s_f_corr = sigma_e * np.sqrt(1.6)
    s_f = skf.gaussian(image=im, sigma=s_f_corr)

    s_f_tau = tau * s_f
    dog = s_e - s_f_tau

    dog_edge_detection = np.zeros_like(dog)
    dim1, dim2 = dog.shape[0], dog.shape[1]

    for i in range(dim1):
        for j in range(dim2):
            if dog[i, j] > 0:
                dog_edge_detection[i, j] = 1
            else:
                dog_edge_detection[i, j] = 1 + np.tanh(phi_e * dog[i, j])

    return dog_edge_detection


def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''

    print("Starting luminance_quantization()")

    l_max = 100
    delta_differ = l_max / n_bins

    quantization_steps_tmp = []
    for i in range(n_bins):
        quantization_steps_tmp.append(i * delta_differ)

    quantization_steps = np.array(quantization_steps_tmp)
    im_tmp_dim = im[:, :, np.newaxis]
    quantization_estimation = quantization_steps[np.newaxis, np.newaxis, :]

    quantization_min = quantization_steps[np.argmin(np.abs(im_tmp_dim - quantization_estimation), axis=2)]
    quantized_luminance_image = quantization_min + (delta_differ / 2) * np.tanh(phi_q * (im - quantization_min))

    return quantized_luminance_image


def bilateral_gaussian(im):
    # Radius of the Gaussian filter
    r = int(2 * sigma_s) + 1
    padded = np.pad(im, ((r, r), (r, r), (0, 0)), 'edge')
    '''
    Implement the bilateral Gaussian filter (Eq. 3).
    Apply it to the padded image.
    '''
    print("Starting bilateral_gaussian()")

    win_size = 2 * r + 1
    win_shape = (win_size, win_size, 3)
    windows_neighborhoods_pixels = ns.sliding_window_view(padded, window_shape=win_shape)

    y, x = np.mgrid[-r:r+1, -r:r+1]
    norm = x ** 2 + y ** 2
    spatial_weights_gauss = np.exp(-norm / (2 * sigma_s ** 2))

    M, N, labdim = im.shape
    U = np.ones_like(im)

    for x_coord in range(0, M):
        for y_coord in range(0, N):
            cur_window = windows_neighborhoods_pixels[x_coord][y_coord][0]

            F_p = cur_window[r, r]
            intensity_diff = np.linalg.norm(cur_window - F_p, axis=2, ord=2)
            intensity_weights_gauss = np.exp(-intensity_diff ** 2 / (2 * sigma_r ** 2))

            pixel_weights_all = spatial_weights_gauss * intensity_weights_gauss
            weighted_sum = np.sum(pixel_weights_all[..., None] * cur_window, axis=(0, 1))
            U[x_coord, y_coord] = weighted_sum / np.sum(pixel_weights_all)
    return U


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)
    edges = edge_detection(filtered[:, :, 0])

    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)
    luminance_quantized = luminance_quantization(filtered[:, :, 0])

    '''Get the final image by merging the channels properly'''
    combined = np.ones_like(filtered)
    combined[:, :, 0] = luminance_quantized * edges
    combined[:, :, 1] = filtered[:, :, 1]
    combined[:, :, 2] = filtered[:, :, 2]
    return skc.lab2rgb(combined)


if __name__ == '__main__':
    # Algorithm
    n_e = 2
    n_b = 4
    # Bilateral Filter
    sigma_r = 4.25  # "Range" sigma
    sigma_s = 3.5  # "Spatial" sigma
    # Edge Detection
    sigma_e = 1
    tau = 0.98
    phi_e = 5
    # Luminance Quantization
    n_bins = 10
    phi_q = 0.7

    im = imageio.imread('./girl.png') / 255.
    abstracted = abstraction(im)
    abstracted = (abstracted * 255).astype(np.uint8)
    imageio.imsave('abstracted.png', abstracted)