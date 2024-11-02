import numpy as np
import numpy.lib.stride_tricks as ns

#import imageio -> this causes error
import imageio.v2 as imageio

import skimage.color as skc
import skimage.filters as skf


def edge_detection(im):
    '''Implement DoG smooth edge detection (Eq. 6)'''

    s_e = skf.gaussian(im, sigma_e)
    s_f = skf.gaussian(im, sigma_e * np.sqrt(1.6))

    dog = s_e - tau * s_f

    dog_result = np.where(dog > 0, 1, (1 + np.tanh(phi_e * dog)))

    return dog_result


def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''
    l_max = 100

    delta_differ = l_max / n_bins

    quantization_steps_tmp = []
    for i in range(n_bins + 1):
        quantization_steps_tmp.append(i * delta_differ)

    quantization_steps = np.array(quantization_steps_tmp)

    im_tmp_dim = im[:, :, np.newaxis]
    quantization_steps_im_dim = quantization_steps[np.newaxis, np.newaxis, :]

    quantization_min = quantization_steps[np.argmin((np.abs(im_tmp_dim - quantization_steps_im_dim)), axis=2)]
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

    hight, width, channel = im.shape

    win_size = 2 * r + 1
    windows_neighborhoods_pixels = ns.sliding_window_view(padded, (win_size, win_size, channel))

    y, x = np.mgrid[-r:r, -r:r]
    gaussian_spatial_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))





    return im


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)
    edges = edge_detection(filtered[:, :, 0])

    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)
    luminance_quantized = luminance_quantization(filtered[:, :, 0])

    '''Get the final image by merging the channels properly'''
    combined = filtered  # Todo
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
    imageio.imsave('abstracted.png', np.clip(abstracted, 0, 1))
