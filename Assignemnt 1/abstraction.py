import numpy as np
import numpy.lib.stride_tricks as ns
import imageio as imageio
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

    dog_result = np.zeros_like(dog)
    dim1, dim2 = dog.shape[0], dog.shape[1]

    for i in range(dim1):
        for j in range(dim2):
            if dog[i, j] > 0:
                dog_result[i, j] = 1
            else:
                dog_result[i, j] = 1 + np.tanh(phi_e * dog[i, j])

    return dog_result


def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''

    print("Starting luminance_quantization()")

    l_max = 100
    delta_differ = l_max / n_bins

    quantization_steps_tmp = []
    for i in range(n_bins+1):
        quantization_steps_tmp.append(i * delta_differ)

    quantization_steps = np.array(quantization_steps_tmp)

    im_tmp_dim = im[:, :, np.newaxis]
    quantization_estimation = quantization_steps[np.newaxis, np.newaxis, :]

    quantization_min = quantization_steps[np.argmin((np.abs(im_tmp_dim - quantization_estimation)), axis=2)]
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
    windows_neighborhoods_pixels = ns.sliding_window_view(padded, window_shape=win_shape) #shape (400, 254, 1, 17, 17, 3)

    # equation 2)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    norm_inf = np.maximum(np.abs(x), np.abs(y))
    spatial_weights = np.exp(-norm_inf**2 / (2 * sigma_s ** 2)) #shape (17, 17) (∥p − q∥_infitiny norm)

    # center of window = current pixel
    U = np.ones_like(im)

    for x_coord in range(0, 400):
        for y_coord in range(0, 254):
            cur_window = windows_neighborhoods_pixels[x_coord][y_coord][0]

            F_p = cur_window[r, r]
            intensity_diff = np.linalg.norm(F_p - cur_window, axis=2, ord=2)
            intensity_weights = np.exp(-intensity_diff ** 2 / (2 * sigma_r ** 2))  # shape (17, 17) (∥F(p) − F(q)∥_2)

            pixel_weights = spatial_weights * intensity_weights
            pixel_weights = spatial_weights * pixel_weights
            weighted_sum = np.sum(pixel_weights[..., None] * cur_window, axis=(0, 1)) #check this part
            U[x_coord, y_coord] = weighted_sum / np.sum(pixel_weights)
    return U


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)

    filtered_bilateral = skc.lab2rgb(filtered)
    imageio.imsave('bilateral.png', (filtered_bilateral * 255).astype(np.uint8))

    edges = edge_detection(filtered[:, :, 0])

    filtered_edge = edges
    imageio.imsave('edge.png', (filtered_edge * 255).astype(np.uint8))

    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)

    filtered_bilateral_second = skc.lab2rgb(filtered)
    imageio.imsave('bilateral_second.png', (filtered_bilateral_second * 255).astype(np.uint8))

    luminance_quantized = luminance_quantization(filtered[:, :, 0])

    imageio.imsave('luminance_quantized.png', np.uint8(luminance_quantized)) #this is too dark, has to be lighter

    '''Get the final image by merging the channels properly'''
    combined = np.ones_like(filtered)  # Todo
    combined[:, :, 0] = luminance_quantized*edges
    combined[:, :, 1] = filtered[:, :, 1]
    combined[:, :, 2] = filtered[:, :, 2]
    return skc.lab2rgb(combined)


if __name__ == '__main__':
    print(imageio.imread('Assignemnt 1/reference.png'))
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

    im = imageio.imread('Assignemnt 1/girl.png') / 255.
    abstracted = abstraction(im)

    abstracted = (np.clip(abstracted, 0, 1) * 255).astype(np.uint8) # professor put the this part in teach center
    imageio.imsave('abstracted.png', abstracted)

