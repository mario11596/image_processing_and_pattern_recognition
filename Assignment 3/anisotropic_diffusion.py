import sys
import imageio.v3 as imageio
import math_tools
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve


def diffusion_tensor(
    u: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    nabla: sp.csr_matrix,
    mode: str,
):
    # Implement the diffusion tensor (9)
    # Keep in mind which operations require a flattened image, and which don't

    M, N = u.shape

    nabla_x = math_tools.spnabla_x_hp(M, N)
    nabla_y = math_tools.spnabla_y_hp(M, N)

    u_x = nabla_x @ u.flatten()
    u_y = nabla_y @ u.flatten()

    u_x_tilde = gaussian_filter(u_x.reshape(M, N), sigma_u)
    u_x_tilde = u_x_tilde.flatten()
    u_y_tilde = gaussian_filter(u_y.reshape(M, N), sigma_u)
    u_y_tilde = u_y_tilde.flatten()

    u_xx = u_x_tilde * u_x_tilde
    u_xy = u_x_tilde * u_y_tilde
    u_yy = u_y_tilde * u_y_tilde

    S_ij_11 = gaussian_filter(u_xx.reshape(M, N), sigma_g)
    S_ij_12_21 = gaussian_filter(u_xy.reshape(M, N), sigma_g)
    S_ij_22 = gaussian_filter(u_yy.reshape(M, N), sigma_g)

    J = np.stack((S_ij_11, S_ij_12_21,
                  S_ij_12_21, S_ij_22), axis=-1).reshape((M, N, 2, 2))

    mu1 = np.zeros_like(S_ij_11)
    mu2 = np.zeros_like(S_ij_11)
    v1x = np.zeros_like(S_ij_11)
    v1y = np.zeros_like(S_ij_11)
    v2x = np.zeros_like(S_ij_11)
    v2y = np.zeros_like(S_ij_11)

    for i in range(M):
        for j in range(N):
            eigvals, eigvecs = np.linalg.eig(J[i, j])

            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            mu1[i, j] = eigvals[1]
            mu2[i, j] = eigvals[0]

            v1x[i, j], v1y[i, j] = eigvecs[:, 1]
            v2x[i, j], v2y[i, j] = eigvecs[:, 0]

    if mode == 'ced':
        lambda1 = alpha
        lambda2 = alpha + (1 - alpha) * (1 - (np.exp(-(mu1 - mu2) ** 2 / (2 * gamma ** 2))))

    elif mode == 'eed':
        lambda1 = (1 + mu1 / gamma ** 2) ** -0.5
        lambda2 = 1.0

    diffusion_tensor_1 = lambda1 * v1x ** 2 + lambda2 * v2x ** 2
    diffusion_tensor_1 = diffusion_tensor_1.flatten()

    diffusion_tensor_2 = lambda1 * v1y ** 2 + lambda2 * v2y ** 2
    diffusion_tensor_2 = diffusion_tensor_2.flatten()

    diffusion_tensor_3 = lambda1 * v1x * v1y + lambda2 * v2x * v2y
    diffusion_tensor_3 = diffusion_tensor_3.flatten()

    diag_diffusion_tensor_1 = sp.diags(diffusion_tensor_1, format="csc")
    diag_diffusion_tensor_2 = sp.diags(diffusion_tensor_2, format="csc")
    diag_diffusion_tensor_3 = sp.diags(diffusion_tensor_3, format="csc")

    matrix = [[diag_diffusion_tensor_1, diag_diffusion_tensor_3],
              [diag_diffusion_tensor_3, diag_diffusion_tensor_2]]
    matrix = sp.bmat(matrix)

    return matrix


def nonlinear_anisotropic_diffusion(
    image: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    tau: float,
    T: float,
    mode: str,
):
    t = 0.
    U_t = image.ravel()
    nabla = math_tools.spnabla_hp(*image.shape)
    id = sp.eye(U_t.shape[0], format="csc")
    while t < T:
        print(f'{t=}')
        D = diffusion_tensor(
            U_t.reshape(image.shape), sigma_g, sigma_u, alpha, gamma, nabla,
            mode
        )
        U_t = spsolve(id + tau * nabla.T @ D @ nabla, U_t)
        t += tau
    return U_t.reshape(image.shape)


params = {
    'ced': {
        'sigma_g': 1.5,
        'sigma_u': 0.7,
        'alpha': 0.0005,
        'gamma': 1e-4,
        'tau': 5.,
        'T': 100.,
    },
    'eed': {
        'sigma_g': 0.,
        'sigma_u': 10,
        'alpha': 0.,
        # This is delta in the assignment sheet, for the sake of an easy
        # implementation we use the same name as in CED
        'gamma': 1e-4,
        'tau': 1.,
        'T': 10.,
    },
}

inputs = {
    'ced': 'starry_night.png',
    'eed': 'fir.png',
}

if __name__ == "__main__":
    mode = sys.argv[1]
    input = imageio.imread(inputs[mode]) / 255.
    output = nonlinear_anisotropic_diffusion(input, **params[mode], mode=mode)
    imageio.imwrite(
        f'./{mode}_out.png', (output.clip(0., 1.) * 255.).astype(np.uint8)
    )
