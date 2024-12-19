import numpy as np
import utils

REPORT = True

if REPORT:
    import os


def expectation_maximization(
    X: np.ndarray,
    K: int,
    max_iter: int = 50,
    plot: bool = False,
    show_each: int = 10,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Number of data points (N) and features (m)
    N, m = X.shape

    # Init: Uniform weights, first K points as means, identity covariances
    alphas = np.full((K,), 1.0 / K)
    mus = X[:K]
    sigmas = np.tile(np.eye(m)[None], (K, 1, 1))
    m_times_log2pi = m * np.log(2 * np.pi)

    for it in range(max_iter):
        log_resp = np.zeros((N, K))

        # E-step
        for k in range(K):
            L_k = np.linalg.cholesky(np.linalg.inv(sigmas[k]))
            mahalanobis = np.sum((L_k.T @ (X - mus[k]).T) ** 2, axis=0)

            sign, log_det_L_k = np.linalg.slogdet(L_k)
            log_resp[:, k] = (
                -0.5 * (mahalanobis + m_times_log2pi) + log_det_L_k + np.log(alphas[k])
            )

        log_sum_exp = np.max(log_resp, axis=1, keepdims=True) + np.log(
            np.sum(
                np.exp(log_resp - np.max(log_resp, axis=1, keepdims=True)),
                axis=1,
                keepdims=True,
            )
        )
        gammas = np.exp(log_resp - log_sum_exp)

        sum_of_gammas = np.sum(gammas, axis=0)
        alphas = sum_of_gammas / N

        mus = (gammas.T @ X) / sum_of_gammas[:, None]

        # M-step
        sigmas = np.zeros((K, m, m))
        for k in range(K):
            diff = X - mus[k]
            weighted_diff = gammas[:, k][:, None] * diff
            sigmas[k] = (weighted_diff.T @ diff) / sum_of_gammas[k]
            sigmas[k] += epsilon * np.eye(m)

        if plot and it % show_each == 0:
            print(f"Iteration {it}")
            utils.plot_gmm(X, alphas, mus, sigmas)

    return alphas, mus, sigmas


def denoise(
    index: int = 1,
    K: int = 10,
    w: int = 5,
    alpha: float = 0.5,
    max_iter: int = 30,
    test: bool = False,
    sigma: float = 0.1,
):
    alphas, mus, sigmas = utils.load_gmm(K, w)
    precs = np.linalg.inv(sigmas)
    precs_chol = np.linalg.cholesky(precs)  # "L" in the assignment sheet
    if test:
        # This is really two times `y` since we dont have access to `x` here
        x, y = utils.test_data(index)
    else:
        x, y = utils.validation_data(index, sigma=sigma, seed=1, w=w)
    # x is image-shaped, y is patch-shaped
    # Initialize the estimate with the noisy patches
    x_est = y.copy()
    m = w**2
    lamda = 1 / sigma**2
    E = np.eye(m) - np.full((m, m), 1 / m)

    # TODO: Precompute A, b (26)
    A = np.zeros((K, m, m))
    b = np.zeros((K, m))
    for k in range(K):
        A[k] = np.linalg.inv(lamda * np.eye(m) + E.T @ precs[k] @ E)
        b[k] = precs[k] @ (E @ mus[k])

    # precompute constant term m_times_log2pi and log dets of cholesky decomposition of sigmas
    m_times_log2pi = m * np.log(2 * np.pi)
    log_det_L_ks = [np.linalg.slogdet(L_k) for L_k in precs_chol]

    for it in range(max_iter):
        # TODO: Implement Line 3, Line 4 of Algorithm 1

        # Line 3
        log_resp = np.zeros((x_est.shape[0], K))
        for k in range(K):
            mahalanobis_distance = np.sum(
                (precs_chol[k].T @ ((E @ x_est.T) - mus[k][:, np.newaxis])) ** 2, axis=0
            )
            log_det_L_k = log_det_L_ks[k][1]
            log_resp[:, k] = (
                -0.5 * (mahalanobis_distance + m_times_log2pi)
                + log_det_L_k
                + np.log(alphas[k])
            )

        k_max = np.argmax(log_resp, axis=1)

        # Line 4
        x_tilde = np.einsum("ijk,ik->ij", A[k_max], (lamda * y) + b[k_max])
        x_est = alpha * x_est + (1 - alpha) * x_tilde

        if not test:
            u = utils.patches_to_image(x_est, x.shape, w)
            if it == max_iter - 1:
                print(f"it: {it+1:03d}, psnr(u, y)={utils.psnr(u, x):.2f}")

    return utils.patches_to_image(x_est, x.shape, w)


def benchmark(K: int = 10, w: int = 5):
    for i in range(1, 5):
        utils.imsave(f"./test/img{i}_out.png", denoise(i, K, w, test=True))


def train(use_toy_data: bool = True, K: int = 2, w: int = 5):
    data = np.load("./toy.npy") if use_toy_data else utils.train_data(w)
    # Plot only if we use toy data
    alphas, mus, sigmas = expectation_maximization(data, K=K, plot=use_toy_data)
    # Save only if we dont use toy data
    if not use_toy_data:
        utils.save_gmm(K, w, alphas, mus, sigmas)


if __name__ == "__main__":
    do_training = False
    # Use the toy data to debug your EM implementation
    use_toy_data = False
    # Parameters for the GMM: Components and window size, m = w ** 2
    # Use K = 2 for toy/debug model
    K = 10
    w = 5
    if do_training:
        train(use_toy_data, K, w)
    else:
        for i in range(1, 6):
            denoised_img = denoise(i, K, w, test=False)
            if REPORT:
                out_path = f"./output/output_K_{K}_w_0{w}/"
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                utils.imsave(out_path + f"denoised_img{i}_out.png", denoised_img)

    # If you want to participate in the challenge, you can benchmark your model
    # Remember to upload the images in the submission.
    # benchmark(K, w)
