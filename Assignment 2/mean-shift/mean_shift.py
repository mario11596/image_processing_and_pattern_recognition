import matplotlib.pyplot as plt
import imageio
import torch as th

# It is highly recommended to set up pytorch to take advantage of CUDA GPUs!
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
# Choose the size of the image here. Prototyping is easier with 128.
M = 128
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.

# We used this setup for image 128
simul = M ** 2 // 2

# We used this setup for image 256
#simul = M ** 2 // 16
im = th.from_numpy(imageio.imread(f'./{M}.png') / 255.).to(device)

# Visualization
_, ax = plt.subplots(1, 2)
ax[0].imshow(im.cpu().numpy())
artist = ax[1].imshow(im.cpu().numpy())

# For task 2 we used zeta 6 and 0.4 and 0.5
for zeta in [1, 4]:
    y, x = th.meshgrid(
        th.linspace(-zeta, zeta, M, device=device),
        th.linspace(-zeta, zeta, M, device=device),
        indexing='xy'
    )
    features = th.cat((im, y[..., None], x[..., None]), dim=-1).reshape(-1, 5)

    # For task 2 we used h 0.4 and 0.25 and 0.55
    for h in [0.1, 0.3]:
        # The `shifted` array contains the iteration variables
        shifted = features.clone()
        # The `to_do` array contains the indices of the pixels for which the
        # stopping criterion is _not_ yet met.
        to_do = th.arange(M ** 2, device=device)
        while len(to_do):
            # We walk through the points in `shifted` in chunks of `simul`
            # points. Note that for each point, you should compute the distance
            # to _all_ other points, not only the points in the current chunk.
            chunk = shifted[to_do[:simul]].clone()
            # TODO: Mean shift iterations (15), writing back the result into shifted.

            distances_features_chunk = th.cdist(chunk, features, p=2)

            i_x_j_mask = th.where(distances_features_chunk <= h, th.ones_like(distances_features_chunk), th.zeros_like(distances_features_chunk))
            features_weights = i_x_j_mask @ features
            cardinality = i_x_j_mask.sum(dim=1, keepdim=True)
            new_updated_chunk = features_weights / cardinality

            shifted[to_do[:simul]] = new_updated_chunk

            # TODO: Termination criterion (17). cond should be True for samples
            # that need updating. Note that chunk contains the 'old' values.

            cond_check = th.sum((chunk - new_updated_chunk).pow(2), dim=-1)
            cond = cond_check >= 1e-6

            # We only keep the points for which the stopping criterion is not met.
            # `cond` should be a boolean array of length `simul` that indicates
            # which points should be kept.
            to_do = to_do[th.cat((
                cond, cond.new_ones(to_do.shape[0] - cond.shape[0])
            ))]
            artist.set_data(shifted.view(M, M, 5).cpu()[:, :, :3])
            plt.pause(0.1)
        # Reference images were saved using this code.
        output_image = (shifted.view(M, M, 5)[..., :3].clone().cpu().numpy() * 255).clip(0, 255).astype('uint8')
        imageio.imsave(f'./reference/{M}/zeta_{zeta:1.1f}_h_{h:.2f}.png', output_image)