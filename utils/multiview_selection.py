"""Pick the most consistent multiview sequence out of K candidates per yaw.

The sampler is stochastic (fresh ``x_T`` per row, ``ddim_eta=1.0``), so the same yaw can be
sampled several times and the best-matching candidate picked per yaw.
"""

import numpy as np
import torch
import torch.nn.functional as F

from cldm.arcface_torch_wrapper import preprocess_arcface_input


def generate_candidates(toss, image, yaws, n_candidates=4, base_seed=40, **generate_kwargs):
    """``{yaw: [img_0, ..., img_K-1]}``, one ``toss.generate`` call per candidate."""
    candidates = {}
    for k in range(n_candidates):
        toss.set_seed(base_seed + k)
        images = toss.generate(image=image, dy_list=list(yaws), **generate_kwargs)
        for yaw, image_k in zip(yaws, images):
            candidates.setdefault(yaw, []).append(image_k)
        print(f"candidate {k + 1}/{n_candidates} done (seed={base_seed + k})")
    return candidates


def _tensor(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_raft(device, small=False):
    """RAFT optical flow from torchvision. Downloads weights on first use."""
    if small:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    else:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    return model.to(device).eval()


@torch.no_grad()
def estimate_flow(image1, image2, raft):
    """Flow from ``image1`` to ``image2`` defined on image1's grid.

    ``image1`` / ``image2`` are ``[B, 3, H, W]`` in [0, 1]; RAFT wants [-1, 1] and returns one
    flow per refinement iteration, so only the last is kept. Output is ``[B, 2, H, W]`` in
    pixels, channel 0 = dx (width), channel 1 = dy (height).
    """
    return raft(image1 * 2 - 1, image2 * 2 - 1)[-1]


def warp_with_flow(image, flow):
    """Sample ``image`` at ``p + flow(p)`` for every output pixel ``p``.

    ``grid_sample`` is a backward warp, so the flow has to be defined on the grid of the frame
    you want to land in: with ``flow = estimate_flow(target, image)`` the result is aligned
    with ``target``.

    Returns ``(warped [B, 3, H, W], valid [B, 1, H, W])``. ``valid`` is 0 where the sample
    coordinate left the image, which ``grid_sample`` reads as black and would otherwise look
    like a huge photometric error.
    """
    _, _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    coords = torch.stack((xs, ys)).unsqueeze(0) + flow  # [B, 2, H, W], (x, y) in pixels

    # pixel -> normalized coords, align_corners=False convention (same as cldm/toss_lora.py)
    scale = coords.new_tensor([w, h]).view(1, 2, 1, 1)
    grid = ((2.0 * coords + 1.0) / scale - 1.0).permute(0, 2, 3, 1)  # [B, H, W, 2]

    warped = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    valid = (grid.abs() <= 1.0).all(dim=-1, keepdim=True).permute(0, 3, 1, 2).to(image.dtype)
    return warped, valid


@torch.no_grad()
def warp_cost(left, right, raft):
    """Masked L1 between ``left`` warped into ``right``'s frame and ``right``. Returns ``[B]``."""
    flow = estimate_flow(right, left, raft)  # on right's grid, pointing into left
    warped, valid = warp_with_flow(left, flow)
    err = (warped - right).abs().mean(dim=1, keepdim=True)
    return (err * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp(min=1.0)


@torch.no_grad()
def score_candidates(candidates, source, arcface, raft, device):
    """Reference-free costs.

    ``id_cost[i, k]`` is ``1 - cos(arcface(candidate), arcface(source))``.
    ``flicker[i, a, b]`` is the optical-flow warp error between candidate ``a`` at yaw ``i``
    and candidate ``b`` at yaw ``i + 1``.
    """
    yaws = sorted(candidates)
    frames = torch.stack(
        [torch.stack([_tensor(image) for image in candidates[yaw]]) for yaw in yaws]
    )  # [N, K, 3, H, W], kept on the host
    n, k = frames.shape[:2]

    def embed(batch):
        arc = preprocess_arcface_input(batch.to(device))
        return F.normalize(arcface(arc).float(), dim=-1)

    emb = torch.stack([embed(frames[i]) for i in range(n)])  # [N, K, D]
    src_emb = embed(_tensor(source).unsqueeze(0))[0]  # [D]
    id_cost = (1.0 - emb @ src_emb).cpu().numpy()

    flicker = np.zeros((n - 1, k, k))
    for i in range(n - 1):
        left = frames[i].repeat_interleave(k, dim=0).to(device)  # a a a a b b b b ...
        right = frames[i + 1].repeat(k, 1, 1, 1).to(device)  # a b c d a b c d ...
        # LPIPS version, kept for comparison:
        # dist = lpips_model(left * 2 - 1, right * 2 - 1).flatten(1).mean(1)
        dist = warp_cost(left, right, raft)
        flicker[i] = dist.reshape(k, k).cpu().numpy()

    return yaws, id_cost, flicker


def select_best_path(id_cost, flicker, w_id=1.0, w_flicker=1.0):
    """Viterbi: one candidate per yaw minimizing identity cost plus adjacent warp error."""
    n, k = id_cost.shape
    dp = w_id * id_cost[0].copy()
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        total = dp[:, None] + w_flicker * flicker[i - 1]  # [k_prev, k_cur]
        back[i] = total.argmin(0)
        dp = w_id * id_cost[i] + total.min(0)

    path = [int(dp.argmin())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]
