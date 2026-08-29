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


@torch.no_grad()
def score_candidates(candidates, source, arcface, lpips_model, device):
    """Reference-free costs.

    ``id_cost[i, k]`` is ``1 - cos(arcface(candidate), arcface(source))``.
    ``flicker[i, a, b]`` is the LPIPS distance between candidate ``a`` at yaw ``i`` and
    candidate ``b`` at yaw ``i + 1``.
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
        dist = lpips_model(left * 2 - 1, right * 2 - 1).flatten(1).mean(1)
        flicker[i] = dist.reshape(k, k).cpu().numpy()

    return yaws, id_cost, flicker


def select_best_path(id_cost, flicker, w_id=1.0, w_flicker=1.0):
    """Viterbi: one candidate per yaw minimizing identity cost plus adjacent LPIPS."""
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
