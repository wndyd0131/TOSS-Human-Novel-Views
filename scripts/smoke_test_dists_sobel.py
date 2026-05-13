"""Smoke test for the DISTS + Sobel-DISTS perceptual loss building blocks.

This intentionally does NOT instantiate the full ``TossLoraModule`` (which would
require Stable Diffusion / TOSS checkpoints, ArcFace weights, a real dataset
and a W&B run). Instead it exercises the two new pieces in isolation:

  1. ``_grayscale_sobel_3ch`` (same body as in ``cldm/toss_lora.py``).
  2. ``DISTS`` from ``DISTS-pytorch`` called exactly the way
     ``TossLoraModule.training_step`` calls it::

         dists_model(pred_rgb, gt_rgb, require_grad=True, batch_average=True)

     plus a second call on Sobel maps, summed and back-propped into ``pred_rgb``.

The helper is intentionally inlined (copy of the one in ``cldm/toss_lora.py``)
so this script does not transitively pull in ``ldm``, ``peft``, ``wandb`` etc.
Update this copy if the helper in ``cldm/toss_lora.py`` ever changes.

Run with:

    python scripts/smoke_test_dists_sobel.py
"""

import sys
import traceback

import torch


def _grayscale_sobel_3ch(rgb_01):
    """rgb_01: [B, 3, H, W] in [0, 1]. Returns 3-channel Sobel magnitude in [0, 1]."""
    from kornia.color import rgb_to_grayscale
    from kornia.filters import sobel
    gray = rgb_to_grayscale(rgb_01)
    edge = sobel(gray)
    return edge.expand(-1, 3, -1, -1).clamp(0.0, 1.0)


def _print(msg):
    print(f"[smoke] {msg}", flush=True)


def test_sobel_helper(device):
    _print("== _grayscale_sobel_3ch ==")
    B, H, W = 2, 64, 64
    rgb = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    edge = _grayscale_sobel_3ch(rgb)

    assert edge.shape == (B, 3, H, W), f"bad shape: {edge.shape}"
    assert edge.min() >= 0.0 and edge.max() <= 1.0, (
        f"out of [0,1]: min={edge.min().item()}, max={edge.max().item()}"
    )
    edge.sum().backward()
    assert rgb.grad is not None, "no grad reached rgb input"
    _print(
        f"  shape={tuple(edge.shape)} dtype={edge.dtype} "
        f"range=[{edge.min().item():.4f},{edge.max().item():.4f}] grad=ok"
    )


def test_dists_forward_backward(device):
    _print("== DISTS forward + backward ==")
    try:
        from DISTS_pytorch import DISTS
    except Exception as e:
        _print(f"  SKIP: DISTS-pytorch not importable ({e})")
        _print("        install with: pip install DISTS-pytorch==0.1")
        return False

    model = DISTS().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    B, H, W = 2, 256, 256
    pred = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    gt = torch.rand(B, 3, H, W, device=device).detach()

    d_img = model(pred, gt, require_grad=True, batch_average=True)
    pred_sobel = _grayscale_sobel_3ch(pred)
    gt_sobel = _grayscale_sobel_3ch(gt).detach()
    d_sobel = model(pred_sobel, gt_sobel, require_grad=True, batch_average=True)

    dists_loss = d_img + d_sobel
    _print(
        f"  d_img={d_img.item():.4f} d_sobel={d_sobel.item():.4f} "
        f"dists_loss={dists_loss.item():.4f}"
    )

    dists_loss.backward()
    assert pred.grad is not None, "no grad reached pred"
    g = pred.grad.detach()
    _print(
        f"  pred.grad: shape={tuple(g.shape)} "
        f"abs_mean={g.abs().mean().item():.3e} "
        f"finite={torch.isfinite(g).all().item()}"
    )

    n_grad_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grad_params == 0, f"DISTS has {n_grad_params} trainable params, expected 0"
    _print(f"  DISTS frozen params verified ({n_grad_params} trainable)")
    return True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"device={device}")

    ok = True
    try:
        test_sobel_helper(device)
    except Exception:
        ok = False
        traceback.print_exc()

    try:
        dists_ok = test_dists_forward_backward(device)
        ok = ok and dists_ok
    except Exception:
        ok = False
        traceback.print_exc()

    _print(f"DONE ok={ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
