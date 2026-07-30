"""Gradient-path tests for the LoRA (trainable-backbone) route.

The frozen path detaches features to CPU on purpose; the LoRA path must NOT, or
the vision tower silently receives no gradient and "fine-tuning" trains nothing
while still costing full backbone forwards. These tests pin the difference
without loading a real VLM.
"""

import pytest

torch = pytest.importorskip("torch")

from macular.models.vlm_backbone import VLMBackbone  # noqa: E402


def test_roi_pool_preserves_gradient_when_requested():
    gh, gw, d = 4, 6, 3
    feats = torch.randn(gh * gw, d, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])

    out = VLMBackbone._roi_pool(feats, gh, gw, boxes, keep_grad=True)
    assert out.requires_grad, "LoRA path must keep the graph alive"
    out.sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0


def test_roi_pool_detaches_on_frozen_path():
    gh, gw, d = 4, 6, 3
    feats = torch.randn(gh * gw, d, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    out = VLMBackbone._roi_pool(feats, gh, gw, boxes, keep_grad=False)
    assert not out.requires_grad


def test_roi_pool_grad_flows_only_from_selected_region():
    """A box covering the top-left quadrant must not push gradient into
    patches outside it — otherwise ROI pooling is not actually spatial."""
    gh, gw, d = 4, 4, 2
    feats = torch.randn(gh * gw, d, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5]])       # rows 0-1, cols 0-1
    VLMBackbone._roi_pool(feats, gh, gw, boxes, keep_grad=True).sum().backward()
    g = feats.grad.reshape(gh, gw, d).abs().sum(-1)
    assert g[:2, :2].sum() > 0                          # inside the box
    assert g[2:, :].sum() == 0 and g[:, 2:].sum() == 0  # outside untouched


def test_finish_helper_keeps_or_detaches():
    x = torch.randn(4, 3, requires_grad=True)
    assert VLMBackbone._finish(x, keep_grad=True).requires_grad
    kept = VLMBackbone._finish(x, keep_grad=False)
    assert not kept.requires_grad and kept.device.type == "cpu"
