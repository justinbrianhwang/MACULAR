"""VLM backbone adapter tests (proposal 11.2).

These run WITHOUT downloading a model: they exercise the pure geometry helpers
that decide how vision patches map onto region boxes. That mapping is where a
silent, result-destroying bug lived — a non-square page was forced onto a
sqrt(P) square grid, scrambling every region's features and making a perfectly
good backbone look useless (measured: Ministral AP 0.37 before the fix, 0.99
after).
"""

import pytest

torch = pytest.importorskip("torch")

from macular.models.vlm_backbone import VLMBackbone  # noqa: E402


def test_roi_pool_is_spatially_correct():
    """A region over a known patch must pool that patch's value."""
    gh, gw, d = 4, 8, 3
    grid = torch.zeros(gh, gw, d)
    grid[0, 0] = torch.tensor([1.0, 0.0, 0.0])     # top-left
    grid[3, 7] = torch.tensor([0.0, 0.0, 1.0])     # bottom-right
    feats = grid.reshape(gh * gw, d)

    boxes = torch.tensor([
        [0.0, 0.0, 1 / gw, 1 / gh],                # exactly the top-left patch
        [1 - 1 / gw, 1 - 1 / gh, 1.0, 1.0],        # exactly the bottom-right
    ])
    out = VLMBackbone._roi_pool(feats, gh, gw, boxes)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.0, 1.0]))


def test_roi_pool_distinguishes_top_from_bottom_on_nonsquare_grid():
    """Regression: a non-square grid must not be collapsed to a square one.

    With a wrong (square) grid the two halves of the page mix together and every
    region ends up nearly identical — the failure that made a working backbone
    score at chance level.
    """
    gh, gw, d = 10, 4, 2                            # tall, non-square
    grid = torch.zeros(gh, gw, d)
    grid[: gh // 2] = torch.tensor([1.0, 0.0])      # top half
    grid[gh // 2:] = torch.tensor([0.0, 1.0])       # bottom half
    feats = grid.reshape(gh * gw, d)

    boxes = torch.tensor([[0.0, 0.0, 1.0, 0.4],     # top region
                          [0.0, 0.6, 1.0, 1.0]])    # bottom region
    out = VLMBackbone._roi_pool(feats, gh, gw, boxes)
    assert out[0][0] > 0.9 and out[0][1] < 0.1      # top -> first channel
    assert out[1][1] > 0.9 and out[1][0] < 0.1      # bottom -> second channel


def test_roi_pool_falls_back_when_grid_mismatched():
    """If the grid does not match the patch count we must degrade to a mean,
    never index into a wrongly-shaped grid."""
    feats = torch.randn(37, 5)                      # 37 is not gh*gw
    boxes = torch.rand(4, 4).sort(dim=1).values     # any boxes
    out = VLMBackbone._roi_pool(feats, 6, 6, boxes)
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()
