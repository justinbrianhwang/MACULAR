"""LEACE — closed-form linear concept erasure (Belrose et al., NeurIPS 2023).

Replaces the differentiable redaction gate, which our own ablation showed buys
nothing over a hard mask. LEACE gives a *provable* property instead of a hoped-for
one: after the transform, the cross-covariance between features and the concept
is exactly zero, so **no linear classifier** can predict the concept better than
a constant predictor.

The eraser is a single affine map, so it is cheap (one covariance estimate plus
an eigendecomposition — no training loop) and differentiable, and it drops into
the forward pass as a frozen layer.

Definition (paper notation). With X the features, Z the one-hot concept labels,
mu = E[X], Sigma_XX = Cov(X), Sigma_XZ = Cov(X, Z):

    W    = Sigma_XX^(-1/2)                   (whitening)
    W+   = Sigma_XX^(+1/2)                   (its pseudo-inverse)
    P    = orthogonal projector onto col(W Sigma_XZ)
    r(x) = x - W+ P W (x - mu)

We keep it as (A, mu) with A = I - W+ P W, i.e. r(x) = A x + (mu - A mu).

WHAT THIS DOES NOT GIVE YOU. The guarantee covers linear readouts only. Nonlinear
probes (RBF-SVM, MLP) recover a large fraction of an "erased" concept in
published stress tests. Always report ``macular.privacy.probes.probe_leakage``,
which runs both, next to any LEACE claim.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _inv_sqrt_psd(mat: torch.Tensor, eps: float = 1e-6):
    """Return (M^-1/2, M^+1/2) for a symmetric PSD matrix.

    Uses an eigendecomposition and floors the eigenvalues at ``eps``: region
    feature covariances are routinely rank-deficient (fewer regions than dims),
    and a raw inverse square root would blow up on the null directions.
    """
    mat = 0.5 * (mat + mat.transpose(-1, -2))          # force symmetry
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(eps)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.transpose(-1, -2)
    sqrt = evecs @ torch.diag(evals.sqrt()) @ evecs.transpose(-1, -2)
    return inv_sqrt, sqrt


def fit_leace(x: torch.Tensor, z: torch.Tensor, eps: float = 1e-6):
    """Fit the eraser. ``x`` is (N, d) features, ``z`` is (N,) integer concept
    labels (e.g. PII type, 0 = non-PII) or an (N, k) indicator matrix.

    Returns a ``LeaceEraser``. Fit on TRAIN features only, then apply to val/test
    — fitting on the evaluation set would leak the labels being erased.
    """
    x = x.detach().to(torch.float64)
    if z.dim() == 1:
        k = int(z.max().item()) + 1
        z = torch.nn.functional.one_hot(z.long(), num_classes=k)
    z = z.detach().to(torch.float64)

    n = x.shape[0]
    if n < 2:
        raise ValueError(f"LEACE needs at least 2 samples, got {n}")
    mu = x.mean(0)
    xc, zc = x - mu, z - z.mean(0)

    sigma_xx = xc.T @ xc / (n - 1)
    sigma_xz = xc.T @ zc / (n - 1)

    w, w_pinv = _inv_sqrt_psd(sigma_xx, eps)

    # Orthogonal projector onto the column space of W @ Sigma_XZ. QR would keep
    # spurious directions when the matrix is rank-deficient (it is, whenever a
    # PII type is absent from the fitting split), so use the SVD and drop the
    # near-zero singular directions.
    m = w @ sigma_xz
    u, s, _ = torch.linalg.svd(m, full_matrices=False)
    keep = s > (s.max() * 1e-6) if s.numel() and float(s.max()) > 0 else s > 1
    u = u[:, keep]
    proj = u @ u.T if u.shape[1] else torch.zeros_like(sigma_xx)

    a = torch.eye(x.shape[1], dtype=x.dtype) - w_pinv @ proj @ w
    return LeaceEraser(a.to(torch.float32), mu.to(torch.float32))


class LeaceEraser(nn.Module):
    """Frozen affine map ``r(x) = A (x - mu) + mu``. Differentiable in x."""

    def __init__(self, a: torch.Tensor, mu: torch.Tensor):
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("mu", mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.mu
        return centered @ self.a.transpose(0, 1).to(x.dtype) + self.mu.to(x.dtype)

    @torch.no_grad()
    def residual_covariance(self, x: torch.Tensor, z: torch.Tensor) -> float:
        """Sanity check of the guarantee: ||Cov(r(x), z)||_F after erasure.

        Should be ~0 on the fitting data. A large value means the fit did not
        take (usually: too few samples, or a concept absent from the split), and
        the linear-guardedness claim does NOT hold — check this before reporting.
        """
        xr = self(x).to(torch.float64)
        if z.dim() == 1:
            z = torch.nn.functional.one_hot(
                z.long(), num_classes=int(z.max().item()) + 1)
        z = z.to(torch.float64)
        xc, zc = xr - xr.mean(0), z - z.mean(0)
        cov = xc.T @ zc / max(1, xr.shape[0] - 1)
        return float(torch.linalg.matrix_norm(cov))
