"""MACULAR model core (proposal 11). Backbone-agnostic, CPU-runnable.

This implements the differentiable heart of MACULAR on top of the Canonical
Document Representation (CDR) — i.e. everything that does NOT need the actual
VLM weights:

  - CDR projector                      (11.3)
  - PII head + differentiable gate     (11.4-11.5)  <- clinical loss backprops
                                                        through m into the PII head
  - Layout-aware relation graph        (11.7)  (a small Transformer encoder)
  - Clinical head (safe student)       (11.8)
  - EMA raw teacher + consistency      (11.6)
  - combined loss                      (11.12)

The real VLM backbones (Qwen/Ministral/Llama) plug in via BackboneAdapter
(macular/models/backbone.py) and produce the per-region features consumed here.
A MockBackboneAdapter lets the whole thing train on CPU for wiring tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MacularConfig:
    d_in: int = 64        # per-region feature dim from the backbone/CDR input
    d: int = 128          # shared model dim
    n_pii_classes: int = 8   # index 0 == NON_PII
    n_clinical: int = 12     # clinical field classes (incl. an "O"/none class)
    n_heads: int = 4
    n_graph_layers: int = 2
    ema_decay: float = 0.99
    # --- privacy adversary (proposal 11.9) ---
    use_adversary: bool = True
    d_recon: int = 54         # width of the text signature the attacker rebuilds
    # --- ablation switches (proposal 17.1) ---
    hard_mask: bool = False   # replace the differentiable gate with a threshold
    use_graph: bool = True    # disable the layout-aware relation graph
    mask_threshold: float = 0.5
    # --- redaction mechanism ---
    # "gate"  : the differentiable gate (original design; our ablation found no
    #           privacy advantage over "hard", so it is no longer the default
    #           claim, only the default code path for backwards compatibility)
    # "hard"  : threshold + detach, the classic sequential pipeline (privacy floor)
    # "leace" : closed-form linear concept erasure on the pooled region features,
    #           installed with model.set_eraser(); the gate is bypassed entirely
    # "none"  : no redaction at all (unprotected upper bound on leakage)
    redaction: str = "gate"


class RedactionGate(nn.Module):
    """Differentiable, layout-aware redaction gate (proposal 11.5).

        m_i     = 1 - P(NON_PII | region_i)
        E_i     = sum_t p_pii[i,t] * type_embedding[t]      (soft type embedding)
        z_safe  = (1 - m_i) * z_i + m_i * E_i

    Because z_safe depends on m (hence on the PII head), a clinical loss computed
    on z_safe backpropagates into the PII head — the core MACULAR property.
    """

    def __init__(self, cfg: MacularConfig):
        super().__init__()
        self.cfg = cfg
        self.pii_head = nn.Linear(cfg.d, cfg.n_pii_classes)
        # type embeddings represent PII *type*, never the literal value.
        self.type_emb = nn.Parameter(torch.randn(cfg.n_pii_classes, cfg.d) * 0.02)

    def forward(self, z: torch.Tensor):
        pii_logits = self.pii_head(z)                  # (B, N, P)
        p_pii = F.softmax(pii_logits, dim=-1)
        m = 1.0 - p_pii[..., 0:1]                      # (B, N, 1); class 0 = NON_PII
        e_type = p_pii @ self.type_emb                 # (B, N, d) soft type embed
        if self.cfg.hard_mask:
            # Ablation: threshold the mask and detach it. This is the classic
            # sequential pipeline — the clinical loss can no longer reach the
            # PII head, which is exactly what the differentiable gate buys.
            m_used = (m > self.cfg.mask_threshold).float().detach()
            e_used = e_type.detach()
        else:
            m_used, e_used = m, e_type
        z_safe = (1.0 - m_used) * z + m_used * e_used
        return z_safe, pii_logits, m_used


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradientReversal.apply(x, lambd)


class PrivacyAdversary(nn.Module):
    """Reconstructs the original text signature from the SAFE representation
    (proposal 11.9).

    Trained through a gradient-reversal layer: the adversary minimizes its own
    reconstruction error while the encoder/gate is pushed to make that
    reconstruction *harder*, i.e. to strip literal PII content from z_safe.

    NOTE: this in-training adversary is a training signal, not evidence. The
    proposal is explicit that privacy must be judged by a HELD-OUT attacker
    (different weights, trained post-hoc on frozen representations) — see
    ``evaluate_leakage`` in train.py.
    """

    def __init__(self, cfg: MacularConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d, cfg.d), nn.ReLU(), nn.Linear(cfg.d, cfg.d_recon))

    def forward(self, z_safe, lambd: float = 1.0):
        return self.net(grad_reverse(z_safe, lambd))


class RelationGraph(nn.Module):
    """Layout-aware contextualizer over regions (proposal 11.7).

    A compact Transformer encoder stands in for the relation graph: self-
    attention lets a region attend to same-row/column/key-value neighbours.
    Positional structure is injected via a linear projection of the box.
    """

    def __init__(self, cfg: MacularConfig):
        super().__init__()
        self.box_proj = nn.Linear(4, cfg.d)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d, nhead=cfg.n_heads, dim_feedforward=cfg.d * 2,
            batch_first=True, dropout=0.0)
        # enable_nested_tensor=False: avoids the prototype nested-tensor fast
        # path (and its warning) when a padding mask is passed; deterministic.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_graph_layers, enable_nested_tensor=False)

    def forward(self, z, boxes, key_padding_mask=None):
        h = z + self.box_proj(boxes)
        return self.encoder(h, src_key_padding_mask=key_padding_mask)


class MacularModel(nn.Module):
    def __init__(self, cfg: MacularConfig):
        super().__init__()
        self.cfg = cfg
        self.projector = nn.Linear(cfg.d_in, cfg.d)
        self.gate = RedactionGate(cfg)
        self.graph = RelationGraph(cfg)
        self.clinical_student = nn.Linear(cfg.d, cfg.n_clinical)
        # EMA "raw teacher" clinical head: same shape, updated by EMA, no grad.
        self.clinical_teacher = nn.Linear(cfg.d, cfg.n_clinical)
        self.clinical_teacher.load_state_dict(self.clinical_student.state_dict())
        for p in self.clinical_teacher.parameters():
            p.requires_grad_(False)
        self.adversary = PrivacyAdversary(cfg) if cfg.use_adversary else None
        # Installed by set_eraser() when cfg.redaction == "leace". Not a
        # parameter: LEACE is fit in closed form on TRAIN features, then frozen.
        self.eraser = None
        if cfg.redaction == "hard":
            self.cfg.hard_mask = True

    def set_eraser(self, eraser):
        """Install a fitted LEACE eraser (see macular.privacy.fit_leace).

        Must be fit on TRAIN features only — fitting on evaluation features would
        leak exactly the labels being erased.
        """
        self.eraser = eraser
        if eraser is not None:
            self.cfg.redaction = "leace"
        return self

    def _contextualize(self, z, boxes, key_padding_mask):
        if not self.cfg.use_graph:      # ablation: no relation graph
            return z
        return self.graph(z, boxes, key_padding_mask)

    def forward(self, region_feats, boxes, key_padding_mask=None,
                adv_lambda: float = 1.0):
        z = self.projector(region_feats)                       # (B,N,d)

        # Raw view -> teacher target (stop-grad).
        z_ctx_raw = self._contextualize(z, boxes, key_padding_mask)
        with torch.no_grad():
            teacher_logits = self.clinical_teacher(z_ctx_raw)

        # Safe view -> student.
        if self.cfg.redaction == "leace" and self.eraser is not None:
            # Erasure happens on the POOLED REGION FEATURES, before the
            # projector — that is the object the guarantee is stated over, and
            # the object an attacker would hold. The PII head still runs on the
            # raw view because PII typing is a task output, not a leak: the
            # safe representation never sees it.
            z_safe = self.projector(self.eraser(region_feats))
            pii_logits = self.gate.pii_head(z)
            m = torch.ones_like(z[..., :1])
        elif self.cfg.redaction == "none":
            z_safe = z
            pii_logits = self.gate.pii_head(z)
            m = torch.zeros_like(z[..., :1])
        else:                                   # "gate" | "hard"
            z_safe, pii_logits, m = self.gate(z)
        z_ctx_safe = self._contextualize(z_safe, boxes, key_padding_mask)
        student_logits = self.clinical_student(z_ctx_safe)

        out = {
            "pii_logits": pii_logits,
            "clinical_student": student_logits,
            "clinical_teacher": teacher_logits.detach(),
            "m": m,
            "z_safe": z_safe,
            # The representation that actually flows downstream. Regions attend
            # to each other here, so a region redacted in isolation can absorb
            # content from its neighbours: attacking only `z_safe` measures the
            # mechanism, while attacking `z_ctx_safe` measures the deployment.
            # Both must be reported.
            "z_ctx_safe": z_ctx_safe,
        }
        if self.adversary is not None:
            out["adv_recon"] = self.adversary(z_safe, adv_lambda)
        return out

    @torch.no_grad()
    def update_teacher(self):
        d = self.cfg.ema_decay
        for t, s in zip(self.clinical_teacher.parameters(),
                        self.clinical_student.parameters()):
            t.mul_(d).add_(s, alpha=1 - d)


def macular_loss(outputs, pii_labels, clinical_labels, valid_mask=None,
                 w_pii=1.0, w_clinical=1.0, w_cons=1.0, pii_weight=None,
                 w_adv=1.0, recon_target=None):
    """Combined loss (proposal 11.12, minimal form).

    - L_pii:      cross-entropy on PII type (class 0 = NON_PII). ``pii_weight``
                  is a per-class weight tensor implementing the cost-sensitive
                  PII loss the proposal requires (11.4) — PII misses cost more
                  than false positives, so NON_PII is down-weighted.
    - L_clinical: cross-entropy on clinical field (safe student)
    - L_cons:     KL(student || teacher) on NON-PII regions only (11.6)
    """
    B, N, P = outputs["pii_logits"].shape
    C = outputs["clinical_student"].shape[-1]
    if valid_mask is None:
        valid_mask = torch.ones(B, N, dtype=torch.bool)

    flat = valid_mask.reshape(-1)
    pii_logits = outputs["pii_logits"].reshape(-1, P)[flat]
    clin_student = outputs["clinical_student"].reshape(-1, C)[flat]
    clin_teacher = outputs["clinical_teacher"].reshape(-1, C)[flat]
    pii_t = pii_labels.reshape(-1)[flat]
    clin_t = clinical_labels.reshape(-1)[flat]

    l_pii = F.cross_entropy(pii_logits, pii_t, weight=pii_weight)
    l_clin = F.cross_entropy(clin_student, clin_t)

    # consistency only on NON-PII regions (pii label == 0), on clinical dists
    nonpii = pii_t == 0
    if nonpii.any():
        log_student = F.log_softmax(clin_student[nonpii], dim=-1)
        target = F.softmax(clin_teacher[nonpii], dim=-1)
        l_cons = F.kl_div(log_student, target, reduction="batchmean")
    else:
        l_cons = torch.zeros((), device=pii_logits.device)

    # Adversarial term (proposal 11.9): the adversary rebuilds the original text
    # signature from z_safe. Gradient reversal means minimizing this loss trains
    # the attacker while pushing the encoder to hide the content from it. Scored
    # on PII regions only — those are the ones whose content must not survive.
    l_adv = torch.zeros((), device=pii_logits.device)
    if recon_target is not None and "adv_recon" in outputs:
        recon = outputs["adv_recon"].reshape(-1, outputs["adv_recon"].shape[-1])[flat]
        target = recon_target.reshape(-1, recon_target.shape[-1])[flat]
        pii_rows = pii_t > 0
        if pii_rows.any():
            l_adv = F.mse_loss(recon[pii_rows], target[pii_rows])

    total = (w_pii * l_pii + w_clinical * l_clin + w_cons * l_cons
             + w_adv * l_adv)
    parts = {"pii": l_pii.detach().item(), "clinical": l_clin.detach().item(),
             "consistency": l_cons.detach().item(),
             "adversary": l_adv.detach().item(), "total": total.detach().item()}
    return total, parts
