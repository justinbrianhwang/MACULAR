"""MACULAR model core (proposal 11).

Implemented and CPU-runnable (real torch modules):
  - MacularConfig, MacularModel         : the assembled core
  - RedactionGate                       : differentiable gate (11.5)
  - RelationGraph                       : layout-aware contextualizer (11.7)
  - macular_loss                        : combined loss (11.12)
  - BackboneAdapter / MockBackboneAdapter, make_synthetic_batch
  - train_step, fit_synthetic           : training loop (11.13)

NOT yet implemented (needs a GPU + the real weights):
  - real BackboneAdapter for Qwen3-VL / Ministral-3 / Llama-3.2-Vision
    (single-forward ROI encoding + region recognition, 11.2 / RQ6)
  - privacy leakage adversary (11.9), FHIR compiler (11.11), regional
    re-reading (11.10) — scaffolded conceptually, wire in next.

``torch`` is required for this subpackage:  pip install -e ".[model]"
"""

from __future__ import annotations


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            'MACULAR model core needs torch: pip install -e ".[model]"') from e


try:  # keep the package importable even without torch installed
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

if _HAS_TORCH:
    from .core import (  # noqa: F401
        MacularConfig, MacularModel, RedactionGate, RelationGraph, macular_loss,
    )
    from .backbone import (  # noqa: F401
        BackboneAdapter, MockBackboneAdapter, make_synthetic_batch,
    )
    from .train import (  # noqa: F401
        train_step, fit_synthetic, fit_on_documents, ocr_error_propagation,
        engine_downstream_comparison, fit_with_vlm, backbone_contribution,
        macular_ablation, MACULAR_ABLATIONS,
        erasure_comparison, ERASURE_MECHANISMS,
    )
    from .vlm_backbone import VLMBackbone, VLMBackboneConfig, FAMILIES  # noqa: F401
    from .features import (  # noqa: F401
        documents_to_batch, config_for_features, corrupt_text, FEATURE_DIM,
        N_PII_CLASSES, N_CLINICAL,
    )
