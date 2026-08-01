"""Real VLM backbone adapters (proposal 11.2, C1).

Turns a page image into per-region features with a SINGLE backbone forward pass,
then ROI-pools the vision patch grid onto each region box. The rest of MACULAR
(gate, graph, heads) is unchanged across backbones — that is the whole point of
RQ5/RQ6: same method, swappable backbone.

Supported families (all via one generic path, with per-model quirks isolated):
  qwen      Qwen/Qwen3-VL-*-Instruct              Apache-2.0
  ministral mistralai/Ministral-3-8B-Instruct-*   Apache-2.0
  llama     meta-llama/Llama-3.2-*-Vision-*       Llama Community License (gated)
  kimi      moonshotai/Kimi-VL-A3B-*              MoE, needs trust_remote_code
  internvl  OpenGVLab/InternVL*                   trust_remote_code

Design notes
------------
* We take the VISION TOWER's patch embeddings, not the LM hidden states: they
  keep a 2-D grid we can ROI-pool, and skipping the LM makes a page forward
  cheap enough to run over a whole dataset.
* ``single forward`` is enforced by construction — one vision-tower call per
  page, all regions pooled from the same grid (proposal 10, and the reason the
  EMA teacher/safe student pair stays affordable).
* Weights stay frozen here; only the MACULAR head/projector train. That matches
  staged training Stage 1 (proposal 11.13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# family -> (default model id, needs trust_remote_code)
FAMILIES = {
    "qwen": ("Qwen/Qwen3-VL-8B-Instruct", False),
    "qwen2b": ("Qwen/Qwen3-VL-2B-Instruct", False),
    "ministral": ("mistralai/Ministral-3-8B-Instruct-2512", False),
    "llama": ("meta-llama/Llama-3.2-11B-Vision-Instruct", False),
    # Kimi's remote code imports transformers internals removed in recent
    # versions (is_torch_fx_available); it cannot load alongside Qwen3-VL, which
    # needs a newer transformers. Kept here for the record.
    "kimi": ("moonshotai/Kimi-VL-A3B-Instruct", True),
    "internvl": ("OpenGVLab/InternVL2-1B", True),
    # The proposal's own target OCR model, natively supported by transformers.
    "paddleocr_vl": ("PaddlePaddle/PaddleOCR-VL", False),
}


@dataclass
class VLMBackboneConfig:
    family: str = "qwen2b"
    model_id: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_pixels: int = 640 * 640      # cap page resolution for memory
    pool: str = "roi"                # roi | mean
    # --- LoRA fine-tuning of the vision tower (proposal 11.13 Stage 1) ---
    # With a frozen backbone the redaction gate can only reweight fixed
    # features; it cannot reshape them, so it has little room to hide PII.
    # Enabling this makes the vision tower trainable through LoRA so the
    # gate/adversary can actually influence the representation.
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def _torch_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


class VLMBackbone:
    """Loads a vision-language model and exposes ``encode_page``.

    ``encode_page(image, boxes) -> (N, d_in)`` where ``d_in`` is the backbone's
    vision hidden size. One forward per page.
    """

    def __init__(self, cfg: VLMBackboneConfig):
        self.cfg = cfg
        model_id, trust = FAMILIES.get(cfg.family, (cfg.family, True))
        self.model_id = cfg.model_id or model_id
        self.trust_remote_code = trust
        self._model = None
        self._processor = None
        self.d_in: int | None = None

    # -- loading ------------------------------------------------------------

    def load(self):
        if self._model is not None:
            return self
        from transformers import AutoProcessor, AutoModel

        dtype = _torch_dtype(self.cfg.dtype)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code)

        model = None
        errors = []
        # Try the vision-language classes in order of specificity; families
        # differ in which one they register.
        for loader in self._loaders():
            try:
                model = loader(dtype)
                break
            except Exception as e:  # noqa: BLE001 - report all attempts
                errors.append(f"{loader.__name__}: {type(e).__name__}: {e}")
        if model is None:
            raise RuntimeError(
                f"could not load {self.model_id}. Attempts:\n  " +
                "\n  ".join(errors))

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._vision = self._find_vision_tower(model)
        self.d_in = self._infer_hidden_size()
        if self.cfg.lora:
            self._apply_lora()
        return self

    def _apply_lora(self):
        """Wrap the vision tower's attention/MLP projections with LoRA."""
        from peft import LoraConfig, get_peft_model

        targets = self._lora_targets()
        if not targets:
            raise RuntimeError(
                f"no LoRA target modules found in the vision tower of "
                f"{self.model_id}")
        peft_cfg = LoraConfig(r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                              lora_dropout=self.cfg.lora_dropout, bias="none",
                              target_modules=targets)
        self._vision = get_peft_model(self._vision, peft_cfg)
        self._vision.train()
        self.lora_trainable = sum(p.numel() for p in self._vision.parameters()
                                  if p.requires_grad)

    def _lora_targets(self) -> list[str]:
        """Leaf Linear module names inside the vision tower, deduplicated.

        Names differ per family (q_proj/k_proj/... vs qkv/fc1/...), so they are
        discovered rather than hard-coded.
        """
        import torch.nn as nn
        names = set()
        for name, mod in self._vision.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf and not leaf.isdigit():
                    names.add(leaf)
        preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj",
                     "fc1", "fc2", "gate_proj", "up_proj", "down_proj",
                     "linear_1", "linear_2"}
        hit = sorted(names & preferred)
        return hit or sorted(names)[:4]

    def trainable_parameters(self):
        if not self.cfg.lora or self._vision is None:
            return []
        return [p for p in self._vision.parameters() if p.requires_grad]

    def _loaders(self):
        from transformers import AutoModel

        def auto_model(dtype):
            return AutoModel.from_pretrained(
                self.model_id, dtype=dtype, device_map=self.cfg.device,
                trust_remote_code=self.trust_remote_code)

        def image_text_to_text(dtype):
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=dtype, device_map=self.cfg.device,
                trust_remote_code=self.trust_remote_code)

        def vision2seq(dtype):
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(
                self.model_id, dtype=dtype, device_map=self.cfg.device,
                trust_remote_code=self.trust_remote_code)

        def causal_lm(dtype):
            # Kimi-VL and other remote-code VLMs register only under CausalLM.
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, device_map="auto",
                trust_remote_code=self.trust_remote_code)

        return [image_text_to_text, auto_model, vision2seq, causal_lm]

    @staticmethod
    def _find_vision_tower(model):
        """Locate the vision tower across naming conventions."""
        for path in ("visual", "vision_tower", "vision_model",
                     "model.visual", "model.vision_tower", "model.vision_model"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if obj is not None and hasattr(obj, "forward"):
                    return obj
            except AttributeError:
                continue
        return None

    def _infer_hidden_size(self):
        cfg = self._model.config
        for attr in ("vision_config", "vision_tower_config"):
            vc = getattr(cfg, attr, None)
            if vc is not None:
                for k in ("hidden_size", "embed_dim", "hidden_dim"):
                    if getattr(vc, k, None):
                        return int(getattr(vc, k))
        for k in ("hidden_size", "embed_dim"):
            if getattr(cfg, k, None):
                return int(getattr(cfg, k))
        return None

    # -- encoding -----------------------------------------------------------

    def encode_page(self, image, boxes) -> torch.Tensor:
        """Single forward over the page; ROI-pool patch features per box.

        image: PIL.Image     boxes: (N,4) normalized xyxy tensor/list
        returns: (N, d_in) float32 tensor (CPU when frozen, on-device with LoRA
        so gradients can flow back into the vision tower).
        """
        if self._model is None:
            self.load()
        if self.cfg.lora:
            return self._encode_page_grad(image, boxes)
        with torch.no_grad():
            feats, gh, gw = self._page_patch_grid(image)   # (P, D), grid h/w
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            return self._roi_pool(feats, gh, gw, boxes_t)

    def _encode_page_grad(self, image, boxes) -> torch.Tensor:
        """Gradient-preserving variant used when LoRA is enabled."""
        feats, gh, gw = self._page_patch_grid(image, keep_grad=True)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        return self._roi_pool(feats, gh, gw, boxes_t, keep_grad=True)

    def _image_token(self) -> str:
        """The placeholder some processors require in the text (e.g. Mllama)."""
        proc = self._processor
        for attr in ("image_token", "image_token_str"):
            tok = getattr(proc, attr, None)
            if isinstance(tok, str) and tok:
                return tok
            if tok is not None and hasattr(tok, "content"):
                return tok.content
        return "<|image|>"

    @staticmethod
    def _finish(feats, keep_grad: bool):
        """Frozen path: detach to float32 CPU (cheap, cacheable).
        LoRA path: stay on-device in float32 so gradients keep flowing.

        The detach is explicit rather than relying on an enclosing no_grad():
        otherwise calling this directly would silently retain the autograd graph
        of a whole vision forward per page.
        """
        return feats.float() if keep_grad else feats.detach().float().cpu()

    def _page_patch_grid(self, image, keep_grad: bool = False):
        """Run the vision tower once and return (patches, grid_h, grid_w)."""
        proc = self._processor
        try:
            inputs = proc(images=image, text="", return_tensors="pt")
        except ValueError as e:
            # Mllama-style processors count image placeholders in the text and
            # refuse an empty prompt.
            if "image tokens" not in str(e):
                raise
            inputs = proc(images=image, text=self._image_token(),
                          return_tensors="pt")
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("processor produced no pixel_values")

        grid_thw = inputs.get("image_grid_thw")
        out = self._run_vision(pixel_values, grid_thw, inputs)
        feats = out if isinstance(out, torch.Tensor) else out.last_hidden_state

        # Mllama tiles the page: (B, media, tiles, patches, D). Tiles must be
        # stitched back into one grid or every region maps to the wrong place.
        if feats.dim() == 5:
            tiled = self._assemble_tiles(feats, inputs, keep_grad)
            if tiled is not None:
                return tiled
            feats = feats.reshape(-1, feats.shape[-1])
        while feats.dim() > 2:    # (B, P, D) / (B, media, P, D) -> (P, D)
            feats = feats[0]
        p, d = feats.shape

        if grid_thw is not None:                      # Qwen-style explicit grid
            t, h, w = [int(x) for x in grid_thw[0].tolist()]
            merge = getattr(getattr(self._model.config, "vision_config", None),
                            "spatial_merge_size", 1) or 1
            gh, gw = max(1, h // merge), max(1, w // merge)
            if gh * gw != p:                          # merge already applied
                gh, gw = h, w
            if gh * gw != p:
                gh = gw = int(math.sqrt(p))
            return self._finish(feats, keep_grad), gh, gw

        # No explicit grid (Pixtral/SigLIP/CLIP style): derive it from the
        # PROCESSED image size and patch size. Assuming a square grid via
        # sqrt(P) silently scrambles every non-square page — which looks exactly
        # like "this backbone's features are useless".
        gh, gw = self._grid_from_image(inputs, p)
        if gh * gw != p:
            # Pixtral appends one [IMG_BREAK] per row except the last.
            if gh * gw + (gh - 1) == p:
                keep = torch.ones(p, dtype=torch.bool)
                for r in range(1, gh):                # break token ends each row
                    keep[r * gw + (r - 1)] = False
                feats = feats[keep]
                p = feats.shape[0]
            elif gh * gw + 1 == p:                    # leading CLS token
                feats = feats[1:]
                p = feats.shape[0]
        if gh * gw != p:                              # last resort
            side = int(math.sqrt(p))
            gh = gw = max(1, side)
        return self._finish(feats, keep_grad), gh, gw

    # Mllama's tile layouts, indexed 1-based by aspect_ratio_ids.
    _MLLAMA_RATIOS = [(1, 1), (1, 2), (1, 3), (1, 4),
                      (2, 1), (2, 2), (3, 1), (4, 1)]

    def _assemble_tiles(self, feats, inputs, keep_grad: bool = False):
        """Stitch Mllama's per-tile patch grids into one page grid.

        feats: (B, media, tiles, patches, D). Each tile is a square patch grid
        (plus a leading CLS token); tiles are laid out per aspect_ratio_ids.
        Returns (patches, grid_h, grid_w) or None if the layout is unclear.
        """
        f = feats[0, 0]                      # (tiles, patches, D)
        n_tiles, p_per_tile, d = f.shape
        side = int(math.sqrt(p_per_tile))
        if side * side != p_per_tile:        # drop CLS-like token
            side = int(math.sqrt(p_per_tile - 1))
            if side * side != p_per_tile - 1:
                return None
            f = f[:, 1:, :]

        ids = inputs.get("aspect_ratio_ids")
        th = tw = None
        if ids is not None:
            try:
                idx = int(ids.reshape(-1)[0]) - 1     # ids are 1-based
                ratios = getattr(getattr(self._model.config, "vision_config", None),
                                 "supported_aspect_ratios", None) or self._MLLAMA_RATIOS
                th, tw = [int(x) for x in ratios[idx]]
            except Exception:
                th = tw = None
        if not th or th * tw > n_tiles:
            return None

        grid = torch.zeros(th * side, tw * side, d, dtype=f.dtype, device=f.device)
        for t in range(th * tw):
            r, c = divmod(t, tw)
            tile = f[t].reshape(side, side, d)
            grid[r * side:(r + 1) * side, c * side:(c + 1) * side] = tile
        gh, gw = th * side, tw * side
        return self._finish(grid.reshape(gh * gw, d), keep_grad), gh, gw

    def _grid_from_image(self, inputs, n_patches):
        """(grid_h, grid_w) from the processed image size and patch size."""
        vc = getattr(self._model.config, "vision_config", None)
        patch = getattr(vc, "patch_size", None) or getattr(
            self._model.config, "patch_size", None) or 16
        H = W = None
        sizes = inputs.get("image_sizes")
        if sizes is not None:
            try:
                H, W = [int(x) for x in list(sizes[0])[:2]]
            except Exception:
                H = W = None
        if H is None:
            pv = inputs.get("pixel_values")
            if pv is not None and pv.dim() == 4:       # (B, C, H, W)
                H, W = int(pv.shape[-2]), int(pv.shape[-1])
        if not H or not W:
            side = int(math.sqrt(n_patches))
            return side, side
        return max(1, H // patch), max(1, W // patch)

    # Extra tensors some vision towers need beyond pixel_values (Mllama).
    _VISION_EXTRAS = ("aspect_ratio_ids", "aspect_ratio_mask")

    def _run_vision(self, pixel_values, grid_thw, inputs=None):
        vt = self._vision
        if vt is None:
            raise RuntimeError(f"no vision tower found on {self.model_id}")
        extras = {}
        if inputs:
            extras = {k: inputs[k] for k in self._VISION_EXTRAS if k in inputs}

        # Vision towers disagree on the expected rank: some take flattened
        # patches (P, C, ph, pw), others require a leading batch/sequence dim
        # (B, P, C, ph, pw). Try as-is, then with a batch dim added.
        variants = [pixel_values]
        if pixel_values.dim() == 4:
            variants.append(pixel_values.unsqueeze(0))
        kwarg_sets = []
        if grid_thw is not None:
            kwarg_sets.append({"grid_thw": grid_thw, **extras})
        if extras:
            kwarg_sets.append(dict(extras))
        kwarg_sets.append({})

        last = None
        for pv in variants:
            for kwargs in kwarg_sets:
                try:
                    return vt(pv, **kwargs)
                except (TypeError, ValueError) as e:
                    last = e
                    continue
        raise RuntimeError(
            f"vision tower of {self.model_id} rejected all input shapes: {last}")

    @staticmethod
    def _roi_pool(feats, gh, gw, boxes, keep_grad: bool = False) -> torch.Tensor:
        """Average patch features inside each normalized box.

        With ``keep_grad`` the result is built by stacking (not by in-place
        assignment into a zeros tensor), so autograd can reach the vision tower.
        """
        p, d = feats.shape
        usable = min(p, gh * gw)
        grid = feats[:usable].reshape(gh, gw, d) if usable == gh * gw else None

        if grid is None:                    # fall back to a global mean
            mean = feats.mean(0)
            if keep_grad:
                return mean.unsqueeze(0).expand(boxes.shape[0], d)
            out = torch.zeros(boxes.shape[0], d)
            out[:] = mean.detach()
            return out

        pooled = []
        for x0, y0, x1, y1 in boxes.tolist():
            c0 = max(0, min(gw - 1, int(x0 * gw)))
            c1 = max(c0 + 1, min(gw, int(math.ceil(x1 * gw))))
            r0 = max(0, min(gh - 1, int(y0 * gh)))
            r1 = max(r0 + 1, min(gh, int(math.ceil(y1 * gh))))
            pooled.append(grid[r0:r1, c0:c1].reshape(-1, d).mean(0))
        if keep_grad:
            return torch.stack(pooled)
        out = torch.zeros(boxes.shape[0], d)
        for i, v in enumerate(pooled):
            out[i] = v.detach()
        return out
