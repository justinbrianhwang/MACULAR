"""Experiment runner and CLI.

Commands:
  macular probe                      -> capability report (GPU, tesseract, ...)
  macular list                       -> available experiments
  macular run <experiment> [--config cfg.yaml] [--out results/]

Experiments that RUN TODAY (CPU, produce result JSON to send back):
  data_gen            generate synthetic dataset (images + labels)
  shortcut_audit      coordinate-only PII baseline (gate 5)
  ocr_baseline        Tesseract CER/WER (needs the ocr extra)
  data_stats          dataset summary statistics

Experiments that are SCAFFOLDED (need model core + GPU):
  train_macular, ablation, backbone_swap  -> raise NotImplementedError clearly
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Optional

DEFAULT_OUT = "results"
DEFAULT_DATA = os.path.join("data", "meddoc")

RUNNABLE = ["data_gen", "shortcut_audit", "ocr_baseline", "data_stats",
            "fetch_funsd", "fetch_xfund", "train_core", "ocr_propagation",
            "engine_downstream", "ocr_cache", "backbone_gate", "ablation",
            "lora_ablation", "erasure_comparison"]
SCAFFOLDED = ["train_macular", "backbone_swap"]


# --- environment probe -----------------------------------------------------

def probe_env() -> dict:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    # GPU
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        report["torch"] = None
        report["cuda_available"] = False
    # OCR engines
    try:
        from .baselines import ocr
        report["ocr_engines"] = {
            name: cls().available() for name, cls in ocr.ENGINES.items()
        }
    except Exception:
        report["ocr_engines"] = {}
    return report


# --- config ----------------------------------------------------------------

def load_config(path: Optional[str]) -> dict:
    cfg = dict(_DEFAULT_CONFIG)
    if path:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg.update(user)
    return cfg


_DEFAULT_CONFIG = {
    "seed": 0,
    "data_dir": DEFAULT_DATA,
    "n_per_split": 200,          # co-author machine is stronger; bump freely
    "languages": ["en", "ko", "ja"],
    "counterfactual_layout": False,
    "render_images": True,
    "doc_types": ["laboratory_report", "prescription"],
    "ocr_engine": "tesseract",   # tesseract | easyocr  (per-doc language auto)
    # real public (non-PHI) scanned-document datasets
    "funsd_raw": "data/funsd_raw",
    "xfund_raw": "data/xfund_raw",
    "xfund_langs": ["ja", "es"],
    # model-core / GPU settings (assume the stronger co-author GPU)
    "device": "cuda",
    "dtype": "bfloat16",
    "backbone": "Qwen/Qwen3-VL-8B-Instruct",
    "train_batch_size": 4,
    "grad_accum": 4,
    "max_page_pixels": 1_600_000,
}


# --- experiments -----------------------------------------------------------

def _exp_data_gen(cfg: dict) -> dict:
    from .data.generate import generate_dataset
    manifest = generate_dataset(
        out_dir=cfg["data_dir"],
        n_per_split=cfg["n_per_split"],
        languages=cfg["languages"],
        seed=cfg["seed"],
        counterfactual_layout=cfg["counterfactual_layout"],
        render_images=cfg["render_images"],
        doc_types=cfg.get("doc_types"),
    )
    return {"experiment": "data_gen", "manifest": manifest}


def _exp_fetch_funsd(cfg: dict) -> dict:
    from .realdata import funsd
    manifest = funsd.fetch_and_convert(cfg["funsd_raw"], cfg["data_dir"])
    return {"experiment": "fetch_funsd", "manifest": manifest,
            "next": "macular run ocr_baseline --config <this config>"}


def _exp_fetch_xfund(cfg: dict) -> dict:
    from .realdata import xfund
    manifest = xfund.fetch_and_convert(
        cfg["xfund_raw"], cfg["data_dir"], langs=cfg.get("xfund_langs", ["ja", "es"]))
    return {"experiment": "fetch_xfund", "manifest": manifest,
            "next": "macular run ocr_baseline --config <this config>"}


def _load_split(cfg, split):
    from .schema import read_jsonl
    path = os.path.join(cfg["data_dir"], f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run:  macular run data_gen  first."
        )
    return read_jsonl(path)


def _exp_shortcut_audit(cfg: dict) -> dict:
    from .baselines import coordinate_only
    train = _load_split(cfg, "train")
    test = _load_split(cfg, "test")
    return coordinate_only.run(train, test)


def _exp_ocr_baseline(cfg: dict) -> dict:
    from .baselines import ocr
    test = _load_split(cfg, "test")
    # Engine is swappable (tesseract | easyocr). Language is chosen per-document.
    return ocr.run(test, cfg["data_dir"], engine=cfg.get("ocr_engine", "tesseract"))


def _exp_data_stats(cfg: dict) -> dict:
    stats = {"experiment": "data_stats", "splits": {}}
    for split in ("train", "val", "test"):
        try:
            docs = _load_split(cfg, split)
        except FileNotFoundError:
            continue
        n_cand = sum(len(d.candidates) for d in docs)
        n_pii = sum(1 for d in docs for c in d.candidates if c.is_pii)
        langs: dict[str, int] = {}
        for d in docs:
            langs[d.language] = langs.get(d.language, 0) + 1
        stats["splits"][split] = {
            "n_documents": len(docs),
            "n_candidates": n_cand,
            "n_pii_candidates": n_pii,
            "pii_ratio": round(n_pii / n_cand, 4) if n_cand else 0.0,
            "languages": langs,
        }
    return stats


def _exp_train_core(cfg: dict) -> dict:
    """Train the MACULAR core on our real data via OCR-derived region features.

    Bridges the OCR pipeline to the model core: features come from region text
    (source='gt' = annotated text; source='ocr' = run an engine on crops so OCR
    errors propagate). CPU-runnable.
    """
    try:
        from .models import fit_on_documents
    except ImportError:
        return {"experiment": "train_core", "skipped": True,
                "reason": 'model core needs torch: pip install -e ".[model]"'}
    train = _load_split(cfg, "train")
    try:
        val = _load_split(cfg, "val")
    except FileNotFoundError:
        val = None

    engine = None
    source = cfg.get("feature_source", "gt")
    if source == "ocr":
        from .baselines import ocr
        cls = ocr.ENGINES.get(cfg.get("ocr_engine", "tesseract"))
        eng = cls() if cls else None
        engine = eng if (eng and eng.available()) else None
        if engine is None:
            source = "gt"   # fall back rather than produce empty text

    _model, result = fit_on_documents(
        train, val, epochs=cfg.get("train_epochs", 15),
        lr=cfg.get("lr", 2e-3), source=source, engine=engine,
        data_dir=cfg["data_dir"], max_docs=cfg.get("train_max_docs", 200))
    h = result["loss_history"]
    return {
        "experiment": "train_core",
        "feature_source": source,
        "n_train_docs": min(len(train), cfg.get("train_max_docs", 200)),
        "loss_start": h[0], "loss_end": h[-1],
        "train_pii_f1": result["train_pii_f1"],
        "val_pii_f1": result.get("val_pii_f1"),
    }


def _exp_ocr_propagation(cfg: dict) -> dict:
    """Quantify the OCR-error cascade: downstream PII performance vs OCR CER.

    This is the empirical version of the proposal's core problem statement
    (section 2) — sequential pipelines let OCR errors propagate into PII
    detection. Uses controlled corruption so it is reproducible on CPU.
    """
    try:
        from .models import ocr_error_propagation
    except ImportError:
        return {"experiment": "ocr_propagation", "skipped": True,
                "reason": 'needs torch: pip install -e ".[model]"'}
    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    levels = cfg.get("cer_levels", [0.0, 0.1, 0.2, 0.3, 0.5])
    rows = ocr_error_propagation(
        train, val, cer_levels=levels, epochs=cfg.get("train_epochs", 40),
        lr=cfg.get("lr", 3e-3), max_docs=cfg.get("train_max_docs", 120))
    clean = rows[0]["val_pii_f1"] if rows else None
    worst = rows[-1]["val_pii_f1"] if rows else None
    return {
        "experiment": "ocr_propagation",
        "n_train_docs": min(len(train), cfg.get("train_max_docs", 120)),
        "curve": rows,
        "f1_drop_clean_to_worst": (clean - worst) if (clean is not None) else None,
        "note": ("Downstream PII F1 vs simulated OCR CER. Demonstrates the "
                 "error cascade that motivates MACULAR (proposal section 2)."),
    }


def _exp_ocr_cache(cfg: dict) -> dict:
    """Run an OCR engine over all regions and cache the text to disk.

    Run this in an OCR-only environment when the engine cannot coexist with
    torch (PaddlePaddle vs PyTorch DLL conflict — proposal 8). Training then
    reads the cache instead of invoking the engine.
    """
    from .baselines import ocr
    from .baselines.ocr_cache import build_cache

    name = cfg.get("ocr_engine", "tesseract")
    cls = ocr.ENGINES.get(name)
    if cls is None:
        return {"experiment": "ocr_cache", "error": f"unknown engine '{name}'"}
    eng = cls()
    if not eng.available():
        return {"experiment": "ocr_cache", "engine": name, "skipped": True,
                "reason": f"engine '{name}' is not installed"}
    out = {"experiment": "ocr_cache", "splits": {}}
    for split in ("train", "val", "test"):
        try:
            docs = _load_split(cfg, split)
        except FileNotFoundError:
            continue
        out["splits"][split] = len(docs)
    # one cache covering every split (doc_ids are unique across splits)
    all_docs = []
    for split in ("train", "val", "test"):
        try:
            all_docs.extend(_load_split(cfg, split))
        except FileNotFoundError:
            pass
    out.update(build_cache(all_docs, cfg["data_dir"], eng,
                           max_docs=cfg.get("cache_max_docs")))
    return out


def _exp_engine_downstream(cfg: dict) -> dict:
    """Does a stronger OCR engine actually make de-identification safer?

    Trains the core on text produced by each real engine and compares downstream
    PII metrics against a perfect-text upper bound.
    """
    try:
        from .models import engine_downstream_comparison
    except ImportError:
        return {"experiment": "engine_downstream", "skipped": True,
                "reason": 'needs torch: pip install -e ".[model]"'}
    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    engines = cfg.get("engines", ["gt", "tesseract", "easyocr", "paddleocr"])
    rows = engine_downstream_comparison(
        train, val, engines, data_dir=cfg["data_dir"],
        epochs=cfg.get("train_epochs", 80), lr=cfg.get("lr", 3e-3),
        max_docs=cfg.get("train_max_docs", 120))
    return {
        "experiment": "engine_downstream",
        "n_train_docs": min(len(train), cfg.get("train_max_docs", 120)),
        "results": rows,
        "note": ("Downstream PII performance per OCR engine (counterfactual "
                 "layout required). 'gt' is the perfect-text upper bound."),
    }


def _exp_backbone_gate(cfg: dict) -> dict:
    """RQ6 / proposal 25 gate #3 — does the VLM backbone actually contribute?

    Runs A1 (text-only) vs A2 (real VLM vision features) per backbone family and
    reports Delta. A Delta of ~0 means the backbone is dead weight; the proposal
    requires reporting that outcome rather than hiding it.
    """
    try:
        from .models import backbone_contribution, VLMBackbone, VLMBackboneConfig
    except ImportError:
        return {"experiment": "backbone_gate", "skipped": True,
                "reason": 'needs torch: pip install -e ".[model]"'}
    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    families = cfg.get("backbones", ["qwen2b"])
    rows = []
    for fam in families:
        bcfg = VLMBackboneConfig(
            family=fam, device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "bfloat16"))
        bk = VLMBackbone(bcfg)
        try:
            bk.load()
        except Exception as e:
            rows.append({"backbone": fam, "loaded": False,
                         "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        try:
            res = backbone_contribution(
                train, val, bk, cfg["data_dir"],
                epochs=cfg.get("train_epochs", 80), lr=cfg.get("lr", 3e-3),
                max_docs=cfg.get("train_max_docs", 120))
            rows.append({"backbone": fam, "model_id": bk.model_id,
                         "loaded": True, **res})
        except Exception as e:
            rows.append({"backbone": fam, "model_id": bk.model_id,
                         "loaded": True, "error":
                         f"{type(e).__name__}: {str(e)[:300]}"})
        finally:
            import gc
            del bk
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    return {"experiment": "backbone_gate", "results": rows,
            "note": ("A1 = shared-parser text features, A2 = backbone vision "
                     "features. Delta<=0 means the backbone does not "
                     "contribute (report it, do not hide it).")}


def _exp_ablation(cfg: dict) -> dict:
    """MACULAR component ablation on real VLM features (proposal 17.1)."""
    try:
        from .models import macular_ablation, VLMBackbone, VLMBackboneConfig
    except ImportError:
        return {"experiment": "ablation", "skipped": True,
                "reason": 'needs torch: pip install -e ".[model]"'}
    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    out = {"experiment": "ablation", "backbones": {}}
    for fam in cfg.get("backbones", ["qwen2b"]):
        bk = VLMBackbone(VLMBackboneConfig(
            family=fam, device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "bfloat16")))
        try:
            bk.load()
            out["backbones"][fam] = macular_ablation(
                train, val, bk, cfg["data_dir"],
                epochs=cfg.get("train_epochs", 80), lr=cfg.get("lr", 3e-3),
                max_docs=cfg.get("train_max_docs", 120))
        except Exception as e:
            out["backbones"][fam] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        finally:
            import gc
            del bk
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    out["note"] = ("Privacy = PII average precision (threshold-free); utility = "
                   "clinical macro-F1 on the SAFE view. 'full' is MACULAR; the "
                   "others remove one component each.")
    return out


_LORA_VARIANTS = {
    "full":         ({}, {}),
    "no_gate":      ({"hard_mask": True}, {"w_cons": 0.0}),
    "no_adversary": ({"use_adversary": False}, {"w_adv": 0.0}),
    "no_consistency": ({}, {"w_cons": 0.0}),
    "hard_mask":    ({"hard_mask": True}, {}),
}


def _exp_erasure_comparison(cfg: dict) -> dict:
    """Closed-form erasure vs the gate vs the hard-mask floor, with real attacks.

    Replaces both halves of the failed gate experiment: the MECHANISM (LEACE
    instead of a differentiable gate that our ablation showed adds nothing) and
    the MEASUREMENT (discrete probe accuracy + inversion exact-match/CER instead
    of a cosine similarity that flipped sign between identical runs).
    """
    try:
        from .models import (VLMBackbone, VLMBackboneConfig, erasure_comparison)
    except ImportError:
        return {"experiment": "erasure_comparison", "skipped": True,
                "reason": 'needs torch: pip install -e ".[model]"'}
    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    out = {"experiment": "erasure_comparison", "backbones": {}}
    for fam in cfg.get("backbones", ["paddleocr_vl"]):
        bk = VLMBackbone(VLMBackboneConfig(
            family=fam, device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "bfloat16")))
        try:
            bk.load()
            out["backbones"][fam] = erasure_comparison(
                train, val, bk, cfg["data_dir"],
                epochs=cfg.get("train_epochs", 80), lr=cfg.get("lr", 3e-3),
                max_docs=cfg.get("max_docs", 120),
                seeds=tuple(cfg.get("seeds", [0, 1, 2])))
        except Exception as e:
            out["backbones"][fam] = {
                "error": f"{type(e).__name__}: {str(e)[:300]}"}
        finally:
            del bk
            _free_gpu()
        print(f"DONE {fam}", flush=True)
    return out


def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _agg_by_variant(rows, keys):
    """mean / std / n per variant, so a spread the size of the effect is visible.

    std is the sample std (ddof=1); with n=1 it is reported as None rather than
    0.0, because 0.0 reads as 'perfectly reproducible' when it means 'unmeasured'.
    """
    out = {}
    for r in rows:
        if "error" in r:
            continue
        out.setdefault(r["variant"], []).append(r)
    agg = {}
    for name, rs in out.items():
        stats = {"n_seeds": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if not vals:
                stats[k] = {"mean": None, "std": None}
                continue
            m = sum(vals) / len(vals)
            sd = None
            if len(vals) > 1:
                sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            stats[k] = {"mean": m, "std": sd}
        agg[name] = stats
    return agg


def _exp_lora_ablation(cfg: dict) -> dict:
    """Ablation with a TRAINABLE vision tower (LoRA), proposal 11.13 Stage 1.

    The frozen-feature ablation could not show any privacy benefit from the gate
    — a frozen backbone can only be reweighted, not reshaped. This variant makes
    the vision tower trainable so the gate/adversary can actually alter the
    representation, and measures identity leakage against the same held-out
    attack.

    MUST run several seeds. Two single-seed runs of an identical config produced
    leakage reductions that disagreed in *sign* (`full`: +0.0224 then -0.0163),
    because bf16 GPU kernels are non-deterministic and 12 epochs amplify it. The
    per-seed spread is the same size as the effect, so a single seed measures
    nothing here — only mean +/- std across seeds is reportable.
    """
    try:
        from .models import VLMBackbone, VLMBackboneConfig
        from .models.train import fit_with_lora, leakage_from_reps
    except ImportError:
        return {"experiment": "lora_ablation", "skipped": True,
                "reason": 'needs torch + peft: pip install -e ".[model]" peft'}
    import gc

    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    seeds = cfg.get("seeds", [0, 1, 2])
    rows = []
    for name in cfg.get("variants", ["full", "no_gate"]):
        mcfg, lw = _LORA_VARIANTS.get(name, ({}, {}))
        for seed in seeds:
            bk = VLMBackbone(VLMBackboneConfig(
                family=cfg.get("backbone", "paddleocr_vl"),
                device=cfg.get("device", "cuda"),
                dtype=cfg.get("dtype", "bfloat16"),
                lora=cfg.get("lora", True), lora_r=cfg.get("lora_r", 8),
                lora_alpha=cfg.get("lora_alpha", 16)))
            try:
                bk.load()
                _m, res = fit_with_lora(
                    train, val, bk, cfg["data_dir"],
                    epochs=cfg.get("train_epochs", 12),
                    lr=cfg.get("lr", 3e-3), lora_lr=cfg.get("lora_lr", 1e-4),
                    max_docs=cfg.get("train_max_docs", 24),
                    eval_max_docs=cfg.get("eval_max_docs", 120),
                    seed=seed, model_cfg=mcfg, loss_weights=lw)
                zs, zr, yy, pl = res["eval"]
                # Same seed for the attacker, so attacker init is not a second
                # uncontrolled source of variance on top of training.
                lk = leakage_from_reps(zs, zr, yy, pl, seed=seed)
                rows.append({
                    "variant": name, "seed": seed,
                    "loss_end": res["loss_history"][-1],
                    "identity_leak_safe": lk["safe"]["identity_leakage"],
                    "identity_leak_raw_unprotected": lk["raw_unprotected"]["identity_leakage"],
                    "type_baseline_cosine": lk["type_baseline_cosine"],
                    "n_pii_regions": lk["n_pii_regions"],
                    # Utility side of the same trade-off, same forward pass.
                    "clinical_macro_f1_safe": res["val_clinical"]["macro_f1"],
                    "pii_average_precision": res["val_pii"]["average_precision"],
                    "pii_f1": res["val_pii"]["f1"],
                    "lora_trainable_params": getattr(bk, "lora_trainable", None),
                })
            except Exception as e:
                rows.append({"variant": name, "seed": seed,
                             "error": f"{type(e).__name__}: {str(e)[:300]}"})
            finally:
                del bk
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            print(f"ROW {json.dumps(rows[-1])}", flush=True)
    # Only the WITHIN-row gap is interpretable. LoRA training changes the
    # representation itself, so each variant ends up with its own raw baseline
    # and comparing safe values across variants would be comparing different
    # reference points.
    for r in rows:
        if "identity_leak_safe" in r:
            r["leak_reduction_vs_own_raw"] = (
                r["identity_leak_raw_unprotected"] - r["identity_leak_safe"])
    return {"experiment": "lora_ablation", "results": rows,
            "summary": _agg_by_variant(
                rows, ["leak_reduction_vs_own_raw", "identity_leak_safe",
                       "clinical_macro_f1_safe", "pii_average_precision"]),
            "note": ("identity_leak = held-out attacker cosine minus the PII-type "
                     "baseline. Interpret leak_reduction_vs_own_raw (within-row); "
                     "raw baselines are NOT comparable across variants because "
                     "LoRA retrains the representation for each one. Compare "
                     "variants only via `summary` mean +/- std: single-seed runs "
                     "of this config have disagreed in sign.")}


def _exp_scaffolded(name: str, cfg: dict) -> dict:
    raise NotImplementedError(
        f"'{name}' needs the MACULAR model core (macular/models) implemented and "
        f"a GPU. See proposal section 11.x. This harness ships the interfaces "
        f"and configs, not the trained forward pass."
    )


_DISPATCH = {
    "data_gen": _exp_data_gen,
    "shortcut_audit": _exp_shortcut_audit,
    "ocr_baseline": _exp_ocr_baseline,
    "data_stats": _exp_data_stats,
    "fetch_funsd": _exp_fetch_funsd,
    "fetch_xfund": _exp_fetch_xfund,
    "train_core": _exp_train_core,
    "ocr_propagation": _exp_ocr_propagation,
    "engine_downstream": _exp_engine_downstream,
    "ocr_cache": _exp_ocr_cache,
    "backbone_gate": _exp_backbone_gate,
    "ablation": _exp_ablation,
    "lora_ablation": _exp_lora_ablation,
    "erasure_comparison": _exp_erasure_comparison,
}


def run_experiment(name: str, cfg: dict) -> dict:
    if name in _DISPATCH:
        return _DISPATCH[name](cfg)
    if name in SCAFFOLDED:
        return _exp_scaffolded(name, cfg)
    raise ValueError(f"unknown experiment '{name}'. Try: {RUNNABLE + SCAFFOLDED}")


def _write_result(out_dir: str, name: str, result: dict, suffix: str = "") -> str:
    """Write results/<name><suffix>.json.

    ``suffix`` exists because the same experiment run against a different
    backbone or config otherwise silently overwrites the previous run's results —
    which nearly cost a completed hour-long comparison.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


# --- CLI -------------------------------------------------------------------

def main(argv=None) -> int:
    # Windows consoles default to a legacy codepage (e.g. cp949) that can't
    # encode non-ASCII in results. Force UTF-8 so printing never crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(prog="macular")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="print capability report")
    sub.add_parser("list", help="list experiments")

    rp = sub.add_parser("run", help="run an experiment")
    rp.add_argument("experiment")
    rp.add_argument("--config", default=None)
    rp.add_argument("--out", default=DEFAULT_OUT)

    args = p.parse_args(argv)

    if args.command == "probe":
        report = probe_env()
        print(json.dumps(report, indent=2))
        os.makedirs(DEFAULT_OUT, exist_ok=True)
        with open(os.path.join(DEFAULT_OUT, "env_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return 0

    if args.command == "list":
        print("Runnable today:")
        for n in RUNNABLE:
            print(f"  {n}")
        print("Scaffolded (needs model core + GPU):")
        for n in SCAFFOLDED:
            print(f"  {n}")
        return 0

    if args.command == "run":
        cfg = load_config(args.config)
        try:
            result = run_experiment(args.experiment, cfg)
        except NotImplementedError as e:
            print(f"[scaffolded] {e}")
            return 2
        path = _write_result(args.out, args.experiment, result,
                             suffix=cfg.get("result_suffix", ""))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n-> wrote {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
