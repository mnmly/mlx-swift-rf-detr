#!/usr/bin/env python3
"""Generate numerical-parity fixtures for the RF-DETR keypoint-preview model.

Mirrors ../generate_fixtures.py but targets the GroupPose keypoint model and uses the
current (1.8.1) ``build_model_from_config`` API. Person-only checkpoint => num_classes=1.

Run from the python rf-detr repo so its package imports:

    cd ../../python/rf-detr
    .venv/bin/python <swift-repo>/Tests/fixtures/keypoint/generate_keypoint_fixtures.py

Outputs (gitignored *.safetensors; regenerate locally) into this directory:
  - weights.safetensors        : model.-prefixed fp32 weights for Swift WeightLoader
  - input.safetensors          : deterministic NHWC pixel_values
  - outputs.safetensors        : pred_logits / pred_boxes / pred_keypoints (authoritative)
  - intermediates.safetensors  : hook-captured submodule outputs (debugging aid)
  - postprocess.safetensors    : final scores/labels/boxes/keypoints/cholesky (image 0)
  - stats.json                 : per-tensor (shape, mean, std, abs_max) sanity stats
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

PYTHON_PROJECT = Path("/Users/mnmly/Development-local/GitHub/python/rf-detr")
FIXTURE_DIR = Path(__file__).resolve().parent
CKPT_PATH = Path(os.path.expanduser("~/.cache/rfdetr/rf-detr-keypoint-preview-xlarge.pth"))
DEAD_KEY_PREFIXES = ("keypoint_head.keypoint_proj.",)

sys.path.insert(0, str(PYTHON_PROJECT / "src"))


def set_determinism(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_model():
    from rfdetr.config import RFDETRKeypointPreviewConfig
    from rfdetr.models.lwdetr import build_model_from_config

    cfg = RFDETRKeypointPreviewConfig(num_classes=1)
    model = build_model_from_config(cfg)

    sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith(DEAD_KEY_PREFIXES)}
    incompatible = model.load_state_dict(sd, strict=False)
    missing = [k for k in incompatible.missing_keys if "criterion" not in k]
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if incompatible.unexpected_keys:
        print(f"  WARNING: {len(incompatible.unexpected_keys)} unexpected: "
              f"{incompatible.unexpected_keys[:5]}")
    if not missing and not incompatible.unexpected_keys:
        print("  state_dict loaded cleanly (0 missing / 0 unexpected)")
    model.eval()
    return model, cfg.model_dump(), sd


def weights_for_swift(state_dict: dict) -> dict:
    """model.-prefixed fp32; conv weights stay NCHW (Swift transposes to NHWC)."""
    return {f"model.{k}": v.detach().cpu().to(torch.float32) for k, v in state_dict.items()}


# Submodules whose forward outputs are worth capturing for staged parity debugging.
HOOK_TARGETS = [
    "backbone.0.projector",
    "backbone.0.cross_attn_projector",
    "transformer",
    "transformer.decoder",
    "transformer.keypoint_query_initializer",
    "transformer.keypoint_query_initializer_enc",
    "class_embed",
    "bbox_embed",
    "keypoint_embed",
]


def _flatten_tensors(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, torch.Tensor):
        # clone(): hook outputs frequently alias one another (tuple elements reused
        # across submodules); safetensors rejects shared-memory tensors.
        out[prefix] = obj.detach().cpu().to(torch.float32).contiguous().clone()
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten_tensors(f"{prefix}.{i}", v, out)


def register_hooks(model, store: dict):
    handles = []
    name_to_mod = dict(model.named_modules())
    for name in HOOK_TARGETS:
        mod = name_to_mod.get(name)
        if mod is None:
            print(f"  (hook skip: {name} not found)")
            continue

        def make_hook(nm):
            def hook(_m, _inp, output):
                _flatten_tensors(nm.replace(".", "_"), output, store)
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))
    # Per decoder layer.
    decoder = name_to_mod.get("transformer.decoder")
    if decoder is not None and hasattr(decoder, "layers"):
        for i, layer in enumerate(decoder.layers):
            def make_hook(idx):
                def hook(_m, _inp, output):
                    _flatten_tensors(f"dec_layer_{idx}", output, store)
                return hook
            handles.append(layer.register_forward_hook(make_hook(i)))
    return handles


def save_tensors(path: Path, tensors: dict) -> None:
    payload = {}
    for k, v in tensors.items():
        payload[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        payload[k] = payload[k].detach().cpu().contiguous().to(torch.float32)
    st.save_file(payload, str(path))


def tensor_stats(tensors: dict) -> dict:
    out = {}
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach().cpu().to(torch.float32)
        out[k] = {
            "shape": list(t.shape),
            "mean": float(t.mean()),
            "std": float(t.std()) if t.numel() > 1 else 0.0,
            "abs_max": float(t.abs().max()),
        }
    return out


def main() -> None:
    set_determinism(42)
    print("Building keypoint-preview model (num_classes=1)...")
    model, cfg, state_dict = build_model()
    res = cfg["resolution"]
    print(f"  resolution={res}, num_queries={cfg['num_queries']}, "
          f"num_keypoints_per_class={cfg['num_keypoints_per_class']}")

    print("Saving Swift weights...")
    save_tensors(FIXTURE_DIR / "weights.safetensors", weights_for_swift(state_dict))

    set_determinism(42)
    # In-distribution-ish input: a smooth low-frequency pattern normalized by the
    # ImageNet stats the model expects. Unnormalized high-frequency noise drives the
    # attention softmaxes into a chaotic regime where tiny FP differences explode,
    # masking real parity; a smooth normalized input keeps predictions stable so the
    # comparison actually measures correctness.
    low = torch.rand(1, 3, res // 32, res // 32, dtype=torch.float32)
    smooth = torch.nn.functional.interpolate(low, size=(res, res), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    x_nchw = (smooth - mean) / std
    save_tensors(FIXTURE_DIR / "input.safetensors",
                 {"pixel_values": x_nchw.permute(0, 2, 3, 1).contiguous()})

    print("Authoritative forward + intermediates...")
    intermediates: dict = {}
    handles = register_hooks(model, intermediates)
    with torch.no_grad():
        out = model(x_nchw)
    for h in handles:
        h.remove()

    auth = {
        "pred_logits": out["pred_logits"],
        "pred_boxes": out["pred_boxes"],
        "pred_keypoints": out["pred_keypoints"],
    }
    save_tensors(FIXTURE_DIR / "outputs.safetensors", auth)
    save_tensors(FIXTURE_DIR / "intermediates.safetensors", intermediates)

    print("Postprocess (full output: scores/labels/boxes/keypoints/cholesky)...")
    try:
        from rfdetr.models.postprocess import PostProcess

        pp = PostProcess(num_select=cfg["num_select"],
                         num_keypoints_per_class=cfg["num_keypoints_per_class"],
                         trace_alpha=0.2)
        target_sizes = torch.tensor([[res, res]])
        with torch.no_grad():
            results = pp({k: v for k, v in out.items()
                          if isinstance(v, torch.Tensor)}, target_sizes)
        r0 = results[0]
        pp_out = {f"pp_{k}": v for k, v in r0.items() if isinstance(v, torch.Tensor)}
        save_tensors(FIXTURE_DIR / "postprocess.safetensors", pp_out)
        print(f"  postprocess keys: {list(r0.keys())}")
    except Exception as e:  # noqa: BLE001 — diagnostic, keep fixtures partial-usable
        print(f"  WARNING: postprocess dump failed: {e!r}")

    stats = {
        "outputs": tensor_stats(auth),
        "intermediates": tensor_stats(intermediates),
        "config": {k: cfg[k] for k in
                   ("hidden_dim", "resolution", "num_classes", "num_queries",
                    "num_keypoints_per_class", "bbox_reparam")},
    }
    with open(FIXTURE_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Summary ===")
    for k, v in auth.items():
        print(f"  {k}: {list(v.shape)}  mean={v.mean():.6f}  abs_max={v.abs().max():.6f}")
    print(f"  intermediates captured: {len(intermediates)} tensors")
    print(f"Done -> {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
