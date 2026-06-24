#!/usr/bin/env python3
"""Convert the RF-DETR keypoint-preview checkpoint to an mlx-swift model directory.

The keypoint-preview weights are not on HuggingFace; they live on GCS alongside the
other rf-detr checkpoints. This script reuses a locally cached ``.pth`` (downloading it
if missing), drops dead keys, and emits the three-file layout the Swift loader expects:

    <out>/config.json
    <out>/preprocessor_config.json
    <out>/model.safetensors

Run from the python rf-detr repo so its package is importable (config values are read
from ``RFDETRKeypointPreviewConfig`` rather than hardcoded):

    cd ../../python/rf-detr
    .venv/bin/python /path/to/mlx-swift-rf-detr/Scripts/convert_keypoint.py

Notes:
  * This checkpoint is a **person-only** model: ``class_embed`` is 2-dim, so we build /
    convert with ``num_classes=1``.
  * 4 dead keys (``keypoint_head.keypoint_proj.*``) are present in the checkpoint but
    consumed by no module; they are dropped so Swift's strict load stays clean.
  * Conv weights are kept NCHW; Swift's WeightLoader transposes them to NHWC.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

CKPT_URL = "https://storage.googleapis.com/rfdetr/rf-detr-keypoint-preview-xlarge.pth"
CACHE_DIR = Path(os.path.expanduser("~/.cache/rfdetr"))
PTH_PATH = CACHE_DIR / "rf-detr-keypoint-preview-xlarge.pth"
DEFAULT_OUT = CACHE_DIR / "rfdetr-keypoint-preview-mlx"

# Keys present in the checkpoint that no current module consumes (verified via
# load_state_dict(strict=False) -> unexpected_keys). Dropped to keep Swift loading clean.
DEAD_KEY_PREFIXES = ("keypoint_head.keypoint_proj.",)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  using cached {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} ...")
    urlretrieve(url, str(dest))
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def build_config() -> dict:
    """Authoritative config.json, read from the rf-detr config object."""
    from rfdetr.config import RFDETRKeypointPreviewConfig

    c = RFDETRKeypointPreviewConfig(num_classes=1).model_dump()
    return {
        "model_type": "rf-detr",
        "encoder": c["encoder"],
        "hidden_dim": c["hidden_dim"],
        "resolution": c["resolution"],
        "dec_layers": c["dec_layers"],
        "num_queries": c["num_queries"],
        "num_classes": c["num_classes"],
        "patch_size": c["patch_size"],
        "num_windows": c["num_windows"],
        "group_detr": c["group_detr"],
        "sa_nheads": c["sa_nheads"],
        "ca_nheads": c["ca_nheads"],
        "dec_n_points": c["dec_n_points"],
        "two_stage": c["two_stage"],
        "bbox_reparam": c["bbox_reparam"],
        "lite_refpoint_refine": c["lite_refpoint_refine"],
        "layer_norm": c["layer_norm"],
        "out_feature_indexes": c["out_feature_indexes"],
        "projector_scale": c["projector_scale"],
        "positional_encoding_size": c["positional_encoding_size"],
        # --- keypoint (GroupPose) fields ---
        "use_grouppose_keypoints": c["use_grouppose_keypoints"],
        "dual_projector": c["dual_projector"],
        "dual_projector_kp_only": c["dual_projector_kp_only"],
        "num_keypoints_per_class": c["num_keypoints_per_class"],
        "keypoint_cross_attn": c["keypoint_cross_attn"],
        "inter_instance_kp_attn": c["inter_instance_kp_attn"],
        "grouppose_keypoint_dim_downscale": c["grouppose_keypoint_dim_downscale"],
        "trace_alpha": 0.2,
    }


def main() -> None:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        sys.exit("torch and safetensors are required (use the rf-detr .venv).")

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: checkpoint")
    _download(CKPT_URL, PTH_PATH)

    print("Step 2: extract + clean weights")
    ckpt = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    dropped = [k for k in state_dict if k.startswith(DEAD_KEY_PREFIXES)]
    prefixed = {
        f"model.{k}": v.cpu().to(torch.float32)
        for k, v in state_dict.items()
        if not k.startswith(DEAD_KEY_PREFIXES)
    }
    print(f"  {len(prefixed)} tensors kept, {len(dropped)} dead keys dropped: {dropped}")

    print("Step 3: write directory")
    config = build_config()
    save_file(prefixed, str(out_dir / "model.safetensors"))
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    preprocessor = {
        "config": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "post_process_config": {"num_select": config["num_queries"]},
    }
    with open(out_dir / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor, f, indent=2)

    sz = (out_dir / "model.safetensors").stat().st_size / 1e6
    print(f"\nDone -> {out_dir}")
    print(f"  model.safetensors ({sz:.1f} MB), config.json, preprocessor_config.json")
    print(f"  num_classes={config['num_classes']} (person-only), "
          f"num_keypoints_per_class={config['num_keypoints_per_class']}")


if __name__ == "__main__":
    main()
