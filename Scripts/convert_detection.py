#!/usr/bin/env python3
"""Convert an RF-DETR detection checkpoint to an mlx-swift model directory.

The detection weights live on the same GCS bucket as the keypoint checkpoint
(``gs://rfdetr``). This is the detection counterpart of ``convert_keypoint.py``:
it fetches a variant's ``.pth`` (caching it under ``~/.cache/rfdetr``) and emits
the three-file layout the Swift loader expects:

    <out>/config.json
    <out>/preprocessor_config.json
    <out>/model.safetensors

Run from the python rf-detr repo so its package is importable (config values are
read from the variant's config class rather than hardcoded):

    cd ../../python/rf-detr
    .venv/bin/python /path/to/mlx-swift-rf-detr/Scripts/convert_detection.py small

Usage:
    convert_detection.py <variant> [out_dir]

    <variant> ∈ {base, small, large, large-2026} — the detection variants
    recognized by the Swift port's ``RFDETRVariant.detect``
    (Sources/MLXRFDETR/Variant.swift). out_dir defaults to
    ``~/.cache/rfdetr/rfdetr-<variant>-mlx``.

Notes:
  * Conv weights are kept NCHW; Swift's WeightLoader transposes them to NHWC.
  * Out of scope here (handled elsewhere / not in the Swift variant table):
      - nano   : https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth
      - medium : https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth
      - seg-*  : .pt checkpoints with a mask head (see README → mlx-vlm converter)
    Add a row to VARIANTS below if/when the Swift port gains support.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

CACHE_DIR = Path(os.path.expanduser("~/.cache/rfdetr"))

# variant -> (config class name, download URL, local checkpoint filename).
# URLs/filenames are the authoritative rf-detr 1.8.1 registry
# (src/rfdetr/assets/model_weights.py). The local filename is canonical because
# several upstream URLs share the basename "checkpoint_best_regular.pth".
VARIANTS = {
    "base": (
        "RFDETRBaseConfig",
        "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "rf-detr-base.pth",
    ),
    "small": (
        "RFDETRSmallConfig",
        "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "rf-detr-small.pth",
    ),
    # Swift's `large` variant (res 560, dec 3, hidden 384) is the *pre-2026*
    # checkpoint (RFDETRLargeDeprecatedConfig). `large-2026` is the current large
    # — same small backbone / hidden 256 as `small`, just res 704 + dec 4.
    "large": (
        "RFDETRLargeDeprecatedConfig",
        "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        "rf-detr-large.pth",
    ),
    "large-2026": (
        "RFDETRLargeConfig",
        "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "rf-detr-large-2026.pth",
    ),
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  using cached {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} ...")
    urlretrieve(url, str(dest))
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def build_config(config_class_name: str) -> dict:
    """Authoritative config.json, read from the rf-detr config object."""
    from rfdetr import config as rfdetr_config

    cls = getattr(rfdetr_config, config_class_name)
    c = cls().model_dump()
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
    }


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
        sys.exit(f"usage: convert_detection.py <{'|'.join(VARIANTS)}> [out_dir]")

    variant = sys.argv[1]
    config_class_name, url, ckpt_name = VARIANTS[variant]
    pth_path = CACHE_DIR / ckpt_name
    out_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else CACHE_DIR / f"rfdetr-{variant}-mlx"
    )

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        sys.exit("torch and safetensors are required (use the rf-detr .venv).")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: checkpoint ({variant})")
    _download(url, pth_path)

    print("Step 2: extract weights")
    ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    prefixed = {
        f"model.{k}": v.cpu().to(torch.float32) for k, v in state_dict.items()
    }
    print(f"  {len(prefixed)} tensors kept")

    print("Step 3: write directory")
    config = build_config(config_class_name)
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
    print(f"  encoder={config['encoder']}, resolution={config['resolution']}, "
          f"num_classes={config['num_classes']}, num_queries={config['num_queries']}")


if __name__ == "__main__":
    main()
