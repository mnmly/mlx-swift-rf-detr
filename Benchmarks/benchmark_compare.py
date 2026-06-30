#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import safetensors.torch as st
import torch


ROOT = Path(__file__).resolve().parent           # Benchmarks/
REPO_ROOT = ROOT.parent                           # mlx-swift-rf-detr/
PYTHON_REPO = (REPO_ROOT / "../../python/rf-detr").resolve()
FIXTURES_DIR = REPO_ROOT / "Tests" / "fixtures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Swift MLX RF-DETR against Python RF-DETR.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--swift-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--python-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--python-device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--skip-swift", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def torch_dtype(name: str) -> "torch.dtype":
    return {"float32": torch.float32, "float16": torch.float16}[name]


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "stddev": statistics.pstdev(values),
    }


def run_swift_benchmark(iterations: int, warmup: int, dtype: str) -> dict[str, Any]:
    derived_data = REPO_ROOT / ".xcode-bench"
    build_cmd = [
        "xcodebuild",
        "build",
        "-scheme",
        "RFDETRBench",
        "-destination",
        "platform=macOS",
        "-derivedDataPath",
        str(derived_data),
        "-quiet",
    ]
    # This package depends on mlx-swift (Metal/C++). A swiftly `TOOLCHAINS` shim or a
    # beta `DEVELOPER_DIR` selects an incompatible toolchain and the build fails — pin
    # the stable Xcode toolchain (see CLAUDE.md). Override XCODE_DEVELOPER_DIR if Xcode
    # lives elsewhere.
    build_env = {k: v for k, v in os.environ.items() if k != "TOOLCHAINS"}
    build_env["DEVELOPER_DIR"] = os.environ.get(
        "XCODE_DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer"
    )
    subprocess.run(
        build_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True, env=build_env
    )

    executable = derived_data / "Build" / "Products" / "Debug" / "RFDETRBench"
    cmd = [
        str(executable),
        "--fixtures",
        str(FIXTURES_DIR),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--dtype",
        dtype,
        "--label",
        f"swift-mlx-small-{dtype}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=executable.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def build_python_model(device: str, dtype: "torch.dtype") -> Any:
    sys.path.insert(0, str(PYTHON_REPO / "src"))
    from rfdetr.config import RFDETRSmallConfig
    from rfdetr.models import build_model_from_config

    # rf-detr >=1.8.2 removed `rfdetr.main.populate_args` and the
    # `build_model(populate_args(**cfg))` flow in favour of the config-native
    # `build_model_from_config(model_config)`.
    cfg = RFDETRSmallConfig(pretrain_weights=str(PYTHON_REPO / "rf-detr-small.pth"))
    model = build_model_from_config(cfg)
    checkpoint = torch.load(str(PYTHON_REPO / "rf-detr-small.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval().to(device).to(dtype)
    return model


def load_fixture_sample(device: str, dtype: "torch.dtype") -> Any:
    sys.path.insert(0, str(PYTHON_REPO / "src"))
    from rfdetr.utilities import NestedTensor  # rfdetr.util.misc deprecated (removed in v1.9)

    tensors = st.load_file(str(FIXTURES_DIR / "input.safetensors"))
    pixel_values = tensors["pixel_values"].to(dtype)
    nchw = pixel_values.permute(0, 3, 1, 2).contiguous().to(device)
    mask = torch.zeros((nchw.shape[0], nchw.shape[2], nchw.shape[3]), dtype=torch.bool, device=device)
    return NestedTensor(nchw, mask)


def run_python_benchmark(iterations: int, warmup: int, device: str, dtype_name: str) -> dict[str, Any]:
    dtype = torch_dtype(dtype_name)
    model = build_python_model(device, dtype)
    samples = load_fixture_sample(device, dtype)

    total_samples: list[float] = []
    stage_samples: dict[str, list[float]] = {
        "backbone": [],
        "projector": [],
        "transformer": [],
        "heads": [],
    }

    query_w = model.query_feat.weight[: model.num_queries]
    refpoint_w = model.refpoint_embed.weight[: model.num_queries]

    with torch.no_grad():
        for step in range(warmup + iterations):
            current: dict[str, float] = {}
            total_start = time.perf_counter()

            start = time.perf_counter()
            raw_features = model.backbone[0].encoder(samples.tensors)
            synchronize(device)
            current["backbone"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            _ = model.backbone[0].projector(raw_features)
            synchronize(device)
            current["projector"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            # rf-detr >=1.8.2: backbone returns (features, pos, cross_attn_features) and
            # the transformer takes cross_attn_srcs (None for plain detection).
            features, poss, cross_attn_features = model.backbone(samples)
            decomp_srcs = []
            masks = []
            for feat in features:
                src, mask = feat.decompose()
                decomp_srcs.append(src)
                masks.append(mask)
            cross_attn_srcs = (
                [f.decompose()[0] for f in cross_attn_features]
                if cross_attn_features is not None else None
            )
            transformer_outputs = model.transformer(
                decomp_srcs, masks, poss, refpoint_w, query_w, cross_attn_srcs=cross_attn_srcs
            )
            hs, ref_unsigmoid, _, _ = transformer_outputs[:4]
            synchronize(device)
            current["transformer"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            if model.bbox_reparam:
                delta = model.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                pred_boxes = torch.cat([cxcy, wh], dim=-1)
            else:
                pred_boxes = (model.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            pred_logits = model.class_embed(hs)
            _ = pred_boxes[-1], pred_logits[-1]
            synchronize(device)
            current["heads"] = (time.perf_counter() - start) * 1000.0

            total = (time.perf_counter() - total_start) * 1000.0

            if step >= warmup:
                for key, value in current.items():
                    stage_samples[key].append(value)
                total_samples.append(total)

    return {
        "label": f"python-pytorch-small-{device}-{dtype_name}",
        "iterations": iterations,
        "warmup": warmup,
        "device": device,
        "dtype": dtype_name,
        "shape": list(samples.tensors.shape),
        "stageStatsMs": {key: summarize(values) for key, values in stage_samples.items()},
        "totalStatsMs": summarize(total_samples),
    }


def relative_ratio(swift_ms: float, python_ms: float) -> float:
    if python_ms == 0:
        return math.inf
    return swift_ms / python_ms


def choose_python_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_report(payload: dict[str, Any]) -> None:
    swift = payload.get("swift")
    python = payload.get("python")

    if swift:
        print("Swift benchmark")
        print(f"  label: {swift['label']}")
        print(f"  total mean:   {swift['totalStatsMs']['mean']:.2f} ms")
        print(f"  total median: {swift['totalStatsMs']['median']:.2f} ms")
        for stage in ("backbone", "projector", "transformer", "heads"):
            stats = swift["stageStatsMs"][stage]
            print(f"  {stage:11s} {stats['mean']:.2f} ms mean")

    if python:
        print("Python benchmark")
        print(f"  label: {python['label']}")
        print(f"  total mean:   {python['totalStatsMs']['mean']:.2f} ms")
        print(f"  total median: {python['totalStatsMs']['median']:.2f} ms")
        for stage in ("backbone", "projector", "transformer", "heads"):
            stats = python["stageStatsMs"][stage]
            print(f"  {stage:11s} {stats['mean']:.2f} ms mean")

    if swift and python:
        ratio = relative_ratio(swift["totalStatsMs"]["mean"], python["totalStatsMs"]["mean"])
        print("Comparison")
        print(f"  swift/python total mean ratio: {ratio:.2f}x")
        slower = [stage for stage in ("backbone", "projector", "transformer", "heads")
                  if relative_ratio(swift["stageStatsMs"][stage]["mean"], python["stageStatsMs"][stage]["mean"]) > 1.5]
        if slower:
            print(f"  stages where Swift is >1.5x slower: {', '.join(slower)}")
        else:
            print("  no stage exceeds the 1.5x slower threshold")


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {"cwd": str(ROOT)}

    if not args.skip_swift:
        payload["swift"] = run_swift_benchmark(args.iterations, args.warmup, args.swift_dtype)

    if not args.skip_python:
        payload["python"] = run_python_benchmark(
            args.iterations,
            args.warmup,
            choose_python_device(args.python_device),
            args.python_dtype,
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_report(payload)
        print()
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
