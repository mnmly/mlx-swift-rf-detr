#!/usr/bin/env python3
"""
Generate numerical parity test fixtures for RF-DETR Small.

Run from the python rf-detr project (so its src/ is importable). Use the repo's
`.venv/bin/python`, NOT `uv run` — `uv run` tries to build the `trt` extra (tensorrt)
and fails on macOS:
    cd ../../python/rf-detr
    .venv/bin/python /Users/mnmly/Development-local/GitHub/swift/mlx-swift-rf-detr/Tests/fixtures/generate_fixtures.py

NOTE: the committed fixtures were generated against rf-detr 1.8.1. The model *build*
path here is updated for rf-detr >=1.8.2 (`build_model_from_config`), but the diagnostic
forward below still assumes the 1.8.1 internals (backbone returns 2 values; `NestedTensor`
in `rfdetr.util.misc`; transformer without `cross_attn_srcs`). To regenerate against a
newer rf-detr, apply the same updates used in `Benchmarks/benchmark_compare.py` — and note
the resulting fixtures will differ from the committed ones (which the parity tests pin to).

Outputs (saved to Tests/fixtures/):
  - weights.safetensors       : model weights in format Swift WeightLoader expects
  - input.safetensors         : test input tensor (deterministic, NHWC)
  - outputs.safetensors       : authoritative pred_logits / pred_boxes from model(samples)
  - intermediates.safetensors : diagnostic intermediate tensors (per-layer)
  - stats.json                : per-tensor (mean, std, abs_max) for quick sanity check
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F

PYTHON_PROJECT = Path("/Users/mnmly/Development-local/GitHub/python/rf-detr")
FIXTURE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PYTHON_PROJECT / "src"))

# Swift WeightLoader.sanitized() conv-detection keywords. Must stay in sync.
# Note: Swift applies remapBackbone() BEFORE the keyword check, so
# `patch_embeddings.projection` (HF form) is renamed to `patch_embed.proj`
# (Swift form) before matching. We list both forms so this script's pre-remap
# check matches Swift's post-remap behavior.
SWIFT_CONV_KEYWORDS = [
    "conv",
    "spatial_features_proj",
    "patch_embed.proj",          # Swift form (post-remap)
    "patch_embeddings.projection",  # HF form (pre-remap)
]


def set_determinism(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_with_config():
    """Build RF-DETR Small using the official config + build path."""
    from rfdetr.config import RFDETRSmallConfig
    from rfdetr.models import build_model_from_config

    # rf-detr >=1.8.2 removed `rfdetr.main.populate_args`; build straight from the config.
    cfg = RFDETRSmallConfig()
    cfg_dict = cfg.model_dump()
    model = build_model_from_config(cfg)

    checkpoint_path = PYTHON_PROJECT / "rf-detr-small.pth"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    # Strict-with-reporting: surface any rename drift loudly.
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if "criterion" not in k]
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  state_dict loaded cleanly (no missing / unexpected keys)")

    model.eval()
    return model, cfg_dict, state_dict


def convert_weights_for_swift(state_dict):
    """
    Convert PyTorch state dict for Swift WeightLoader.

    Swift's sanitized() expects 'model.' prefix and Conv2d weights in NCHW
    (which it transposes to NHWC). We keep NCHW here and trust Swift to transpose.
    """
    converted = {}
    suspicious_4d = []  # 4D tensors not matched by Swift's keyword list

    for key, value in state_dict.items():
        new_key = "model." + key
        if value.ndim == 4:
            key_lower = key.lower()
            if not any(kw in key_lower for kw in SWIFT_CONV_KEYWORDS):
                suspicious_4d.append((key, tuple(value.shape)))
        converted[new_key] = value.cpu().to(torch.float32)

    if suspicious_4d:
        print(f"  WARNING: {len(suspicious_4d)} 4D tensors won't be transposed by Swift "
              f"(no keyword match). First 5: {suspicious_4d[:5]}")
        print("  → If any of these are conv weights, update SWIFT_CONV_KEYWORDS and "
              "WeightLoader.swift in lockstep.")

    return converted


def run_authoritative_forward(model, x_nchw):
    """Call model(samples) — this is the parity oracle."""
    from rfdetr.util.misc import NestedTensor

    bs, _, h, w = x_nchw.shape
    mask = torch.zeros(bs, h, w, dtype=torch.bool)
    samples = NestedTensor(x_nchw, mask)

    with torch.no_grad():
        out = model(samples)

    return {
        "pred_logits": out["pred_logits"],
        "pred_boxes": out["pred_boxes"],
    }


def run_diagnostic_forward(model, x_nchw):
    """
    Re-run forward, capturing intermediates at meaningful boundaries.
    Mirrors the real LWDETR.forward path so diagnostic outputs == authoritative.
    """
    from rfdetr.util.misc import NestedTensor

    intermediates = {}

    with torch.no_grad():
        bs, _, H, W = x_nchw.shape
        mask = torch.zeros(bs, H, W, dtype=torch.bool)
        samples = NestedTensor(x_nchw, mask)

        # --- Sub-stage parity probes (localize bugs inside the backbone) ---
        # `model.backbone` is a Joiner; `model.backbone[0]` is the Backbone wrapper;
        # `model.backbone[0].encoder` is DinoV2; `.encoder.encoder` is the HF model.
        hf_model = model.backbone[0].encoder.encoder
        embeddings_mod = hf_model.embeddings
        encoder_mod = hf_model.encoder

        # (a) Patch-embed conv output BEFORE flattening.
        #     Dinov2WithRegistersPatchEmbeddings.forward flattens to (B, num_patches, C),
        #     so we manually invoke the underlying Conv2d to keep spatial dims.
        pe_proj = embeddings_mod.patch_embeddings.projection
        patch_conv = pe_proj(x_nchw)  # (B, C, h, w)
        intermediates["patch_embed_conv"] = patch_conv.permute(0, 2, 3, 1).contiguous()  # NHWC

        # (b) Full embeddings module output: post-conv + cls + pos + windowing
        #     (with registers if num_register_tokens > 0).
        emb_out = embeddings_mod(x_nchw)  # (B*nW², 1+R+win_h*win_w, C)
        intermediates["embeddings_out"] = emb_out

        # (c) Per-block outputs through layer 2 (where fb_0 is read).
        cur = emb_out
        wbi = set(hf_model.config.window_block_indexes)
        for i in range(3):  # blocks 0, 1, 2
            run_full = i not in wbi
            cur = encoder_mod.layer[i](cur, None, False, run_full)[0]
            intermediates[f"block_{i}_out"] = cur

        # (d) Sub-stages of block 2 (the first full-attention block).
        #     Drives down from "after block 1" to find which sub-op explodes.
        block2 = encoder_mod.layer[2]
        nW = hf_model.config.num_windows
        nW2 = nW * nW
        x_in = intermediates["block_1_out"]  # (4, 257, 384)
        shortcut = x_in
        # Merge windows for full-attn
        Bx, HWx, Cx = x_in.shape
        merged = x_in.reshape(Bx // nW2, nW2 * HWx, Cx).contiguous()  # (1, 1028, 384)
        intermediates["b2_merged"] = merged.clone()
        # norm1
        n1 = block2.norm1(merged)
        intermediates["b2_norm1"] = n1
        # attention (q/k/v + sdpa + out proj)
        attn_out_merged = block2.attention(n1, None, False)[0]  # (1, 1028, 384)
        intermediates["b2_attn_merged"] = attn_out_merged
        # split back
        attn_out = attn_out_merged.reshape(Bx, HWx, Cx).contiguous()  # (4, 257, 384)
        intermediates["b2_attn_split"] = attn_out.clone()
        # layer_scale1 + residual
        ls1_out = block2.layer_scale1(attn_out)
        intermediates["b2_ls1"] = ls1_out
        post_attn = ls1_out + shortcut  # drop_path is identity at inference
        intermediates["b2_post_attn"] = post_attn
        # norm2 + mlp + layer_scale2 + residual
        n2 = block2.norm2(post_attn)
        intermediates["b2_norm2"] = n2
        mlp_out = block2.mlp(n2)
        intermediates["b2_mlp"] = mlp_out
        ls2_out = block2.layer_scale2(mlp_out)
        intermediates["b2_ls2"] = ls2_out

        # --- Raw backbone (pre-projector) ---
        # model.backbone is a Joiner; model.backbone[0] is the Backbone wrapper
        # holding .encoder (DinoV2) and .projector. Capturing encoder outputs
        # before the projector lets us localize parity bugs to backbone vs projector.
        encoder_feats = model.backbone[0].encoder(samples.tensors)
        for i, ef in enumerate(encoder_feats):
            intermediates[f"fb_{i}"] = ef.permute(0, 2, 3, 1).contiguous()  # NHWC

        # --- Backbone (post-projector, single-scale) ---
        features, poss = model.backbone(samples)
        srcs, masks = [], []
        for i, feat in enumerate(features):
            src, m = feat.decompose()
            srcs.append(src)
            masks.append(m)
            intermediates[f"fs_{i}"] = src.permute(0, 2, 3, 1).contiguous()  # NHWC

        # --- Transformer (full real path: handles iterative refinement etc.) ---
        refpoint_w = model.refpoint_embed.weight[: model.num_queries]
        query_w = model.query_feat.weight[: model.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = model.transformer(
            srcs, masks, poss, refpoint_w, query_w
        )
        # hs: (num_layers, B, nQ, D) — stacked decoder outputs after iterative refinement
        intermediates["hs_all_layers"] = hs
        intermediates["hs"] = hs[-1].clone()
        intermediates["ref_unsigmoid"] = ref_unsigmoid
        if hs_enc is not None:
            intermediates["hs_enc"] = hs_enc
        if ref_enc is not None:
            intermediates["ref_enc"] = ref_enc

        # --- Detection heads (per-layer, take last) ---
        if model.bbox_reparam:
            delta = model.bbox_embed(hs)
            cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.cat([cxcy, wh], dim=-1)
        else:
            outputs_coord = (model.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = model.class_embed(hs)

        intermediates["pred_logits_all_layers"] = outputs_class
        intermediates["pred_boxes_all_layers"] = outputs_coord
        intermediates["pred_logits"] = outputs_class[-1].clone()
        intermediates["pred_boxes"] = outputs_coord[-1].clone()

    return intermediates


def assert_diagnostic_matches_authoritative(diag, auth, atol=1e-5, rtol=1e-5):
    for k in ("pred_logits", "pred_boxes"):
        a, b = diag[k], auth[k]
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs().max().item()
            raise RuntimeError(
                f"Diagnostic forward diverged from model(samples) on {k}: "
                f"max abs diff = {diff:.3e} (atol={atol})"
            )
    print("  diagnostic ≡ authoritative ✓")


def save_tensors(filename, tensors):
    """Save dict of tensors as .safetensors (float32)."""
    payload = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            payload[k] = v.detach().cpu().contiguous().to(torch.float32)
        else:
            payload[k] = torch.tensor(v, dtype=torch.float32)
    st.save_file(payload, str(filename))


def tensor_stats(tensors):
    out = {}
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach().cpu().to(torch.float32)
        out[k] = {
            "shape": list(t.shape),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()) if t.numel() > 1 else 0.0,
            "abs_max": float(t.abs().max().item()),
            "sum": float(t.sum().item()),
        }
    return out


def main():
    set_determinism(42)

    print("Building RF-DETR Small model...")
    model, cfg, state_dict = build_model_with_config()
    print(f"  hidden_dim={cfg['hidden_dim']}, resolution={cfg['resolution']}")
    print(f"  num_classes={cfg['num_classes']}, bbox_reparam={cfg['bbox_reparam']}")

    print("\nConverting weights for Swift...")
    swift_weights = convert_weights_for_swift(state_dict)
    print(f"  {len(swift_weights)} weight tensors converted")

    # Sample conv shape sanity print
    sample_conv = next(
        (k for k in swift_weights if "patch_embed.proj.weight" in k), None
    )
    if sample_conv:
        s = tuple(swift_weights[sample_conv].shape)
        print(f"  sample conv weight {sample_conv}: NCHW={s} (Swift transposes to NHWC)")

    weight_path = FIXTURE_DIR / "weights.safetensors"
    print(f"Saving weights to {weight_path}")
    save_tensors(weight_path, swift_weights)

    # --- Test input ---
    print("\nCreating test input...")
    resolution = cfg["resolution"]
    set_determinism(42)
    x_nchw = torch.rand(1, 3, resolution, resolution, dtype=torch.float32)

    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    save_tensors(FIXTURE_DIR / "input.safetensors", {"pixel_values": x_nhwc})

    # --- Authoritative forward (parity oracle) ---
    print("\nRunning authoritative forward (model(samples))...")
    auth = run_authoritative_forward(model, x_nchw)
    save_tensors(FIXTURE_DIR / "outputs.safetensors", auth)

    # --- Diagnostic forward (per-layer intermediates) ---
    print("\nRunning diagnostic forward (capturing intermediates)...")
    diag = run_diagnostic_forward(model, x_nchw)
    assert_diagnostic_matches_authoritative(diag, auth)
    save_tensors(FIXTURE_DIR / "intermediates.safetensors", diag)

    # --- Stats ---
    print("\nComputing stats...")
    stats = {
        "outputs": tensor_stats(auth),
        "intermediates": tensor_stats(diag),
        "config": {
            "hidden_dim": cfg["hidden_dim"],
            "resolution": cfg["resolution"],
            "num_classes": cfg["num_classes"],
            "bbox_reparam": cfg["bbox_reparam"],
            "num_queries": cfg["num_queries"],
        },
    }
    with open(FIXTURE_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Reference Output Summary ===")
    for key in ("pred_logits", "pred_boxes", "hs"):
        src = auth.get(key) if key in auth else diag.get(key)
        if src is not None:
            print(f"  {key}: shape={list(src.shape)}  "
                  f"min={src.min().item():.6f}  max={src.max().item():.6f}  "
                  f"mean={src.mean().item():.6f}")

    print(f"\nDone! Fixtures in: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
