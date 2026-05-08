# Performance Notes

Per the `port-mlx-to-swift` performance gate (matched-dtype 1.5×/stage,
1.2×/total): the rf-detr Small port currently fails the per-stage gate on
the **backbone** stage at fp16. Total is within the gate (1.10×).

## Measurement (M5 Max, 128 GB, macOS 26.5, 20 iter / 5 warmup, fp16/fp16)

| Stage       | Torch / MPS (ms) | mlx-swift (ms) | Ratio (Swift / Torch) |
|-------------|------------------|----------------|-----------------------|
| backbone    | 4.91             | 10.22          | **2.08×** ⚠           |
| projector   | 1.32             | 1.81           | 1.37×                 |
| transformer | 15.75            | 7.41           | 0.47× ✅              |
| heads       | 0.42             | 0.78           | 1.85× ⚠ (sub-ms, low impact) |
| **total**   | **22.39**        | **20.21**      | 0.90× ✅              |

Reproduce: `python Benchmarks/benchmark_compare.py --iterations 20 --warmup 5
--swift-dtype float16 --python-dtype float16`.

## Diagnosis (per-block timing)

`RFDETRBench --diagnose --iterations 50 --warmup 10 --dtype float16` on
the same hardware:

```
patch_embed:  0.486 ms median
token_setup:  0.481 ms median
Per-block (median ms):
  block  0 [win ]: 1.459    block  6 [win ]: 1.482
  block  1 [win ]: 1.506    block  7 [win ]: 1.525
  block  2 [full]: 1.601    block  8 [full]: 1.587
  block  3 [win ]: 1.516    block  9 [win ]: 1.452
  block  4 [win ]: 1.477    block 10 [win ]: 1.467
  block  5 [full]: 1.588    block 11 [full]: 1.531
windowed avg per block: 1.700 ms
full     avg per block: 1.776 ms
sum of per-block medians: 18.190 ms (vs 10.22 ms lazy backbone)
```

### Findings

1. **Windowed-reshape overhead hypothesis is refuted.** Windowed (1.70 ms
   avg) vs full-attention (1.78 ms avg) per block are within ~5% of each
   other. The window-partition / unwindow ops are not the bottleneck.
2. **Per-block compute is the bottleneck.** Effective per-block cost in
   the lazy run is `10.22 / 12 ≈ 0.85 ms`. Torch reference is
   `4.91 / 12 ≈ 0.41 ms`. The ~2× per-block gap is in the actual
   computation (Linear projections, SDPA, LayerNorm, GELU, MLP), not the
   surrounding bookkeeping.
3. **Lazy evaluation is already saving ~50%.** Sum of per-block medians
   with `eval()` after each block is 18.19 ms; the lazy backbone runs in
   10.22 ms. Removing the diagnostic eval barriers recovers ~8 ms across
   the 12-block stack, so the production path is *not* leaking sync
   points. There's no easy win from "stop calling eval".
4. **Patch-embed and token-setup are sub-millisecond each** — not worth
   optimizing.

### Remaining suspects

In rough order of likelihood for the per-block compute gap:

1. **Element-wise op fusion via `MLX.compile`.** Each block has ≥4
   element-wise ops (LayerNorm × 2, LayerScale × 2, two residual adds,
   GELU) that aren't currently fused. mlx-swift's compile pass can fuse
   these into the surrounding matmuls. Expected win: 10–30%.
2. **Fused QKV projection.** Current code runs three separate Linear
   ops on the same input (`q(x)`, `k(x)`, `v(x)`). A single Linear of
   size `dim × 3*dim` followed by a split is one matmul instead of
   three. The HF DINOv2 checkpoint stores them separately, but they can
   be concatenated at load time. Expected win: 5–15%.
3. **SDPA kernel cost on ViT-S 6×64.** `MLXFast.scaledDotProductAttention`
   may be less optimized for this head config than MPS. Hard to fix
   without a kernel change; deferred until #1 and #2 are tried.

## Obvious wins — verified

Per the perf-gate checklist, all four were checked before deferring:

- **SDPA used everywhere.** `Attention.callAsFunction` in
  `Sources/MLXRFDETR/Backbone/Backbone.swift` calls
  `MLXFast.scaledDotProductAttention`; no hand-rolled `softmax(QKᵀ)V`.
- **No stray `eval()` / `.item()` in forward.** Verified across
  `Backbone.swift` (PatchEmbed, Block, DINOv2Backbone). Lazy graph stays
  intact through the full backbone forward.
- **NHWC conv weights at load time.** `WeightLoader.swift` transposes
  Conv2d `(O, I, kH, kW) → (O, kH, kW, I)` and ConvTransposed2d
  `(I, O, kH, kW) → (O, kH, kW, I)` once at load (`sanitized(key:value:)`
  + the transpose branches). No per-forward layout shuffling.
- **dtype consistency end-to-end.** `loadWeights(url:into:dtype:)` casts
  every value with `.asType(dtype)` at load. The only intentional fp32
  upcast is in pos-embed resampling (`resample(_:)`), which happens at
  load time and downcasts back before storing. Forward pass stays at the
  configured dtype.

## Disposition

The total inference time is **0.90× of Torch at matched fp16** (20.21 ms
vs 22.39 ms median), which clears the 1.2× total-time gate. The per-stage
gate fires only on the backbone (2.08×), but the cost is absorbed by the
transformer-stage win (0.47×). For the rf-detr port specifically, the
per-stage failure is acknowledged and the optimization work is deferred —
shipping the parity-correct port with this PERF_NOTES.md as the public
record, rather than blocking on a marginal end-to-end gain.

## If/when revisiting

1. **`MLX.compile` on per-block forward.** The most likely material win
   (10–30%). Each block has ≥4 element-wise ops not currently fused with
   the surrounding matmuls.
2. **Fused QKV.** Pre-concatenate `q.weight / k.weight / v.weight` at
   weight-load time, run one matmul + split. Expected 5–15%.
3. **SDPA on ViT-S 6×64.** Compare per-call SDPA cost against other
   mlx-swift ViT ports (mlx-swift-da3, mlx-swift-moge). If in-line, the
   gap is structural; if anomalous, file upstream.
4. Reproduce the per-block diagnostic with
   `RFDETRBench --diagnose --iterations 50 --warmup 10 --dtype float16`
   to verify the bottleneck distribution hasn't shifted.
