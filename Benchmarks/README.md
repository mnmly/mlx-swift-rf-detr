# RF-DETR Benchmarks

Compares the mlx-swift port (`Sources/RFDETRBench`) against the PyTorch
reference (`../../python/rf-detr`) on the same input image and model variant.

## Layout

- `benchmark_compare.py` — driver. Builds the Swift bench with
  `xcodebuild`, runs both Swift and PyTorch inference loops, and emits a
  side-by-side table.
- `../Sources/RFDETRBench/main.swift` — Swift bench target (built as
  `RFDETRBench` via the Swift package).
- `../Tests/fixtures/` — shared input fixture (`input.safetensors`) and
  weights. The driver reads from here for both backends.

## Reproduction

Generate the parity fixtures (one-time, requires the python rf-detr venv). Use the
repo's `.venv/bin/python` — **not** `uv run`, which tries to build the `trt` extra
(tensorrt) and fails on macOS:

```bash
cd ../../python/rf-detr
.venv/bin/python ../../swift/mlx-swift-rf-detr/Tests/fixtures/generate_fixtures.py
```

Run the comparison from the swift repo root with the same venv's Python (it has
`torch`, `safetensors`, and `rfdetr`). The driver builds the Swift bench with the
stable Xcode toolchain itself:

```bash
../../python/rf-detr/.venv/bin/python Benchmarks/benchmark_compare.py \
  --iterations 20 --warmup 5 \
  --swift-dtype float16 --python-dtype float16
```

Flags:
- `--iterations N` — timed iterations after warmup (default 20).
- `--warmup N` — warmup iterations (default 5).
- `--swift-dtype {float32,float16}` — Swift parameter dtype (default float32).
- `--python-dtype {float32,float16}` — Torch parameter dtype (default float32).
  For an apples-to-apples comparison, match the Swift dtype.
- `--python-device {mps,cpu}` — override torch device (default: mps if available).
- `--skip-swift` / `--skip-python` — run only one side.
- `--json` — machine-readable output.

## Synchronization

- Swift: `MLX.eval(...)` is called inside the timed loop so we measure
  compute, not graph construction.
- Python: `torch.mps.synchronize()` / `torch.cuda.synchronize()` is called
  before starting the timer and inside the per-iteration measurement.

Both sides preload the input tensor; the timed loop runs only the model
forward + post-processing.

## Hardware

Document the machine you ran on when posting numbers (Apple silicon
generation, RAM, macOS version). Don't quote relative speedups in
isolation — show the absolute numbers and let readers compute the ratio.
