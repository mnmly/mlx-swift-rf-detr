# RF-DETR for mlx-swift

Real-time detection transformer ([RF-DETR](https://github.com/roboflow/rf-detr), ICLR 2026) ported to Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift). Detection, instance segmentation, and keypoint (pose) estimation on COCO classes.

📖 [API documentation](https://mnmly.github.io/mlx-swift-rf-detr/)

<img width="1200" height="727" alt="Image" src="https://github.com/user-attachments/assets/7212e4c3-2916-4729-a725-81d3e703cdb3" />

<small>Reference Video: Merce Cunningham Dance Company performing “Antic Meet” (1958), with costume design by Robert Rauschenberg and music by John Cage, 1964</small>

## Install

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/mnmly/mlx-swift-rf-detr", from: "0.1.0"),
```

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "MLXRFDETR", package: "mlx-swift-rf-detr"),
    ]
)
```

## Convert weights

The reference PyTorch model lives at [roboflow/rf-detr](https://github.com/roboflow/rf-detr); this port tracks upstream **[`1.8.1`](https://github.com/roboflow/rf-detr/releases/tag/1.8.1)** (detection + segmentation deliverables, plus the `keypoint-preview` GroupPose model). The Swift loader reads a converted directory containing `config.json`, `preprocessor_config.json`, and `model.safetensors` (no PyTorch dependency at inference time).

Available variants: `base`, `small`, `large`, `seg-small`, `seg-large`, `seg-xlarge`, `seg-2xlarge`, `keypoint-preview`.

Any converter that emits the above three files works; the [mlx-vlm rfdetr converter](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/rfdetr) is one option for detection/segmentation. For the `keypoint-preview` model, use [`Scripts/convert_keypoint.py`](Scripts/convert_keypoint.py) (downloads the upstream checkpoint and writes the converted directory).

## Quick start

```swift
import MLXRFDETR

let dir = URL(fileURLWithPath: "./rfdetr-base-mlx")
let predictor = try MLXRFDETR.fromPretrained(dir, scoreThreshold: 0.3, nmsThreshold: 0.5)

// From a file URL
let result = try predictor.predict(url: URL(fileURLWithPath: "image.jpg"))

// Or from a CGImage
// let result = try predictor.predict(cgImage: cg)

for i in 0..<result.count {
    let box = result.boxes[i]
    print("\(result.classNames[i]): \(result.scores[i]) [\(box[0]), \(box[1]), \(box[2]), \(box[3])]")
}
```

`DetectionResult` exposes `boxes` (xyxy pixel coords on the original image), `scores`, `labels`, `classNames`, and — for segmentation variants — `masks` (per-instance mask logits).

## Filtering

```swift
var predictor = try MLXRFDETR.fromPretrained(dir, scoreThreshold: 0.3)
predictor.excludeClasses = ["couch", "potted plant"]
```

## Variants

| Variant | Task | Resolution |
|---------|------|-----------|
| `base` | Detection | 560 |
| `small` | Detection | 512 |
| `large` | Detection | 560 |
| `seg-small` | Detection + masks | 384 |
| `seg-large` | Detection + masks | 504 |
| `seg-xlarge` | Detection + masks | 624 |
| `seg-2xlarge` | Detection + masks | 768 |
| `keypoint-preview` | Detection + keypoints (person, 17 COCO) | 576 |

`MLXRFDETR.fromPretrained(_:)` (and the underlying `RFDETR.load(directory:)`) reads the variant fields from `config.json`, so you don't need to pick the right variant manually.

## Lower-level API

If you need finer control, you can build the model yourself — see `RFDETRModel`, `DINOv2Backbone`, `MultiScaleProjector`, `SegmentationHead`, `loadWeights(url:into:)`, and `postProcess(...)`.

## Benchmarks

See [`Benchmarks/README.md`](Benchmarks/README.md) for the reproduction
protocol. Inference-only, batch=1, 512×512 input, RF-DETR Small.

| Backend | dtype | Median (ms) | Mean (ms) | Min | Max |
|---|---|---|---|---|---|
| Torch / MPS | float32 | 29.62 | 29.77 | 25.69 | 33.32 |
| Torch / MPS | float16 | 22.39 | 22.20 | 19.66 | 24.12 |
| mlx-swift | float16 | 20.21 | 20.30 | 19.24 | 21.53 |

Per-stage medians (ms, all fp16):

| Stage | Torch / MPS | mlx-swift |
|---|---|---|
| backbone | 4.91 | 10.22 |
| projector | 1.32 | 1.81 |
| transformer | 15.75 | 7.41 |
| heads | 0.42 | 0.78 |

Hardware: Apple M5 Max, 128 GB, macOS 26.5. 20 iterations, 5 warmup.

> Backbone runs ~2.1× slower than Torch MPS at matched fp16; total stays
> within 10%. See [`PERF_NOTES.md`](PERF_NOTES.md) for the diagnosis,
> verified obvious wins, and deferred optimization plan.
Reproduce with:

```bash
python Benchmarks/benchmark_compare.py \
  --iterations 20 --warmup 5 \
  --swift-dtype float16 --python-dtype float16
```

## Example app

`Examples/RFDETRApp/` contains a SwiftUI macOS/iOS demo with image, video, and live-camera inference.

## Reference

- [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2511.09554) (ICLR 2026)
- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
- [mlx-vlm rfdetr converter](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/rfdetr)
