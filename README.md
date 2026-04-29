# RF-DETR for mlx-swift

Real-time detection transformer ([RF-DETR](https://github.com/roboflow/rf-detr), ICLR 2026) ported to Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift). Detection and instance segmentation on COCO 80 classes.

## Install

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/<you>/mlx-swift-rf-detr", from: "0.0.1"),
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

Use the [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) converter to produce an MLX-compatible directory (one-time, requires Python + PyTorch + safetensors):

```bash
git clone https://github.com/Blaizzy/mlx-vlm
cd mlx-vlm
python -m mlx_vlm.models.rfdetr.convert --variant base --output ./rfdetr-base-mlx
```

Available `--variant` values: `base`, `small`, `large`, `seg-small`, `seg-large`, `seg-xlarge`, `seg-2xlarge`.

The output directory contains `config.json`, `preprocessor_config.json`, and `model.safetensors`. No PyTorch dependency at inference time.

## Quick start

```swift
import MLXRFDETR

let dir = URL(fileURLWithPath: "./rfdetr-base-mlx")
let (model, processor, _) = try RFDETR.load(directory: dir)

let predictor = RFDETRPredictor(
    model: model,
    processor: processor,
    scoreThreshold: 0.3,
    nmsThreshold: 0.5
)

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
let predictor = RFDETRPredictor(
    model: model,
    processor: processor,
    scoreThreshold: 0.3,
    excludeClasses: ["couch", "potted plant"]
)
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

`RFDETR.load(directory:)` reads the variant fields from `config.json`, so you don't need to pick the right variant manually.

## Lower-level API

If you need finer control, you can build the model yourself — see `RFDETRModel`, `DINOv2Backbone`, `MultiScaleProjector`, `SegmentationHead`, `loadWeights(url:into:)`, and `postProcess(...)`.

## Example app

`Examples/RFDETRApp/` contains a SwiftUI macOS/iOS demo with image, video, and live-camera inference.

## Reference

- [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2511.09554) (ICLR 2026)
- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
- [mlx-vlm rfdetr converter](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/rfdetr)
