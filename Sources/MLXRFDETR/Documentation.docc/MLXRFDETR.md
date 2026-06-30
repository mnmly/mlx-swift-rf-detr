# ``MLXRFDETR``

Real-time RF-DETR object detection, instance segmentation, and keypoint
(pose) estimation on Apple Silicon, ported to MLX.

## Overview

`MLXRFDETR` is a pure-Swift, inference-only port of
[RF-DETR](https://github.com/roboflow/rf-detr) (ICLR 2026) onto
[mlx-swift](https://github.com/ml-explore/mlx-swift). It runs a
DINOv2-windowed backbone, a multi-scale projector, and an RF-DETR decoder
with deformable cross-attention — no PyTorch at inference time. Three task
families are supported, selected automatically from the model's
`config.json`:

- **Detection** — COCO boxes + labels.
- **Instance segmentation** — per-detection masks.
- **Keypoints (GroupPose)** — per-detection keypoints with confidence and
  precision-Cholesky uncertainty.

Load a converted model directory (`config.json` +
`preprocessor_config.json` + `model.safetensors`) and run inference:

```swift
import MLXRFDETR

let dir = URL(fileURLWithPath: "./rfdetr-base-mlx")
let predictor = try MLXRFDETR.fromPretrained(dir, scoreThreshold: 0.3, nmsThreshold: 0.5)

let result = try predictor.predict(url: imageURL)
for i in 0..<result.count {
    print(result.classNames[i], result.scores[i], result.boxes[i])
}
```

For finer control, build the model and processor yourself with
``RFDETR/load(directory:dtype:)`` and post-process with ``postProcess(predLogits:predBoxes:originalSize:scoreThreshold:numSelect:classNames:predMasks:nmsThreshold:)``
or ``postProcessKeypoints(predLogits:predBoxes:predKeypoints:originalSize:numSelect:numKeypointsPerClass:traceAlpha:classNames:)``.

## Topics

### Loading a model

- ``MLXRFDETR``
- ``RFDETR``
- ``RFDETRVariant``

### Running inference

- ``RFDETRPipeline``
- ``RFDETRProcessor``
- ``DetectionResult``

### Post-processing outputs

- ``postProcess(predLogits:predBoxes:originalSize:scoreThreshold:numSelect:classNames:predMasks:nmsThreshold:)``
- ``postProcessKeypoints(predLogits:predBoxes:predKeypoints:originalSize:numSelect:numKeypointsPerClass:traceAlpha:classNames:)``
- ``nmsPerClass(boxes:scores:classes:iouThreshold:)``
- ``boxCxcywhToXyxy(_:)``
- ``Array2D``

### Configuring the model

- ``RFDETRConfig``

### Model architecture

- ``RFDETRModel``
- ``DINOv2Backbone``
- ``MultiScaleProjector``
- ``Transformer``
- ``Decoder``
- ``DecoderLayer``
- ``SegmentationHead``
- ``ConditionalQueryInitializer``
- ``MSDeformableAttention``
