// Post-processing: box decoding, NMS, and detection result assembly.
//
// Runs on CPU (Swift [Float]) for per-element control flow (NMS, thresholding)
// which is more natural outside the MLX graph.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/lwdetr.py (PostProcess class)

import Foundation
import MLX

// MARK: - DetectionResult

/// Single-image detection output.
public struct DetectionResult {
    /// Boxes in `(N, 4)` xyxy format (pixel coordinates, original image size).
    public var boxes: [[Float]]
    /// Confidence scores `(N,)`.
    public var scores: [Float]
    /// Integer class indices `(N,)`.
    public var labels: [Int]
    /// Class names `(N,)`.
    public var classNames: [String]
    /// Optional mask logits `(N, H, W)` — raw logits from segmentation head, may be nil.
    public var masks: [Array2D<Float>]?

    public init(
        boxes: [[Float]],
        scores: [Float],
        labels: [Int],
        classNames: [String],
        masks: [Array2D<Float>]? = nil
    ) {
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.classNames = classNames
        self.masks = masks
    }

    /// Number of detections.
    public var count: Int { boxes.count }
}

/// A 2D array stored as flat row-major `[Float]` with dimensions.
public struct Array2D<Scalar: BinaryFloatingPoint> {
    public let data: [Scalar]
    public let rows: Int
    public let cols: Int

    public init(data: [Scalar], rows: Int, cols: Int) {
        self.data = data
        self.rows = rows
        self.cols = cols
    }

    public subscript(_ y: Int, _ x: Int) -> Scalar {
        data[y * cols + x]
    }
}

// MARK: - Box conversion

/// Convert boxes from center-format to corner-format.
/// - Parameter boxes: `(N, 4)` in `[cx, cy, w, h]` format.
/// - Returns: `(N, 4)` in `[x1, y1, x2, y2]` format.
public func boxCxcywhToXyxy(_ boxes: [[Float]]) -> [[Float]] {
    boxes.map { b in
        let cx = b[0], cy = b[1], w = b[2], h = b[3]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    }
}

// MARK: - IoU computation

private func boxIou(_ boxes1: [[Float]], _ boxes2: [[Float]]) -> [[Float]] {
    let m = boxes1.count
    let n = boxes2.count
    var iou = [[Float]](repeating: [Float](repeating: 0, count: n), count: m)

    for i in 0..<m {
        let b1 = boxes1[i]
        let area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        for j in 0..<n {
            let b2 = boxes2[j]
            let x1 = max(b1[0], b2[0])
            let y1 = max(b1[1], b2[1])
            let x2 = min(b1[2], b2[2])
            let y2 = min(b1[3], b2[3])
            let inter = max(0, x2 - x1) * max(0, y2 - y1)
            let area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            let union = area1 + area2 - inter
            iou[i][j] = inter / (union + 1e-6)
        }
    }
    return iou
}

// MARK: - NMS

/// Per-class Non-Maximum Suppression.
///
/// - Parameters:
///   - boxes: `(N, 4)` xyxy format
///   - scores: `(N,)`
///   - classes: `(N,)` class indices
///   - iouThreshold: suppress boxes with IoU ≥ this value
/// - Returns: indices of boxes to keep
public func nmsPerClass(
    boxes: [[Float]],
    scores: [Float],
    classes: [Int],
    iouThreshold: Float = 0.5
) -> [Int] {
    var keep: [Int] = []
    let uniqueClasses = Set(classes)

    for cls in uniqueClasses {
        let clsIndices = classes.enumerated().filter { $0.1 == cls }.map { $0.0 }
        let clsBoxes = clsIndices.map { boxes[$0] }
        let clsScores = clsIndices.map { scores[$0] }

        // Sort by score descending
        let order = clsScores.enumerated().sorted { $0.1 > $1.1 }.map { $0.0 }
        var sortedIndices = order.map { clsIndices[$0] }
        var sortedBoxes = order.map { clsBoxes[$0] }

        while !sortedIndices.isEmpty {
            keep.append(sortedIndices[0])
            if sortedIndices.count == 1 { break }

            let ious = boxIou([sortedBoxes[0]], Array(sortedBoxes.dropFirst()))[0]
            let remaining = sortedIndices.dropFirst().enumerated()
                .filter { ious[$0.0] < iouThreshold }
                .map { $0.1 }

            sortedIndices = remaining
            sortedBoxes = remaining.map { boxes[$0] }
        }
    }

    if keep.isEmpty { return [] }
    return keep.sorted { scores[$0] > scores[$1] }
}

// MARK: - Full post-process

/// Post-process model outputs into detections.
///
/// - Parameters:
///   - predLogits: `(B, Q, numClasses)` raw class logits (MLXArray, batch size 1 for single image)
///   - predBoxes: `(B, Q, 4)` boxes in cxcywh normalized [0, 1]
///   - originalSize: `(H, W)` original image dimensions (pixels)
///   - scoreThreshold: minimum confidence threshold
///   - numSelect: max detections before thresholding
///   - classNames: optional class name list (defaults to COCO)
///   - predMasks: optional `(B, Q, Hm, Wm)` mask logits
///   - nmsThreshold: NMS IoU threshold
/// - Returns: `DetectionResult` for the first image in batch
public func postProcess(
    predLogits: MLXArray,
    predBoxes: MLXArray,
    originalSize: (Int, Int),
    scoreThreshold: Float = 0.5,
    numSelect: Int = 300,
    classNames: [String]? = nil,
    predMasks: MLXArray? = nil,
    nmsThreshold: Float = 0.5
) -> DetectionResult {
    let names = classNames ?? COCO_CLASSES

    // Evaluate and pull to CPU
    eval(predLogits, predBoxes)

    // Sigmoid
    let logits: [Float] = predLogits.asArray(Float.self)
    let Q = predLogits.dim(1)
    let C = predLogits.dim(-1)
    let boxesRaw: [Float] = predBoxes.asArray(Float.self)

    var scores = [Float](repeating: 0, count: Q)
    var maxClasses = [Int](repeating: 0, count: Q)

    for q in 0..<Q {
        let base = q * C
        var maxScore: Float = -Float.infinity
        var maxCls = 0
        for c in 0..<C {
            let s = 1 / (1 + exp(-logits[base + c]))  // sigmoid
            if s > maxScore {
                maxScore = s
                maxCls = c
            }
        }
        scores[q] = maxScore
        maxClasses[q] = maxCls
    }

    // Top-K by score
    var topkIdx = [Int](0..<Q)
    if numSelect < Q {
        topkIdx = topkIdx.sorted { scores[$0] > scores[$1] }
        topkIdx = Array(topkIdx.prefix(numSelect))
    } else {
        topkIdx.sort { scores[$0] > scores[$1] }
    }

    var selectedScores = topkIdx.map { scores[$0] }
    var selectedClasses = topkIdx.map { maxClasses[$0] }
    var selectedBoxes = topkIdx.map { idx -> [Float] in
        let b = idx * 4
        return [boxesRaw[b], boxesRaw[b + 1], boxesRaw[b + 2], boxesRaw[b + 3]]
    }

    // Filter by threshold (apply unconditionally — an empty result is the
    // correct outcome when nothing crosses the threshold).
    let keep = selectedScores.enumerated().filter { $0.1 > scoreThreshold }
    selectedScores = keep.map { $0.1 }
    selectedClasses = keep.map { selectedClasses[$0.0] }
    selectedBoxes = keep.map { selectedBoxes[$0.0] }
    topkIdx = keep.map { topkIdx[$0.0] }

    // cxcywh → xyxy
    selectedBoxes = boxCxcywhToXyxy(selectedBoxes)

    // Scale to original image size
    let origH = Float(originalSize.0)
    let origW = Float(originalSize.1)
    for i in 0..<selectedBoxes.count {
        selectedBoxes[i][0] *= origW
        selectedBoxes[i][1] *= origH
        selectedBoxes[i][2] *= origW
        selectedBoxes[i][3] *= origH
        // Clip
        selectedBoxes[i][0] = max(0, min(selectedBoxes[i][0], origW))
        selectedBoxes[i][1] = max(0, min(selectedBoxes[i][1], origH))
        selectedBoxes[i][2] = max(0, min(selectedBoxes[i][2], origW))
        selectedBoxes[i][3] = max(0, min(selectedBoxes[i][3], origH))
    }

    // NMS
    if nmsThreshold < 1.0 && !selectedBoxes.isEmpty {
        let nmsKeep = nmsPerClass(
            boxes: selectedBoxes,
            scores: selectedScores,
            classes: selectedClasses,
            iouThreshold: nmsThreshold
        )
        selectedScores = nmsKeep.map { selectedScores[$0] }
        selectedClasses = nmsKeep.map { selectedClasses[$0] }
        selectedBoxes = nmsKeep.map { selectedBoxes[$0] }
        topkIdx = nmsKeep.map { topkIdx[$0] }
    }

    // Map class indices to names
    let selNames = selectedClasses.map { c in
        c < names.count ? names[c] : "class_\(c)"
    }

    // Process masks
    var resultMasks: [Array2D<Float>]?
    if let predMasks, !topkIdx.isEmpty {
        // predMasks: (1, Q, mH, mW) → evaluate
        eval(predMasks)
        let mDim1 = predMasks.dim(2)
        let mDim2 = predMasks.dim(3)
        let flatMasks: [Float] = predMasks.asArray(Float.self)
        var masks = [Array2D<Float>]()
        for idx in topkIdx {
            let start = idx * mDim1 * mDim2
            let data = Array(flatMasks[start..<(start + mDim1 * mDim2)])
            masks.append(Array2D(data: data, rows: mDim1, cols: mDim2))
        }
        resultMasks = masks
    }

    return DetectionResult(
        boxes: selectedBoxes,
        scores: selectedScores,
        labels: selectedClasses,
        classNames: selNames,
        masks: resultMasks
    )
}

// MARK: - COCO class names

public let COCO_CLASSES: [String] = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
