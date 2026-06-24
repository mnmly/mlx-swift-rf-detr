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
    /// Optional keypoints `(N, maxKeypoints, 3)` — `(x_px, y_px, confidence)` per detection.
    public var keypoints: [Array2D<Float>]?
    /// Optional keypoint precision Cholesky params `(N, maxKeypoints, 3)` — `(log_l11, l21, log_l22)`,
    /// `NaN` for inactive/padded keypoints.
    public var keypointPrecisionCholesky: [Array2D<Float>]?

    public init(
        boxes: [[Float]],
        scores: [Float],
        labels: [Int],
        classNames: [String],
        masks: [Array2D<Float>]? = nil,
        keypoints: [Array2D<Float>]? = nil,
        keypointPrecisionCholesky: [Array2D<Float>]? = nil
    ) {
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.classNames = classNames
        self.masks = masks
        self.keypoints = keypoints
        self.keypointPrecisionCholesky = keypointPrecisionCholesky
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

    // Map class indices to names. RF-DETR emits one extra logit at index
    // `numClasses` for the no-object/background slot; surface it as
    // "__background__" when a caller-supplied class list is in play (the
    // default sparse COCO list already covers id 90 = toothbrush, so the
    // sentinel only fires for fine-tuned models).
    let bgIndex = (classNames != nil) ? classNames!.count : -1
    let selNames = selectedClasses.map { c -> String in
        if c == bgIndex { return "__background__" }
        return c < names.count ? names[c] : ""
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

// MARK: - Keypoint post-process

/// Post-process keypoint-model outputs (GroupPose) into detections with keypoints.
///
/// Mirrors the Python `PostProcess` keypoint path: flattened top-k selection over
/// `(Q × C)`, per-class keypoint extraction, pixel scaling, `sigmoid(findable)`
/// confidence, raw Cholesky precision, and optional `trace_alpha` score fusion.
/// No NMS/threshold is applied (matches the reference), so callers that want those
/// should filter the returned result.
///
/// - Parameters:
///   - predLogits: `(1, Q, C)` raw class logits.
///   - predBoxes: `(1, Q, 4)` cxcywh normalized boxes.
///   - predKeypoints: `(1, Q, P, 8)` padded keypoints (`P = numClasses × maxKeypoints`).
///   - originalSize: `(H, W)` target image size (pixels).
///   - numSelect: number of query/class pairs to keep.
///   - numKeypointsPerClass: per-class active keypoint counts (e.g. `[0, 17]`).
///   - traceAlpha: keypoint-uncertainty score fusion weight (`0` disables).
///   - classNames: optional class-name list (defaults to COCO).
public func postProcessKeypoints(
    predLogits: MLXArray,
    predBoxes: MLXArray,
    predKeypoints: MLXArray,
    originalSize: (Int, Int),
    numSelect: Int = 100,
    numKeypointsPerClass: [Int],
    traceAlpha: Float = 0.2,
    classNames: [String]? = nil
) -> DetectionResult {
    eval(predLogits, predBoxes, predKeypoints)
    let Q = predLogits.dim(1)
    let C = predLogits.dim(-1)
    let P = predKeypoints.dim(2)
    let D = predKeypoints.dim(3)
    let logits: [Float] = predLogits.asArray(Float.self)
    let boxesRaw: [Float] = predBoxes.asArray(Float.self)
    let kps: [Float] = predKeypoints.asArray(Float.self)

    let numClasses = numKeypointsPerClass.count
    let maxK = numKeypointsPerClass.max() ?? 0

    // Flattened top-k over (Q × C) by sigmoid prob.
    var flat = [(Float, Int)]()
    flat.reserveCapacity(Q * C)
    for q in 0..<Q {
        for c in 0..<C {
            let s = 1 / (1 + exp(-logits[q * C + c]))
            flat.append((s, q * C + c))
        }
    }
    flat.sort { $0.0 > $1.0 }
    let k = min(numSelect, flat.count)
    let topk = Array(flat.prefix(k))
    var scores = topk.map { $0.0 }
    let queryIdx = topk.map { $0.1 / C }
    let labels = topk.map { $0.1 % C }

    // Boxes: cxcywh → xyxy, scaled to pixels.
    let origH = Float(originalSize.0), origW = Float(originalSize.1)
    var boxes = [[Float]]()
    boxes.reserveCapacity(k)
    for qi in queryIdx {
        let b = qi * 4
        let cx = boxesRaw[b], cy = boxesRaw[b + 1], w = boxesRaw[b + 2], h = boxesRaw[b + 3]
        boxes.append([(cx - w / 2) * origW, (cy - h / 2) * origH, (cx + w / 2) * origW, (cy + h / 2) * origH])
    }

    func logSumExp(_ xs: [Float]) -> Float {
        guard let m = xs.max() else { return -Float.infinity }
        return m + log(xs.reduce(Float(0)) { $0 + exp($1 - m) })
    }

    var outKeypoints = [Array2D<Float>]()
    var outPrecision = [Array2D<Float>]()
    outKeypoints.reserveCapacity(k)
    outPrecision.reserveCapacity(k)

    for det in 0..<k {
        let qi = queryIdx[det]
        let label = labels[det]
        var kdata = [Float](repeating: 0, count: maxK * 3)
        var pdata = [Float](repeating: Float.nan, count: maxK * 3)

        if label < numClasses {
            let activeCount = numKeypointsPerClass[label]
            let classOffset = label * maxK
            if activeCount > 0 {
                // trace_alpha score fusion (findability-weighted mean covariance trace).
                if traceAlpha > 0 && D >= 7 {
                    var logTraceSigma = [Float]()
                    var logWFind = [Float]()
                    for kk in 0..<activeCount {
                        let base = (qi * P + classOffset + kk) * D
                        let logL11 = kps[base + 4], l21 = kps[base + 5], logL22 = kps[base + 6]
                        let wFind = 1 / (1 + exp(-kps[base + 2]))
                        let logT1 = -2 * logL11
                        let logT2 = -2 * logL22
                        let logT3 = 2 * log(max(abs(l21), 1e-12)) + logT1 + logT2
                        logTraceSigma.append(logSumExp([logT1, logT2, logT3]))
                        logWFind.append(log(max(wFind, 1e-12)))
                    }
                    let combined = zip(logTraceSigma, logWFind).map { $0 + $1 }
                    let logMeanTrace = logSumExp(combined) - logSumExp(logWFind)
                    scores[det] *= exp(-traceAlpha * logMeanTrace)
                }
                // Pixel-space keypoints + raw Cholesky.
                for kk in 0..<activeCount {
                    let base = (qi * P + classOffset + kk) * D
                    kdata[kk * 3 + 0] = kps[base + 0] * origW
                    kdata[kk * 3 + 1] = kps[base + 1] * origH
                    kdata[kk * 3 + 2] = 1 / (1 + exp(-kps[base + 2]))
                    if D >= 7 {
                        pdata[kk * 3 + 0] = kps[base + 4]
                        pdata[kk * 3 + 1] = kps[base + 5]
                        pdata[kk * 3 + 2] = kps[base + 6]
                    }
                }
            }
        }
        outKeypoints.append(Array2D(data: kdata, rows: maxK, cols: 3))
        outPrecision.append(Array2D(data: pdata, rows: maxK, cols: 3))
    }

    let names = classNames ?? COCO_CLASSES
    let selNames = labels.map { $0 < names.count ? names[$0] : "class_\($0)" }

    return DetectionResult(
        boxes: boxes, scores: scores, labels: labels, classNames: selNames,
        keypoints: outKeypoints, keypointPrecisionCholesky: outPrecision
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
