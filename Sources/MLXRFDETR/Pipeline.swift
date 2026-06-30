// High-level inference wrapper. Mirrors the Python `RFDETRPipeline`:
// holds a model + processor + thresholds and exposes `predict(...)`
// that runs preprocessing, the forward pass, and post-processing.

import Foundation
import MLX

public final class RFDETRPipeline {
    public let model: RFDETRModel
    public let processor: RFDETRProcessor
    public var scoreThreshold: Float
    public var nmsThreshold: Float
    public var classNames: [String]?
    public var excludeClasses: Set<String>
    /// Keypoint score-fusion weight (GroupPose models only).
    public var keypointTraceAlpha: Float
    /// OKS threshold for pose-NMS deduplication of GroupPose detections. The
    /// keypoint head can emit several overlapping/offset skeletons for one person;
    /// detections of the same class whose Object Keypoint Similarity is `≥` this
    /// value are suppressed (lower score removed). Set `≥ 1.0` to disable.
    public var keypointOksThreshold: Float

    public init(
        model: RFDETRModel,
        processor: RFDETRProcessor,
        scoreThreshold: Float = 0.5,
        nmsThreshold: Float = 0.5,
        classNames: [String]? = nil,
        excludeClasses: [String] = [],
        keypointTraceAlpha: Float = 0.2,
        keypointOksThreshold: Float = 0.7
    ) {
        self.model = model
        self.processor = processor
        self.scoreThreshold = scoreThreshold
        self.nmsThreshold = nmsThreshold
        self.classNames = classNames
        self.excludeClasses = Set(excludeClasses)
        self.keypointTraceAlpha = keypointTraceAlpha
        self.keypointOksThreshold = keypointOksThreshold
    }

    /// Run inference on a pre-normalized `(1, H, W, 3)` tensor.
    public func predict(
        pixelValues: MLXArray,
        originalSize: (Int, Int)
    ) throws -> DetectionResult {
        let out = model(pixelValues)
        guard let logits = out["pred_logits"], let boxes = out["pred_boxes"] else {
            throw RFDETRError.invalidOutput
        }
        var result: DetectionResult
        if let keypoints = out["pred_keypoints"], model.config.useGroupposeKeypoints {
            result = postProcessKeypoints(
                predLogits: logits,
                predBoxes: boxes,
                predKeypoints: keypoints,
                originalSize: originalSize,
                numSelect: processor.numSelect,
                numKeypointsPerClass: model.config.numKeypointsPerClass,
                traceAlpha: keypointTraceAlpha,
                classNames: classNames
            )
            // postProcessKeypoints matches Python (threshold-less); apply the score
            // threshold here so callers/UI see only confident detections.
            result = filterByScore(result, threshold: scoreThreshold)
            // Deduplicate the overlapping/offset skeletons the set-prediction head
            // can emit for one person (the keypoint path applies no NMS upstream).
            if keypointOksThreshold < 1.0, let kps = result.keypoints, !kps.isEmpty {
                let keep = oksNmsKeep(
                    boxes: result.boxes,
                    scores: result.scores,
                    labels: result.labels,
                    keypoints: kps,
                    numKeypointsPerClass: model.config.numKeypointsPerClass,
                    oksThreshold: keypointOksThreshold
                )
                result = subset(result, keep)
            }
        } else {
            result = postProcess(
                predLogits: logits,
                predBoxes: boxes,
                originalSize: originalSize,
                scoreThreshold: scoreThreshold,
                numSelect: processor.numSelect,
                classNames: classNames,
                predMasks: out["pred_masks"],
                nmsThreshold: nmsThreshold
            )
        }
        if !excludeClasses.isEmpty {
            result = filterExcluded(result, excluded: excludeClasses)
        }
        return result
    }
}

#if canImport(AppKit) || canImport(UIKit)
import CoreGraphics
import ImageIO

public extension RFDETRPipeline {
    /// Run detection on a CGImage.
    func predict(cgImage: CGImage) throws -> DetectionResult {
        let (pixelValues, originalSize) = try loadAndPreprocess(cgImage: cgImage, processor: processor)
        return try predict(pixelValues: pixelValues, originalSize: originalSize)
    }

    /// Run detection on an image file (PNG, JPEG, etc.).
    func predict(url: URL) throws -> DetectionResult {
        let (pixelValues, originalSize) = try loadAndPreprocess(url: url, processor: processor)
        return try predict(pixelValues: pixelValues, originalSize: originalSize)
    }
}
#endif

public enum RFDETRError: LocalizedError {
    case invalidOutput

    public var errorDescription: String? {
        switch self {
        case .invalidOutput: "Model produced no detection outputs."
        }
    }
}

/// Reindex a result by `keep`, preserving every optional per-detection field.
private func subset(_ r: DetectionResult, _ keep: [Int]) -> DetectionResult {
    DetectionResult(
        boxes: keep.map { r.boxes[$0] },
        scores: keep.map { r.scores[$0] },
        labels: keep.map { r.labels[$0] },
        classNames: keep.map { r.classNames[$0] },
        masks: r.masks.map { m in keep.map { m[$0] } },
        keypoints: r.keypoints.map { k in keep.map { k[$0] } },
        keypointPrecisionCholesky: r.keypointPrecisionCholesky.map { p in keep.map { p[$0] } }
    )
}

/// Keep detections whose score is above `threshold`, preserving keypoints/precision.
private func filterByScore(_ r: DetectionResult, threshold: Float) -> DetectionResult {
    subset(r, (0..<r.count).filter { r.scores[$0] > threshold })
}

private func filterExcluded(_ r: DetectionResult, excluded: Set<String>) -> DetectionResult {
    var boxes = [[Float]]()
    var scores = [Float]()
    var labels = [Int]()
    var names = [String]()
    var masks: [Array2D<Float>]? = r.masks == nil ? nil : []
    var keypoints: [Array2D<Float>]? = r.keypoints == nil ? nil : []
    var precision: [Array2D<Float>]? = r.keypointPrecisionCholesky == nil ? nil : []

    for i in 0..<r.count {
        if excluded.contains(r.classNames[i]) { continue }
        boxes.append(r.boxes[i])
        scores.append(r.scores[i])
        labels.append(r.labels[i])
        names.append(r.classNames[i])
        if let m = r.masks { masks?.append(m[i]) }
        if let kp = r.keypoints { keypoints?.append(kp[i]) }
        if let pr = r.keypointPrecisionCholesky { precision?.append(pr[i]) }
    }
    return DetectionResult(boxes: boxes, scores: scores, labels: labels, classNames: names,
                           masks: masks, keypoints: keypoints, keypointPrecisionCholesky: precision)
}
