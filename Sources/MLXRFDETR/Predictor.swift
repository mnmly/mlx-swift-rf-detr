// High-level inference wrapper. Mirrors the Python `RFDETRPredictor`:
// holds a model + processor + thresholds and exposes `predict(...)`
// that runs preprocessing, the forward pass, and post-processing.

import Foundation
import MLX

public final class RFDETRPredictor {
    public let model: RFDETRModel
    public let processor: RFDETRProcessor
    public var scoreThreshold: Float
    public var nmsThreshold: Float
    public var classNames: [String]?
    public var excludeClasses: Set<String>

    public init(
        model: RFDETRModel,
        processor: RFDETRProcessor,
        scoreThreshold: Float = 0.5,
        nmsThreshold: Float = 0.5,
        classNames: [String]? = nil,
        excludeClasses: [String] = []
    ) {
        self.model = model
        self.processor = processor
        self.scoreThreshold = scoreThreshold
        self.nmsThreshold = nmsThreshold
        self.classNames = classNames
        self.excludeClasses = Set(excludeClasses)
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
        var result = postProcess(
            predLogits: logits,
            predBoxes: boxes,
            originalSize: originalSize,
            scoreThreshold: scoreThreshold,
            numSelect: processor.numSelect,
            classNames: classNames,
            predMasks: out["pred_masks"],
            nmsThreshold: nmsThreshold
        )
        if !excludeClasses.isEmpty {
            result = filterExcluded(result, excluded: excludeClasses)
        }
        return result
    }
}

#if canImport(AppKit) || canImport(UIKit)
import CoreGraphics
import ImageIO

public extension RFDETRPredictor {
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

private func filterExcluded(_ r: DetectionResult, excluded: Set<String>) -> DetectionResult {
    var boxes = [[Float]]()
    var scores = [Float]()
    var labels = [Int]()
    var names = [String]()
    var masks: [Array2D<Float>]? = r.masks == nil ? nil : []

    for i in 0..<r.count {
        if excluded.contains(r.classNames[i]) { continue }
        boxes.append(r.boxes[i])
        scores.append(r.scores[i])
        labels.append(r.labels[i])
        names.append(r.classNames[i])
        if let m = r.masks { masks?.append(m[i]) }
    }
    return DetectionResult(boxes: boxes, scores: scores, labels: labels, classNames: names, masks: masks)
}
