import SwiftUI
import CoreGraphics
import MLX
import MLXRFDETR

// Local Sendable conformances so we can hop between actors. The MLX-backed
// values are read-only after model load; processor is a value type with
// plain numeric fields; DetectionResult contains only Swift value types.
extension RFDETRModel: @unchecked @retroactive Sendable {}
extension RFDETRProcessor: @unchecked @retroactive Sendable {}
extension DetectionResult: @unchecked @retroactive Sendable {}

/// Variant presets matching the RF-DETR backbone configurations.
enum RFDETRVariant: String, CaseIterable, Identifiable {
    case small
    case base
    case medium

    var id: String { rawValue }

    var label: String {
        switch self {
        case .small:  "Small (S)"
        case .base:   "Base (B)"
        case .medium: "Medium (M)"
        }
    }

    /// Backbone hyperparameters (mirrors rf-detr-mlx model variants).
    /// Defaults assume DINOv2-style backbones with 4 feature taps.
    var backboneSpec: (imgSize: Int, embedDim: Int, depth: Int, numHeads: Int, numWindows: Int, featureIndices: [Int], numRegisterTokens: Int) {
        switch self {
        case .small:
            return (512, 384, 12, 6, 2, [2, 5, 8, 11], 0)
        case .base:
            return (560, 768, 12, 12, 4, [2, 5, 8, 11], 4)
        case .medium:
            return (576, 768, 12, 12, 4, [2, 5, 8, 11], 4)
        }
    }

    var projectorInChannels: Int {
        switch self {
        case .small:  return 384 * 4
        case .base:   return 768 * 4
        case .medium: return 768 * 4
        }
    }

    var resolution: Int { backboneSpec.imgSize }

    var config: RFDETRConfig {
        // numClasses=90 → +1 background = 91 (COCO_CLASSES has 91 entries)
        RFDETRConfig(numClasses: 90)
    }
}

@MainActor
@Observable
final class RFDETRViewModel {
    var model: RFDETRModel?
    var processor: RFDETRProcessor = RFDETRProcessor()
    var variant: RFDETRVariant = .small
    var enableSegmentation: Bool = false
    var isLoaded = false
    var isLoading = false
    var errorMessage: String?
    var modelPath: String = ""
    private var loadedURL: URL?

    func loadModel(from url: URL) async {
        isLoading = true
        errorMessage = nil
        modelPath = url.path
        defer { isLoading = false }

        do {
            let spec = variant.backboneSpec
            let backbone = DINOv2Backbone(
                imgSize: spec.imgSize,
                patchSize: 16,
                embedDim: spec.embedDim,
                depth: spec.depth,
                numHeads: spec.numHeads,
                numWindows: spec.numWindows,
                featureIndices: spec.featureIndices,
                numRegisterTokens: spec.numRegisterTokens
            )
            let projector = MultiScaleProjector(
                inChannels: variant.projectorInChannels,
                hiddenDim: variant.config.hiddenDim
            )
            let segHead: SegmentationHead? = enableSegmentation
                ? SegmentationHead(inDim: variant.config.hiddenDim)
                : nil
            let m = RFDETRModel(
                config: variant.config,
                backbone: backbone,
                projector: projector,
                segmentationHead: segHead
            )
            try loadWeights(url: url, into: m, dtype: .float16)
            eval(m)

            self.model = m
            self.processor = RFDETRProcessor(resolution: spec.imgSize)
            self.isLoaded = true
            self.loadedURL = url
        } catch {
            self.errorMessage = error.localizedDescription
            self.model = nil
            self.isLoaded = false
        }
    }

    func reloadIfNeeded() async {
        guard let url = loadedURL, !isLoading else { return }
        await loadModel(from: url)
    }

    /// Runs a single image through the model on a background task.
    nonisolated func predictAsync(_ cgImage: CGImage, scoreThreshold: Float) async throws -> DetectionResult {
        guard let m = await self.model else { throw RFDETRViewModelError.modelNotLoaded }
        let proc = await self.processor

        return try await Task.detached(priority: .userInitiated) {
            let (pixelValues, originalSize) = try preprocessCGImage(cgImage, processor: proc)
            let out = m(pixelValues)
            guard let logits = out["pred_logits"], let boxes = out["pred_boxes"] else {
                throw RFDETRViewModelError.invalidOutput
            }
            return postProcess(
                predLogits: logits,
                predBoxes: boxes,
                originalSize: originalSize,
                scoreThreshold: scoreThreshold,
                numSelect: proc.numSelect,
                classNames: nil,
                predMasks: out["pred_masks"],
                nmsThreshold: 0.5
            )
        }.value
    }
}

enum RFDETRViewModelError: LocalizedError {
    case modelNotLoaded
    case noImage
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: "Model not loaded. Select a weights file first."
        case .noImage:        "No image selected."
        case .invalidOutput:  "Model produced no detection outputs."
        }
    }
}

// MARK: - CGImage preprocessing

/// Resize a CGImage to (resolution, resolution) and produce a normalized
/// (1, H, W, 3) MLXArray, matching `RFDETRProcessor.normalize`.
nonisolated func preprocessCGImage(_ cgImage: CGImage, processor: RFDETRProcessor) throws -> (pixelValues: MLXArray, originalSize: (Int, Int)) {
    let origH = cgImage.height
    let origW = cgImage.width
    let res = processor.resolution

    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil,
        width: res, height: res,
        bitsPerComponent: 8,
        bytesPerRow: res * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        throw RFDETRViewModelError.invalidOutput
    }
    ctx.interpolationQuality = .default
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: res, height: res))

    guard let data = ctx.data else { throw RFDETRViewModelError.invalidOutput }
    var rgb = [Float](repeating: 0, count: res * res * 3)
    let buf = data.bindMemory(to: UInt8.self, capacity: res * res * 4)
    for i in 0..<(res * res) {
        rgb[i * 3]     = Float(buf[i * 4])     / 255.0
        rgb[i * 3 + 1] = Float(buf[i * 4 + 1]) / 255.0
        rgb[i * 3 + 2] = Float(buf[i * 4 + 2]) / 255.0
    }
    let tensor = MLXArray(rgb, [res, res, 3])
    let normalized = processor.normalize(tensor)
    return (normalized, (origH, origW))
}
