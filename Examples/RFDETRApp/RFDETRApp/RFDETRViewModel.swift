import SwiftUI
import CoreGraphics
import MLX
import MLXRFDETR

// Sendable conformances so we can hop between actors. The MLX-backed values
// are read-only after model load; DetectionResult contains only Swift value types.
extension RFDETRPredictor: @unchecked @retroactive Sendable {}
extension DetectionResult: @unchecked @retroactive Sendable {}

@MainActor
@Observable
final class RFDETRViewModel {
    var predictor: RFDETRPredictor?
    var variant: RFDETRVariant?
    var resolution: Int = 0
    var hasSegmentation: Bool = false

    var isLoaded = false
    var isLoading = false
    var errorMessage: String?
    var modelPath: String = ""
    private var loadedURL: URL?

    func loadModel(from directory: URL) async {
        isLoading = true
        errorMessage = nil
        modelPath = directory.path
        defer { isLoading = false }

        do {
            let (model, processor, detected) = try await Task.detached(priority: .userInitiated) {
                try RFDETR.load(directory: directory, dtype: .float16)
            }.value

            self.predictor = RFDETRPredictor(
                model: model,
                processor: processor,
                scoreThreshold: 0.5,
                nmsThreshold: 0.5
            )
            self.variant = detected
            self.resolution = processor.resolution
            self.hasSegmentation = (model.segmentationHead != nil)
            self.isLoaded = true
            self.loadedURL = directory
        } catch {
            self.errorMessage = error.localizedDescription
            self.predictor = nil
            self.isLoaded = false
        }
    }

    func reloadIfNeeded() async {
        guard let url = loadedURL, !isLoading else { return }
        await loadModel(from: url)
    }

    /// Run inference on a CGImage off the main actor.
    nonisolated func predictAsync(_ cgImage: CGImage, scoreThreshold: Float) async throws -> DetectionResult {
        guard let predictor = await self.predictor else { throw RFDETRViewModelError.modelNotLoaded }
        return try await Task.detached(priority: .userInitiated) {
            predictor.scoreThreshold = scoreThreshold
            return try predictor.predict(cgImage: cgImage)
        }.value
    }
}

enum RFDETRViewModelError: LocalizedError {
    case modelNotLoaded
    case noImage

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: "Model not loaded. Select a converted model directory first."
        case .noImage:        "No image selected."
        }
    }
}
