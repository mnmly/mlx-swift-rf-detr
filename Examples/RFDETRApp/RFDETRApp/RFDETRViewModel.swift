import SwiftUI
import CoreGraphics
import MLX
import MLXRFDETR

// Sendable conformances so we can hop between actors. The MLX-backed values
// are read-only after model load; DetectionResult contains only Swift value types.
extension RFDETRPipeline: @unchecked @retroactive Sendable {}
extension DetectionResult: @unchecked @retroactive Sendable {}

@MainActor
@Observable
final class RFDETRViewModel {
    var predictor: RFDETRPipeline?
    var variant: RFDETRVariant?
    var resolution: Int = 0
    var hasSegmentation: Bool = false
    /// True when the loaded model is a GroupPose keypoint model (enables the pose-dedup control).
    var hasKeypoints: Bool = false

    var isLoaded = false
    var isLoading = false
    var errorMessage: String?
    var modelPath: String = ""
    private var loadedURL: URL?

    /// Non-nil while a model download is in flight; 0...1 fraction of the (dominant) weights file.
    var downloadProgress: Double?
    /// Id of the model currently downloading, for the status label.
    var downloadingName: String?

    func loadModel(from directory: URL) async {
        isLoading = true
        errorMessage = nil
        modelPath = directory.path
        defer { isLoading = false }

        // Sandbox: a bookmark-resolved URL needs its scope started for the read.
        // Panel URLs and download dirs (covered by an enclosing scope held by the
        // caller) return false here but stay readable, so bracketing is always safe.
        let accessing = directory.startAccessingSecurityScopedResource()
        defer { if accessing { directory.stopAccessingSecurityScopedResource() } }

        do {
            // RFDETR.load is synchronous and heavy (weights I/O + graph build), so run
            // it off the main actor. Only a Sendable tuple crosses back — the
            // non-Sendable model/processor never leave the detached task.
            let (loaded, detected, res, hasSeg, hasKp) = try await Task.detached(priority: .userInitiated) {
                let (model, processor, detected) = try RFDETR.load(directory: directory, dtype: .float16)
                let predictor = RFDETRPipeline(
                    model: model, processor: processor,
                    scoreThreshold: 0.5, nmsThreshold: 0.5
                )
                return (predictor, detected, processor.resolution,
                        model.segmentationHead != nil, model.config.useGroupposeKeypoints)
            }.value

            self.predictor = loaded
            self.variant = detected
            self.resolution = res
            self.hasSegmentation = hasSeg
            self.hasKeypoints = hasKp
            self.isLoaded = true
            self.loadedURL = directory
            ModelBookmark.store(directory)
        } catch {
            self.errorMessage = error.localizedDescription
            self.predictor = nil
            self.isLoaded = false
        }
    }

    func reloadIfNeeded() async {
        // Prefer the security-scoped bookmark so reload works even after the original
        // open-panel/download scope has been released.
        guard !isLoading, let url = ModelBookmark.resolve() ?? loadedURL else { return }
        await loadModel(from: url)
    }

    /// Re-open the most recently loaded model from its security-scoped bookmark, if one
    /// was persisted. Call once at launch; no-op when nothing is bookmarked.
    func restoreLastModel() async {
        guard !isLoaded, !isLoading, let url = ModelBookmark.resolve() else { return }
        await loadModel(from: url)
    }

    /// Download a catalog model's three files into the user-chosen `parent` folder, then load it.
    func downloadAndLoad(_ model: RemoteModel, intoParent parent: URL) async {
        downloadingName = model.id
        downloadProgress = 0
        errorMessage = nil

        // Hold the user-granted folder scope across BOTH the download (write) and the
        // subsequent load + bookmark of the model subdirectory.
        let scoped = parent.startAccessingSecurityScopedResource()
        defer { if scoped { parent.stopAccessingSecurityScopedResource() } }

        do {
            let dir = try await ModelFetcher.fetch(model, intoParent: parent) { frac in
                Task { @MainActor in self.downloadProgress = frac }
            }
            downloadingName = nil
            downloadProgress = nil
            await loadModel(from: dir)
        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
            downloadingName = nil
            downloadProgress = nil
        }
    }

    /// Run inference on a CGImage off the main actor.
    ///
    /// - Parameter oksThreshold: Pose-NMS OKS threshold for GroupPose models
    ///   (`1.0` disables dedup); ignored by detection/segmentation models.
    nonisolated func predictAsync(
        _ cgImage: CGImage, scoreThreshold: Float, oksThreshold: Float = 0.7
    ) async throws -> DetectionResult {
        guard let predictor = await self.predictor else { throw RFDETRViewModelError.modelNotLoaded }
        return try await Task.detached(priority: .userInitiated) {
            predictor.scoreThreshold = scoreThreshold
            predictor.keypointOksThreshold = oksThreshold
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
