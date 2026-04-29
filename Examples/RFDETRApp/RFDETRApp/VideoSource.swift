import Foundation
import AVFoundation
import CoreGraphics
import CoreMedia

/// Extracts frames from a video file using AVAssetImageGenerator.
///
/// `vidStride` controls how many frames to skip between yields (1 = every frame).
/// Frames are decoded at the asset's natural resolution.
final class VideoSource {
    private let generator: AVAssetImageGenerator
    private let timestamps: [CMTime]
    private var index: Int = 0

    let totalFrames: Int
    let nominalFPS: Double

    init(url: URL, vidStride: Int = 1) async throws {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else {
            throw NSError(domain: "VideoSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "No video track in \(url.lastPathComponent)"])
        }

        let duration = try await asset.load(.duration)
        let fps = try await track.load(.nominalFrameRate)
        self.nominalFPS = Double(max(fps, 1))

        let durationSec = CMTimeGetSeconds(duration)
        let totalApprox = max(1, Int(durationSec * nominalFPS))

        var times: [CMTime] = []
        let stride = max(1, vidStride)
        var i = 0
        while i < totalApprox {
            let t = CMTime(seconds: Double(i) / nominalFPS, preferredTimescale: 600)
            times.append(t)
            i += stride
        }
        self.timestamps = times
        self.totalFrames = times.count

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = CMTime(seconds: 1.0 / nominalFPS, preferredTimescale: 600)
        self.generator = gen
    }

    /// Returns the next CGImage frame, or nil at end-of-stream.
    func nextFrame() -> CGImage? {
        guard index < timestamps.count else { return nil }
        let t = timestamps[index]
        index += 1
        do {
            let cg = try generator.copyCGImage(at: t, actualTime: nil)
            return cg
        } catch {
            return nil
        }
    }

    func release() {
        generator.cancelAllCGImageGeneration()
    }
}
