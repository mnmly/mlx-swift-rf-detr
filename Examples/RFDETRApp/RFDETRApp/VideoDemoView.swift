import SwiftUI
import CoreGraphics
import UniformTypeIdentifiers
import MLXRFDETR

struct VideoDemoView: View {
    let model: RFDETRViewModel

    @State private var sourceURL: URL?
    @State private var frames: [ProcessedFrame] = []
    @State private var currentFrameIndex: Int = 0
    @State private var isProcessing = false
    @State private var scoreThreshold: Float = 0.5
    @State private var errorMessage: String?
    @State private var isPlaying = false
    @State private var playTimer: Timer?
    @State private var progress: Double = 0
    @State private var fps: Double = 15
    @State private var vidStride: Int = 1
    @State private var totalFrames: Int = 0
    @State private var processingTask: Task<Void, Never>?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            videoArea
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Video").font(.title2.bold())

            Button("Select Video\u{2026}") { selectVideo() }

            if sourceURL != nil {
                Text("Score threshold: \(scoreThreshold, specifier: "%.2f")").font(.caption)
                Slider(value: $scoreThreshold, in: 0.01...1.0, step: 0.01)

                Picker("Frame stride", selection: $vidStride) {
                    ForEach([1, 2, 3, 5, 10], id: \.self) { s in
                        Text(s == 1 ? "every frame" : "every \(s)").tag(s)
                    }
                }

                Picker("Playback FPS", selection: $fps) {
                    ForEach([5.0, 10.0, 15.0, 24.0, 30.0], id: \.self) { f in
                        Text("\(Int(f)) fps").tag(f)
                    }
                }

                HStack {
                    Button("Process") { runProcessing() }
                        .disabled(!model.isLoaded || isProcessing)
                    if isProcessing {
                        Button("Cancel", role: .destructive) { cancelProcessing() }
                    }
                }

                if isProcessing {
                    ProgressView(value: progress) { Text("Processing\u{2026}") }
                    Text("\(Int(progress * 100))%  (\(currentFrameIndex)/\(totalFrames))")
                        .font(.caption2)
                }

                if !frames.isEmpty {
                    Divider()
                    HStack {
                        Button(isPlaying ? "Pause" : "Play") { togglePlayback() }
                        Button("Reset") { resetPlayback() }
                    }
                    if frames.count > 1 {
                        Slider(
                            value: Binding(
                                get: { Double(currentFrameIndex) },
                                set: { currentFrameIndex = Int($0) }
                            ),
                            in: 0...Double(frames.count - 1),
                            step: 1
                        )
                    }
                    Text("Frame \(currentFrameIndex + 1) of \(frames.count)").font(.caption)

                    if currentFrameIndex < frames.count {
                        Divider()
                        ResultsSummary(result: frames[currentFrameIndex].result)
                    }
                }
            }

            if let errorMessage {
                Text(errorMessage).font(.caption).foregroundStyle(.red)
            }
            Spacer()
        }
        .padding(12)
        .frame(width: 240)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    private var videoArea: some View {
        if !frames.isEmpty {
            let frame = frames[currentFrameIndex]
            DetectionOverlayView(cgImage: frame.image, result: frame.result)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "No Video Selected",
                systemImage: "film.stack",
                description: Text("Select a video to run inference")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func selectVideo() {
        let panel = NSOpenPanel()
        panel.title = "Select Video"
        panel.allowedContentTypes = [.movie, .video]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        sourceURL = url
        frames = []
        currentFrameIndex = 0
        errorMessage = nil
    }

    private func runProcessing() {
        guard let url = sourceURL else { return }
        isProcessing = true
        frames = []
        currentFrameIndex = 0
        errorMessage = nil
        progress = 0

        let stride = vidStride
        let threshold = scoreThreshold

        processingTask = Task {
            do {
                let source = try await VideoSource(url: url, vidStride: stride)
                let total = source.totalFrames
                await MainActor.run { totalFrames = total }

                var collected: [ProcessedFrame] = []
                var idx = 0
                var cancelled = false

                while let frame = source.nextFrame() {
                    if Task.isCancelled { cancelled = true; break }
                    let result = try await model.predictAsync(frame, scoreThreshold: threshold)
                    collected.append(ProcessedFrame(image: frame, result: result))
                    idx += 1

                    let p = total > 0 ? Double(idx) / Double(total) : 0
                    if idx % 3 == 0 || idx == 1 {
                        let snapshot = collected
                        let lastIdx = idx - 1
                        await MainActor.run {
                            progress = p
                            currentFrameIndex = lastIdx
                            frames = snapshot
                        }
                    }
                }
                source.release()

                let finalFrames = collected
                let wasCancelled = cancelled
                await MainActor.run {
                    frames = finalFrames
                    currentFrameIndex = 0
                    isProcessing = false
                    progress = wasCancelled ? progress : 1
                    processingTask = nil
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isProcessing = false
                    processingTask = nil
                }
            }
        }
    }

    private func cancelProcessing() {
        processingTask?.cancel()
    }

    private func togglePlayback() {
        if isPlaying {
            playTimer?.invalidate()
            playTimer = nil
            isPlaying = false
        } else {
            isPlaying = true
            let interval = fps > 0 ? 1.0 / fps : 1.0 / 15.0
            playTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { _ in
                Task { @MainActor in
                    if currentFrameIndex < frames.count - 1 {
                        currentFrameIndex += 1
                    } else {
                        isPlaying = false
                        playTimer?.invalidate()
                        playTimer = nil
                    }
                }
            }
        }
    }

    private func resetPlayback() {
        playTimer?.invalidate()
        playTimer = nil
        isPlaying = false
        currentFrameIndex = 0
    }
}

struct ProcessedFrame {
    let image: CGImage
    let result: DetectionResult
}
