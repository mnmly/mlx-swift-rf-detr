import SwiftUI
import CoreGraphics
import UniformTypeIdentifiers
import MLXRFDETR

struct ImageDemoView: View {
    let model: RFDETRViewModel

    @State private var sourceImage: CGImage?
    @State private var result: DetectionResult?
    @State private var isRunning = false
    @State private var scoreThreshold: Float = 0.5
    @State private var errorMessage: String?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            imageArea
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Image").font(.title2.bold())

            Button("Select Image\u{2026}") { selectImage() }

            if sourceImage != nil {
                Text("Score threshold: \(scoreThreshold, specifier: "%.2f")").font(.caption)
                Slider(value: $scoreThreshold, in: 0.01...1.0, step: 0.01)

                Button("Run") { run() }
                    .disabled(!model.isLoaded || isRunning)

                if isRunning { ProgressView("Running\u{2026}") }

                if let result {
                    Divider()
                    ResultsSummary(result: result)
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
    private var imageArea: some View {
        if let sourceImage {
            DetectionOverlayView(cgImage: sourceImage, result: result)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "No Image Selected",
                systemImage: "photo.badge.plus",
                description: Text("Select an image to run inference")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.title = "Select Image"
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let cg = loadCGImage(from: url) else {
            errorMessage = "Failed to load image"
            return
        }
        sourceImage = cg
        result = nil
        errorMessage = nil
    }

    private func run() {
        guard let sourceImage else { return }
        isRunning = true
        errorMessage = nil
        Task {
            do {
                let r = try await model.predictAsync(sourceImage, scoreThreshold: scoreThreshold)
                await MainActor.run {
                    result = r
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isRunning = false
                }
            }
        }
    }
}

func loadCGImage(from url: URL) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}

struct ResultsSummary: View {
    let result: DetectionResult
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Results").font(.headline)
            Text("Detections: \(result.count)").font(.caption)
            if let m = result.masks { Text("Masks: \(m.count)").font(.caption) }
            let groups = Dictionary(grouping: result.labels.indices, by: { result.labels[$0] })
            ForEach(groups.keys.sorted(), id: \.self) { cls in
                let name = cls < result.classNames.count ? result.classNames[groups[cls]!.first!] : "class\(cls)"
                Text("\(name): \(groups[cls]!.count)").font(.caption2)
            }
        }
    }
}
