import SwiftUI
import AVFoundation
import CoreGraphics
import RFDETRMLX

struct CameraDemoView: View {
    let model: RFDETRViewModel

    @State private var capture = CameraCapture()
    @State private var currentFrame: CGImage?
    @State private var result: DetectionResult?
    @State private var scoreThreshold: Float = 0.5
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var deviceID: String?
    @State private var devices: [AVCaptureDevice] = []
    @State private var fps: Double = 0
    @State private var lastFrameTime = Date()
    @State private var inferenceInFlight = false

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            cameraArea
        }
        .onDisappear { stop() }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Camera").font(.title2.bold())

            Picker("Device", selection: Binding(
                get: { deviceID ?? "" },
                set: { newID in
                    deviceID = newID.isEmpty ? nil : newID
                    if isRunning { restart() }
                }
            )) {
                Text("Default").tag("")
                ForEach(devices, id: \.uniqueID) { dev in
                    Text(dev.localizedName).tag(dev.uniqueID)
                }
            }

            Text("Score threshold: \(scoreThreshold, specifier: "%.2f")").font(.caption)
            Slider(value: $scoreThreshold, in: 0.01...1.0, step: 0.01)

            HStack {
                Button(isRunning ? "Stop" : "Start") {
                    if isRunning { stop() } else { Task { await start() } }
                }
                .disabled(!model.isLoaded && !isRunning)
            }

            if isRunning {
                Text(String(format: "FPS: %.1f", fps)).font(.caption.monospacedDigit())
            }

            if let result {
                Divider()
                ResultsSummary(result: result)
            }

            if let errorMessage {
                Text(errorMessage).font(.caption).foregroundStyle(.red)
            }
            Spacer()
        }
        .padding(12)
        .frame(width: 240)
        .background(Color(nsColor: .controlBackgroundColor))
        .onAppear { devices = capture.availableDevices }
    }

    @ViewBuilder
    private var cameraArea: some View {
        if let currentFrame {
            DetectionOverlayView(cgImage: currentFrame, result: result)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if isRunning {
            ProgressView("Starting camera\u{2026}")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "Camera Off",
                systemImage: "camera",
                description: Text(model.isLoaded ? "Press Start to begin live inference" : "Load a model first, then press Start")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func start() async {
        guard model.isLoaded else {
            errorMessage = "Load a model first."
            return
        }
        let granted = await capture.authorize()
        guard granted else {
            errorMessage = CameraCapture.CaptureError.denied.localizedDescription
            return
        }

        capture.onFrame = { cg in
            Task { @MainActor in
                handleFrame(cg)
            }
        }

        do {
            try capture.start(deviceID: deviceID)
            isRunning = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            isRunning = false
        }
    }

    private func stop() {
        capture.stop()
        capture.onFrame = nil
        isRunning = false
        inferenceInFlight = false
    }

    private func restart() {
        stop()
        Task { await start() }
    }

    private func handleFrame(_ cg: CGImage) {
        currentFrame = cg

        let now = Date()
        let dt = now.timeIntervalSince(lastFrameTime)
        if dt > 0 { fps = 0.8 * fps + 0.2 * (1.0 / dt) }
        lastFrameTime = now

        if inferenceInFlight { return }
        inferenceInFlight = true

        Task {
            do {
                let r = try await model.predictAsync(cg, scoreThreshold: scoreThreshold)
                result = r
            } catch {
                errorMessage = error.localizedDescription
            }
            inferenceInFlight = false
        }
    }
}
