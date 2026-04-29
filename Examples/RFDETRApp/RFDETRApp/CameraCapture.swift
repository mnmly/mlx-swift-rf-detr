import AVFoundation
import CoreImage
import CoreGraphics
import Foundation

/// Drives an `AVCaptureSession` and emits a `CGImage` per frame to a callback.
/// Backpressure: drops frames if the consumer hasn't completed the previous call.
nonisolated final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "rfdetrapp.camera.capture")
    private let ciContext = CIContext()
    private var output: AVCaptureVideoDataOutput?
    private var device: AVCaptureDevice?
    private var inFlight = false
    private let inFlightLock = NSLock()

    var onFrame: ((CGImage) -> Void)?

    var availableDevices: [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video,
            position: .unspecified
        ).devices
    }

    var currentDeviceID: String? { device?.uniqueID }

    func authorize() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func start(deviceID: String? = nil) throws {
        session.beginConfiguration()
        session.sessionPreset = .high

        for input in session.inputs { session.removeInput(input) }
        for out in session.outputs { session.removeOutput(out) }

        let candidate: AVCaptureDevice? = {
            if let id = deviceID, let d = AVCaptureDevice(uniqueID: id) { return d }
            return AVCaptureDevice.default(for: .video)
        }()
        guard let dev = candidate else {
            session.commitConfiguration()
            throw CaptureError.noDevice
        }

        let input = try AVCaptureDeviceInput(device: dev)
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddInput
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddOutput
        }
        session.addOutput(output)

        self.device = dev
        self.output = output
        session.commitConfiguration()

        if !session.isRunning {
            session.startRunning()
        }
    }

    func stop() {
        if session.isRunning { session.stopRunning() }
    }

    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        inFlightLock.lock()
        if inFlight {
            inFlightLock.unlock()
            return
        }
        inFlight = true
        inFlightLock.unlock()

        defer {
            inFlightLock.lock()
            inFlight = false
            inFlightLock.unlock()
        }

        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ci = CIImage(cvPixelBuffer: pb)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return }
        onFrame?(cg)
    }

    enum CaptureError: LocalizedError {
        case noDevice, cannotAddInput, cannotAddOutput, denied

        var errorDescription: String? {
            switch self {
            case .noDevice: "No camera device available."
            case .cannotAddInput: "Capture session refused the camera input."
            case .cannotAddOutput: "Capture session refused the video output."
            case .denied: "Camera access denied. Enable it in System Settings → Privacy & Security → Camera."
            }
        }
    }
}
