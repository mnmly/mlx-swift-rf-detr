import Foundation

enum DownloadError: LocalizedError {
    case http(Int)
    var errorDescription: String? {
        switch self {
        case .http(let code): "Server returned HTTP \(code)."
        }
    }
}

/// Downloads a single file to `destination`, reporting fractional progress.
/// Uses a download-task delegate so the large `model.safetensors` streams to disk
/// with progress instead of buffering in memory.
nonisolated final class FileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var continuation: CheckedContinuation<Void, Error>?
    private let destination: URL
    private let onProgress: @Sendable (Double) -> Void

    init(destination: URL, onProgress: @escaping @Sendable (Double) -> Void) {
        self.destination = destination
        self.onProgress = onProgress
        super.init()
    }

    func run(from url: URL) async throws {
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.continuation = cont
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            finish(.failure(DownloadError.http(http.statusCode)))
            return
        }
        do {
            let fm = FileManager.default
            try? fm.removeItem(at: destination)
            try fm.moveItem(at: location, to: destination)
        } catch {
            finish(.failure(error))
        }
        // Success resumes in didCompleteWithError (fires after this with nil error).
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        finish(error.map { .failure($0) } ?? .success(()))
    }

    /// Serialized by URLSession's delegate queue, so first-wins is race-free.
    private func finish(_ result: Result<Void, Error>) {
        guard let cont = continuation else { return }
        continuation = nil
        cont.resume(with: result)
    }
}

enum ModelFetcher {
    /// Downloads `model`'s three files into `parent/rfdetr-<id>-mlx` and returns that
    /// directory. The caller must hold `parent`'s security scope (started before this
    /// call and kept active through the subsequent load + bookmark).
    static func fetch(_ model: RemoteModel, intoParent parent: URL,
                      onProgress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let fm = FileManager.default
        let dir = parent.appendingPathComponent("rfdetr-\(model.id)-mlx", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let files = model.files
        for (index, file) in files.enumerated() {
            let isLast = index == files.count - 1   // safetensors dominates size → drives the bar
            let dest = dir.appendingPathComponent(file)

            // Skip files already present. The destination only ever holds a complete
            // file — FileDownloader moves the temp file into place atomically once the
            // download finishes — so a non-empty file here is a finished download.
            if let size = try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int, size > 0 {
                if isLast { onProgress(1.0) }
                continue
            }

            let downloader = FileDownloader(destination: dest) { frac in
                onProgress(isLast ? frac : 0.0)
            }
            try await downloader.run(from: model.url(for: file))
        }
        onProgress(1.0)
        return dir
    }
}
