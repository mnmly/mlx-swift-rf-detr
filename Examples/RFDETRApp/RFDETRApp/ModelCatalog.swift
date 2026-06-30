import Foundation

/// A converted RF-DETR model hosted on Hugging Face as the three-file MLX layout
/// (`config.json`, `preprocessor_config.json`, `model.safetensors`) that
/// `RFDETR.load(directory:)` consumes directly.
struct RemoteModel: Identifiable, Hashable, Sendable {
    /// Short variant id; also the local directory stem (`rfdetr-<id>-mlx`).
    let id: String
    /// Hugging Face repo, e.g. `mlx-community/rfdetr-base-fp32`.
    let repo: String
    /// One-line description for the menu.
    let subtitle: String

    var files: [String] { ["config.json", "preprocessor_config.json", "model.safetensors"] }

    func url(for file: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
    }
}

/// Models downloadable from Hugging Face. `mlx-community` repos are produced by the
/// same converter this port targets and load as-is (verified for `base` via
/// `LoaderTests.testLoadAndForwardBase`). The `mnmly/*` repos are the gap variants
/// converted with `Scripts/convert_detection.py` / `convert_keypoint.py` and hosted
/// from this project (Apache-2.0, attributing Roboflow).
enum ModelCatalog {
    static let all: [RemoteModel] = [
        RemoteModel(id: "base",             repo: "mlx-community/rfdetr-base-fp32",        subtitle: "Detection · 560"),
        RemoteModel(id: "small",            repo: "mnmly/rfdetr-small-mlx-fp32",          subtitle: "Detection · 512"),
        RemoteModel(id: "large-2026",       repo: "mnmly/rfdetr-large-2026-mlx-fp32",     subtitle: "Detection · 704"),
        RemoteModel(id: "seg-small",        repo: "mlx-community/rfdetr-seg-small-fp32",   subtitle: "Detection + masks · 384"),
        RemoteModel(id: "seg-large",        repo: "mlx-community/rfdetr-seg-large-fp32",   subtitle: "Detection + masks · 504"),
        RemoteModel(id: "seg-xlarge",       repo: "mlx-community/rfdetr-seg-xlarge-fp32",  subtitle: "Detection + masks · 624"),
        RemoteModel(id: "seg-2xlarge",      repo: "mlx-community/rfdetr-seg-2xlarge-fp32", subtitle: "Detection + masks · 768"),
        RemoteModel(id: "keypoint-preview", repo: "mnmly/rfdetr-keypoint-preview-mlx-fp32", subtitle: "Keypoints (person, 17) · 576"),
    ]
}
