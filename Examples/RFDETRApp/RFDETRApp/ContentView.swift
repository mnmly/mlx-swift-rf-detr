import SwiftUI
import MLXRFDETR

struct ContentView: View {
    @Bindable var model: RFDETRViewModel
    @State private var selectedTab: Tab = .image

    enum Tab: String, CaseIterable, Identifiable {
        case image, video, camera
        var id: String { rawValue }
        var label: String {
            switch self {
            case .image: "Image"
            case .video: "Video"
            case .camera: "Camera"
            }
        }
        var systemImage: String {
            switch self {
            case .image: "photo"
            case .video: "film"
            case .camera: "camera"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            modelBar
            Divider()
            TabView(selection: $selectedTab) {
                ImageDemoView(model: model)
                    .tabItem { Label(Tab.image.label, systemImage: Tab.image.systemImage) }
                    .tag(Tab.image)

                VideoDemoView(model: model)
                    .tabItem { Label(Tab.video.label, systemImage: Tab.video.systemImage) }
                    .tag(Tab.video)

                CameraDemoView(model: model)
                    .tabItem { Label(Tab.camera.label, systemImage: Tab.camera.systemImage) }
                    .tag(Tab.camera)
            }
        }
        .task { await model.restoreLastModel() }
    }

    private var modelBar: some View {
        HStack(spacing: 12) {
            Text("Model:").font(.headline)

            Button("Select Model Directory\u{2026}") { selectModelDirectory() }
                .disabled(model.downloadingName != nil)

            Menu("Download Model\u{2026}") {
                ForEach(ModelCatalog.all) { m in
                    Button("\(m.id) — \(m.subtitle)") { startDownload(m) }
                }
            }
            .fixedSize()
            .disabled(model.isLoading || model.downloadingName != nil)

            if let name = model.downloadingName, let p = model.downloadProgress {
                ProgressView(value: p).frame(width: 110)
                Text("\(name) \(Int(p * 100))%")
                    .font(.caption).monospacedDigit().foregroundStyle(.secondary)
            } else if model.isLoading {
                ProgressView().controlSize(.small)
            }

            if model.isLoaded {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text(URL(fileURLWithPath: model.modelPath).lastPathComponent)
                    .font(.caption).lineLimit(1).truncationMode(.middle)
                if let v = model.variant {
                    Text(v.rawValue).font(.caption).foregroundStyle(.secondary)
                }
                Text("res: \(model.resolution)")
                    .font(.caption).foregroundStyle(.secondary)
                if model.hasSegmentation {
                    Text("seg").font(.caption).foregroundStyle(.secondary)
                }
                if model.hasKeypoints {
                    Text("pose").font(.caption).foregroundStyle(.secondary)
                }
            }

            if let err = model.errorMessage {
                Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                Text(err).font(.caption).foregroundStyle(.red).lineLimit(2)
            }

            Spacer()
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    private func selectModelDirectory() {
        let panel = NSOpenPanel()
        panel.title = "Select RF-DETR Model Directory"
        panel.message = "Choose a directory containing config.json and model.safetensors"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await model.loadModel(from: url) }
    }

    private func startDownload(_ remote: RemoteModel) {
        let panel = NSOpenPanel()
        panel.title = "Choose where to download \(remote.id)"
        panel.message = "Pick a folder; the model is saved as rfdetr-\(remote.id)-mlx"
        panel.prompt = "Download Here"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await model.downloadAndLoad(remote, intoParent: url) }
    }
}
