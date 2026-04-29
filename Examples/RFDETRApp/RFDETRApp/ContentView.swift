import SwiftUI
import UniformTypeIdentifiers
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
    }

    private var modelBar: some View {
        HStack(spacing: 12) {
            Text("Model:").font(.headline)

            Picker("Variant", selection: $model.variant) {
                ForEach(RFDETRVariant.allCases) { v in
                    Text(v.label).tag(v)
                }
            }
            .frame(width: 160)
            .labelsHidden()
            .onChange(of: model.variant) { _, _ in
                Task { await model.reloadIfNeeded() }
            }

            Toggle("Segmentation", isOn: $model.enableSegmentation)
                .toggleStyle(.checkbox)
                .onChange(of: model.enableSegmentation) { _, _ in
                    Task { await model.reloadIfNeeded() }
                }

            Button("Select Weights\u{2026}") { selectWeights() }

            if model.isLoading {
                ProgressView().controlSize(.small)
            }

            if model.isLoaded {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text(URL(fileURLWithPath: model.modelPath).lastPathComponent)
                    .font(.caption).lineLimit(1).truncationMode(.middle)
                Text("res: \(model.processor.resolution)")
                    .font(.caption).foregroundStyle(.secondary)
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

    private func selectWeights() {
        let panel = NSOpenPanel()
        panel.title = "Select RF-DETR Weights File"
        if let st = UTType(filenameExtension: "safetensors") {
            panel.allowedContentTypes = [st]
        }
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await model.loadModel(from: url) }
    }
}
