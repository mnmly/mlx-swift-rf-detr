import SwiftUI

@main
struct RFDETRAppApp: App {
    @State private var model = RFDETRViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1100, height: 720)
    }
}
