import SwiftUI
import Foundation
import CoreGraphics
import AppKit
import MLXRFDETR

/// Renders an image with RF-DETR detection boxes (and optional segmentation masks).
struct DetectionOverlayView: View {
    let cgImage: CGImage
    let result: DetectionResult?

    static let palette: [SIMD4<Float>] = [
        SIMD4(0, 0.8, 1, 1),
        SIMD4(1, 0.4, 0, 1),
        SIMD4(0.4, 1, 0.2, 1),
        SIMD4(1, 0.8, 0, 1),
        SIMD4(0.8, 0.2, 0.8, 1),
        SIMD4(0.2, 0.6, 1, 1),
        SIMD4(1, 0.2, 0.4, 1),
        SIMD4(0.4, 0.8, 0.6, 1),
    ]

    var body: some View {
        let imgW = CGFloat(cgImage.width)
        let imgH = CGFloat(cgImage.height)

        GeometryReader { geo in
            let scale = min(geo.size.width / imgW, geo.size.height / imgH)
            let drawW = imgW * scale
            let drawH = imgH * scale
            let offsetX = (geo.size.width - drawW) / 2
            let offsetY = (geo.size.height - drawH) / 2

            ZStack {
                Image(cgImage, scale: 1.0, label: Text("Source"))
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                if let result, let masksImage = maskOverlayImage(result: result) {
                    Image(masksImage, scale: 1.0, label: Text("Masks"))
                        .resizable()
                        .frame(width: drawW, height: drawH)
                        .position(x: geo.size.width / 2, y: geo.size.height / 2)
                        .opacity(0.5)
                        .allowsHitTesting(false)
                }

                if let result {
                    Canvas { context, _ in
                        drawBoxes(in: context, result: result, scale: scale, offset: CGPoint(x: offsetX, y: offsetY))
                    }
                    .allowsHitTesting(false)
                }
            }
        }
    }

    private func drawBoxes(in context: GraphicsContext, result: DetectionResult, scale: CGFloat, offset: CGPoint) {
        for i in 0..<result.count {
            let b = result.boxes[i]
            let rect = CGRect(
                x: CGFloat(b[0]) * scale + offset.x,
                y: CGFloat(b[1]) * scale + offset.y,
                width: CGFloat(b[2] - b[0]) * scale,
                height: CGFloat(b[3] - b[1]) * scale
            )
            let color = colorAt(i)
            context.stroke(Path(rect), with: .color(color), lineWidth: 2)
            let label = result.classNames[i]
            let conf = result.scores[i]
            drawTag(text: "\(label) \(String(format: "%.2f", conf))", at: rect.origin, color: color, in: context)
        }
    }

    /// Composite per-instance mask logits (sigmoid > 0.5) into a single CGImage at mask resolution.
    private func maskOverlayImage(result: DetectionResult) -> CGImage? {
        guard let masks = result.masks, !masks.isEmpty else { return nil }
        let h = masks[0].rows
        let w = masks[0].cols
        guard h > 0, w > 0 else { return nil }

        var pixels = [UInt8](repeating: 0, count: h * w * 4)
        let palette = Self.palette
        for (i, mask) in masks.enumerated() {
            let color = palette[i % palette.count]
            let r = UInt8(color.x * 255)
            let g = UInt8(color.y * 255)
            let b = UInt8(color.z * 255)
            for px in 0..<(h * w) {
                let v = mask.data[px]
                let s = 1 / (1 + Foundation.exp(-v))
                if s < 0.5 { continue }
                let pi = px * 4
                pixels[pi]     = r
                pixels[pi + 1] = g
                pixels[pi + 2] = b
                pixels[pi + 3] = 180
            }
        }

        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(
            width: w,
            height: h,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: cs,
            bitmapInfo: CGBitmapInfo(rawValue: info),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    private func colorAt(_ index: Int) -> Color {
        let v = Self.palette[index % Self.palette.count]
        return Color(red: Double(v.x), green: Double(v.y), blue: Double(v.z), opacity: Double(v.w))
    }

    private func drawTag(text: String, at origin: CGPoint, color: Color, in context: GraphicsContext) {
        let fontSize: CGFloat = 11
        let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .semibold)
        let textWidth = text.size(withAttributes: [.font: font]).width + 6
        let textHeight: CGFloat = fontSize + 4
        let tag = CGRect(x: origin.x, y: origin.y - textHeight, width: textWidth, height: textHeight)
        context.fill(Path(tag), with: .color(color))
        context.draw(
            Text(text).font(.system(size: fontSize, weight: .semibold)).foregroundStyle(.white),
            at: CGPoint(x: tag.midX, y: tag.midY)
        )
    }
}
