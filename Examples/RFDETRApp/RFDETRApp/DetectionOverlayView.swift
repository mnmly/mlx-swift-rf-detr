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
                        drawKeypoints(in: context, result: result, scale: scale,
                                      offset: CGPoint(x: offsetX, y: offsetY),
                                      imageSize: CGSize(width: imgW, height: imgH))
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

    /// COCO-17 skeleton edges (0-indexed): connects nose/eyes/ears, shoulders/arms,
    /// hips/legs. Used to draw a stick figure over keypoint detections.
    static let cocoSkeleton: [(Int, Int)] = [
        (5, 7), (7, 9), (6, 8), (8, 10),        // arms
        (11, 13), (13, 15), (12, 14), (14, 16), // legs
        (5, 6), (11, 12), (5, 11), (6, 12),     // torso
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), // head/neck
    ]

    /// Draw keypoints (confidence-gated) plus the COCO skeleton per detection,
    /// including per-keypoint uncertainty ellipses from the precision-Cholesky output.
    private func drawKeypoints(in context: GraphicsContext, result: DetectionResult, scale: CGFloat, offset: CGPoint, imageSize: CGSize) {
        guard let keypoints = result.keypoints else { return }
        let precision = result.keypointPrecisionCholesky
        let confThreshold: Float = 0.5
        // Ellipse drawn at this many standard deviations (1σ ≈ 39% mass for 2D Gaussian).
        let nSigma: CGFloat = 2.0
        for i in 0..<min(result.count, keypoints.count) {
            let kp = keypoints[i]            // (maxK, 3): x_px, y_px, confidence
            let prec = (precision != nil && i < precision!.count) ? precision![i] : nil
            let color = colorAt(i)
            func point(_ k: Int) -> CGPoint {
                CGPoint(x: CGFloat(kp[k, 0]) * scale + offset.x, y: CGFloat(kp[k, 1]) * scale + offset.y)
            }

            // Uncertainty ellipses (drawn under the dots/skeleton).
            if let prec {
                for k in 0..<kp.rows where kp[k, 2] >= confThreshold {
                    guard let e = covarianceEllipse(
                        logL11: prec[k, 0], l21: prec[k, 1], logL22: prec[k, 2],
                        width: Double(imageSize.width), height: Double(imageSize.height)
                    ) else { continue }
                    let c = point(k)
                    let ax = CGFloat(e.semiMajor) * nSigma * scale
                    let ay = CGFloat(e.semiMinor) * nSigma * scale
                    // Skip degenerate/huge ellipses that would smear across the image.
                    guard ax.isFinite, ay.isFinite, ax > 0.5, max(ax, ay) < imageSize.width * scale else { continue }
                    let base = Path(ellipseIn: CGRect(x: -ax, y: -ay, width: 2 * ax, height: 2 * ay))
                    let t = CGAffineTransform(translationX: c.x, y: c.y).rotated(by: CGFloat(e.angle))
                    context.stroke(base.applying(t), with: .color(color.opacity(0.45)), lineWidth: 1)
                }
            }
            // Skeleton edges (both endpoints confident); edge opacity tracks the
            // weaker of the two endpoint confidences.
            if kp.rows >= 17 {
                for (a, b) in Self.cocoSkeleton where kp[a, 2] >= confThreshold && kp[b, 2] >= confThreshold {
                    var path = Path()
                    path.move(to: point(a))
                    path.addLine(to: point(b))
                    let edgeConf = min(kp[a, 2], kp[b, 2])
                    context.stroke(path, with: .color(color.opacity(Double(edgeConf))), lineWidth: 2)
                }
            }
            // Keypoint dots: radius and opacity encode per-keypoint confidence.
            for k in 0..<kp.rows where kp[k, 2] >= confThreshold {
                let p = point(k)
                let conf = kp[k, 2]                       // 0…1 (sigmoid of findable logit)
                let radius = 2.0 + 3.0 * CGFloat(conf)    // 2…5 px
                let dot = CGRect(x: p.x - radius, y: p.y - radius, width: radius * 2, height: radius * 2)
                context.fill(Path(ellipseIn: dot), with: .color(.white.opacity(Double(conf))))
                context.stroke(Path(ellipseIn: dot), with: .color(color), lineWidth: 1.5)
                // Confidence label for higher-confidence joints (avoids clutter).
                if conf >= 0.7 {
                    context.draw(
                        Text(String(format: "%.2f", conf)).font(.system(size: 8, weight: .medium)).foregroundStyle(color),
                        at: CGPoint(x: p.x, y: p.y - radius - 6)
                    )
                }
            }
        }
    }

    /// Convert precision-Cholesky params `(log_l11, l21, log_l22)` to a pixel-space
    /// covariance ellipse (semi-axes + rotation). Ports
    /// `rfdetr.utilities.keypoints.precision_cholesky_to_pixel_covariance` followed by a
    /// 2×2 symmetric eigen-decomposition.
    private func covarianceEllipse(
        logL11: Float, l21: Float, logL22: Float, width: Double, height: Double
    ) -> (semiMajor: Double, semiMinor: Double, angle: Double)? {
        let a0 = Double(logL11), a1 = Double(l21), a2 = Double(logL22)
        guard a0.isFinite, a1.isFinite, a2.isFinite else { return nil }
        let l11 = exp(a0), l22 = exp(a2)
        let invDet = 1.0 / (l11 * l11 * l22 * l22)
        guard invDet.isFinite else { return nil }
        // Covariance = precision⁻¹ (normalized coords), scaled to pixels.
        let cov00 = invDet * (a1 * a1 + l22 * l22)
        let cov01 = invDet * (-l11 * a1)
        let cov11 = invDet * (l11 * l11)
        let a = width * width * cov00      // px00
        let b = width * height * cov01     // px01
        let c = height * height * cov11    // px11
        guard a.isFinite, b.isFinite, c.isFinite else { return nil }
        // Eigenvalues of symmetric [[a, b], [b, c]].
        let half = (a + c) / 2
        let common = ((a - c) / 2 * (a - c) / 2 + b * b).squareRoot()
        let lambda1 = half + common
        let lambda2 = max(0, half - common)
        guard lambda1.isFinite, lambda1 > 0 else { return nil }
        let angle = 0.5 * atan2(2 * b, a - c)
        return (lambda1.squareRoot(), lambda2.squareRoot(), angle)
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
