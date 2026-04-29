// Image preprocessing pipeline for RF-DETR.
//
// Provides image normalization utilities. Actual image loading/resizing
// is platform-specific; the package exposes the normalization parameters
// and a helper that runs the full pipeline from a pre-loaded MLXArray.
//
// Expected input: MLXArray with shape (H, W, 3), uint8 RGB pixel values [0, 255].
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/processing_rfdetr.py

import Foundation
import MLX

/// Default normalization parameters for RF-DETR (ImageNet stats).
public struct RFDETRProcessor {
    public let imageMean: [Float]
    public let imageStd: [Float]
    public let resolution: Int
    public let numSelect: Int

    /// Create a processor with defaults matching the base model.
    public init(
        resolution: Int = 560,
        imageMean: [Float] = [0.485, 0.456, 0.406],
        imageStd: [Float] = [0.229, 0.224, 0.225],
        numSelect: Int = 300
    ) {
        self.resolution = resolution
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.numSelect = numSelect
    }

    /// Normalize a pre-loaded and resized image.
    ///
    /// Input should already be resized to `(resolution, resolution, 3)` and
    /// converted to float32 in [0, 1] range.
    ///
    /// - Parameter pixelValues: `(H, W, 3)` float32 image tensor in [0, 1].
    /// - Returns: `(1, H, W, 3)` normalized float32 batch.
    public func normalize(_ pixelValues: MLXArray) -> MLXArray {
        let mean = MLXArray(imageMean, [1, 1, 3])
        let std = MLXArray(imageStd, [1, 1, 3])
        var x = pixelValues - mean
        x = x / std
        return x.expandedDimensions(axis: 0)  // (1, H, W, 3)
    }
}

#if canImport(AppKit) || canImport(UIKit)

import CoreGraphics
import ImageIO

/// Load an image from a file URL, resize, and normalize for RF-DETR.
///
/// This uses CoreGraphics (available on macOS/iOS). For other platforms,
/// use the `normalize(_:)` method directly with a pre-loaded MLXArray.
///
/// - Parameters:
///   - url: path to image file (PNG, JPEG, etc.)
///   - processor: configured processor (resolution, mean, std)
/// - Returns: `(1, H, W, 3)` normalized float32 batch and the original `(H, W)` size
public func loadAndPreprocess(url: URL, processor: RFDETRProcessor) throws -> (pixelValues: MLXArray, originalSize: (Int, Int)) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw NSError(domain: "MLXRFDETR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot load image at \(url.path)"])
    }

    let origW = cgImage.width
    let origH = cgImage.height
    let res = processor.resolution

    // Resize using CoreGraphics
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(
        data: nil,
        width: res, height: res,
        bitsPerComponent: 8,
        bytesPerRow: res * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    )
    ctx?.interpolationQuality = .default  // bilinear
    ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: res, height: res))

    guard let context = ctx, let data = context.data else {
        throw NSError(domain: "MLXRFDETR", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
    }

    // Copy RGBA → RGB float (normalize to [0, 1])
    var rgb = [Float](repeating: 0, count: res * res * 3)
    let buf = data.bindMemory(to: UInt8.self, capacity: res * res * 4)
    for i in 0..<(res * res) {
        rgb[i * 3] = Float(buf[i * 4]) / 255.0       // R
        rgb[i * 3 + 1] = Float(buf[i * 4 + 1]) / 255.0 // G
        rgb[i * 3 + 2] = Float(buf[i * 4 + 2]) / 255.0 // B
    }

    let tensor = MLXArray(rgb, [res, res, 3])
    let normalized = processor.normalize(tensor)

    return (normalized, (origH, origW))
}

#endif
