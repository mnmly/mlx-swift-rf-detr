// MultiScaleProjector (C2f-based neck) for merging backbone feature maps.
//
// Concatenates multi-scale backbone features then projects through a C2f
// (Cross Stage Partial bottleneck) block into a single feature map.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/vision.py (ConvBN, Bottleneck, C2f, MultiScaleProjector)

import Foundation
import MLX
import MLXNN

// MARK: - ConvBN

/// Conv2d (no bias) + LayerNorm + SiLU activation.
/// Weight key is `bn` (LayerNorm) matching the checkpoint naming convention
/// despite being a LayerNorm rather than BatchNorm.
public final class ConvBN: Module {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    @ModuleInfo(key: "bn") public var bn: LayerNorm

    public init(inChannels: Int, outChannels: Int, kernelSize: Int = 1, stride: Int = 1, padding: Int = 0) {
        self._conv = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: .init(kernelSize),
                stride: .init(stride),
                padding: .init(padding),
                bias: false
            ),
            key: "conv"
        )
        self._bn = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: outChannels, eps: 1e-6),
            key: "bn"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(bn(conv(x)))
    }
}

// MARK: - Bottleneck

/// Bottleneck block with two 3x3 ConvBN blocks. No residual (shortcut=False).
public final class Bottleneck: Module {
    @ModuleInfo(key: "cv1") public var cv1: ConvBN
    @ModuleInfo(key: "cv2") public var cv2: ConvBN

    public init(channels: Int) {
        self._cv1 = ModuleInfo(
            wrappedValue: ConvBN(inChannels: channels, outChannels: channels, kernelSize: 3, padding: 1),
            key: "cv1"
        )
        self._cv2 = ModuleInfo(
            wrappedValue: ConvBN(inChannels: channels, outChannels: channels, kernelSize: 3, padding: 1),
            key: "cv2"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        cv2(cv1(x))
    }
}

// MARK: - C2f

/// Cross Stage Partial bottleneck with 2 convolutions (C2f from YOLOv8).
///
/// cv1 (ConvBN) reduces channels then splits: first half is passed through
/// directly, second half goes through bottleneck modules. All outputs
/// (cv1_full, bottleneck_0, bottleneck_1, ...) are concatenated then
/// compressed by cv2.
public final class C2f: Module {
    @ModuleInfo(key: "cv1") public var cv1: ConvBN
    @ModuleInfo(key: "m") public var m: [Bottleneck]
    @ModuleInfo(key: "cv2") public var cv2: ConvBN

    public init(inChannels: Int, outChannels: Int, numBottlenecks: Int, bottleneckChannels: Int) {
        // cv1 reduces to outChannels
        self._cv1 = ModuleInfo(
            wrappedValue: ConvBN(inChannels: inChannels, outChannels: outChannels, kernelSize: 1),
            key: "cv1"
        )
        // m: bottleneck modules
        self._m = ModuleInfo(
            wrappedValue: (0..<numBottlenecks).map { _ in Bottleneck(channels: bottleneckChannels) },
            key: "m"
        )
        // cv2: concat outChannels (from cv1) + bottleneckChannels * numBottlenecks → outChannels
        let concatChannels = outChannels + bottleneckChannels * numBottlenecks
        self._cv2 = ModuleInfo(
            wrappedValue: ConvBN(inChannels: concatChannels, outChannels: outChannels, kernelSize: 1),
            key: "cv2"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // cv1
        let y = cv1(x)
        let splitDim = y.dim(-1) / 2
        let x1 = y[.ellipsis, ..<splitDim]  // first half
        let x2 = y[.ellipsis, splitDim...]  // second half

        var outputs = [y]  // full cv1 output first
        var bottleneckOut = x2
        for bottleneck in m {
            bottleneckOut = bottleneck(bottleneckOut)
            outputs.append(bottleneckOut)
        }

        return cv2(concatenated(outputs, axis: -1))
    }
}

// MARK: - MultiScaleProjector

/// Projects concatenated multi-scale backbone features through C2f + LayerNorm.
public final class MultiScaleProjector: Module {
    @ModuleInfo(key: "c2f") public var c2f: C2f
    @ModuleInfo(key: "final_norm") public var finalNorm: LayerNorm

    public init(inChannels: Int, hiddenDim: Int, numBottlenecks: Int = 3, bottleneckChannels: Int = 128) {
        self._c2f = ModuleInfo(
            wrappedValue: C2f(
                inChannels: inChannels,
                outChannels: hiddenDim,
                numBottlenecks: numBottlenecks,
                bottleneckChannels: bottleneckChannels
            ),
            key: "c2f"
        )
        self._finalNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: hiddenDim, eps: 1e-6),
            key: "final_norm"
        )
        super.init()
    }

    /// - Parameter features: list of `(B, h, w, D)` feature maps from backbone.
    /// - Returns: `(B, h, w, hiddenDim)` projected features.
    public func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        var x = concatenated(features, axis: -1)
        x = c2f(x)
        return finalNorm(x)
    }
}
