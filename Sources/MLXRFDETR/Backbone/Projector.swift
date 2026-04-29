// MultiScaleProjector for RF-DETR.
//
// Mirrors the Python MultiScaleProjector: for each output scale a set of
// per-feature samplers (upsample / identity / downsample) is applied, the
// results are concatenated channel-wise, then refined by a C2f + LayerNorm.
//
// Weight key structure (matches exported safetensors):
//   stages_sampling[scaleIdx][featureIdx][0]  – sampler op (ConvTransposed2d or ConvBN)
//   stages[scaleIdx][0]                       – C2f
//   stages[scaleIdx][1]                       – LayerNorm
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/vision.py

import Foundation
import MLX
import MLXNN

// MARK: - ConvBN

/// Conv2d (no bias) + LayerNorm + SiLU activation.
/// Weight key is `bn` (LayerNorm) matching the checkpoint naming convention.
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

/// Bottleneck block with two 3×3 ConvBN blocks (no residual).
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

/// Cross Stage Partial bottleneck with 2 convolutions (YOLOv8 C2f).
public final class C2f: Module {
    @ModuleInfo(key: "cv1") public var cv1: ConvBN
    @ModuleInfo(key: "m") public var m: [Bottleneck]
    @ModuleInfo(key: "cv2") public var cv2: ConvBN

    public init(inChannels: Int, outChannels: Int, numBottlenecks: Int, bottleneckChannels: Int) {
        self._cv1 = ModuleInfo(
            wrappedValue: ConvBN(inChannels: inChannels, outChannels: outChannels, kernelSize: 1),
            key: "cv1"
        )
        self._m = ModuleInfo(
            wrappedValue: (0..<numBottlenecks).map { _ in Bottleneck(channels: bottleneckChannels) },
            key: "m"
        )
        let concatChannels = outChannels + bottleneckChannels * numBottlenecks
        self._cv2 = ModuleInfo(
            wrappedValue: ConvBN(inChannels: concatChannels, outChannels: outChannels, kernelSize: 1),
            key: "cv2"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = cv1(x)
        let splitDim = y.dim(-1) / 2
        let x2 = y[.ellipsis, splitDim...]

        var outputs = [y]
        var bottleneckOut = x2
        for bottleneck in m {
            bottleneckOut = bottleneck(bottleneckOut)
            outputs.append(bottleneckOut)
        }
        return cv2(concatenated(outputs, axis: -1))
    }
}

// MARK: - FeatureSamplerStep

/// Wraps one optional sampling module stored under key "0" to match the
/// nn.Sequential([module]) structure used by the Python projector.
///
/// - nil  → identity (scale = 1.0, no weights in checkpoint)
/// - ConvTransposed2d → 2× upsample (scale = 2.0 / P3)
/// - ConvBN with stride=2 → 2× downsample (scale = 0.5 / P5)
public final class FeatureSamplerStep: Module {
    @ModuleInfo(key: "op") public var op: Module?

    init(_ module: Module?) {
        self._op = ModuleInfo(wrappedValue: module, key: "op")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let op else { return x }
        if let ct = op as? ConvTransposed2d { return ct(x) }
        if let cb = op as? ConvBN { return cb(x) }
        return x
    }
}

// MARK: - ProjectorStage

/// One C2f + LayerNorm stage. Keys "c2f" and "norm" are remapped from "0"/"1" in the loader.
public final class ProjectorStage: Module {
    @ModuleInfo(key: "c2f") public var c2f: C2f
    @ModuleInfo(key: "norm") public var norm: LayerNorm

    init(inChannels: Int, outChannels: Int, numBottlenecks: Int, bottleneckChannels: Int) {
        self._c2f = ModuleInfo(
            wrappedValue: C2f(
                inChannels: inChannels,
                outChannels: outChannels,
                numBottlenecks: numBottlenecks,
                bottleneckChannels: bottleneckChannels
            ),
            key: "c2f"
        )
        self._norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: outChannels, eps: 1e-6),
            key: "norm"
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(c2f(x))
    }
}

// MARK: - MultiScaleProjector

/// Projects multi-scale backbone features through per-scale sampling + C2f blocks.
///
/// For each output scale:
///   1. Apply per-feature sampler (upsample / identity / downsample).
///   2. Concatenate sampled features channel-wise.
///   3. Refine with C2f + LayerNorm.
///
/// Returns one projected feature map per scale.
public final class MultiScaleProjector: Module {
    /// stages_sampling[scaleIdx][featureIdx] — sampler wrapping 0 or 1 module
    @ModuleInfo(key: "stages_sampling") public var stagesSampling: [[FeatureSamplerStep]]
    /// stages[scaleIdx] — C2f + LayerNorm pair
    @ModuleInfo(key: "stages") public var stages: [ProjectorStage]

    public let scaleFactors: [Float]

    public init(scaleFactors: [Float], inChannelsList: [Int], hiddenDim: Int, numBottlenecks: Int = 3) {
        self.scaleFactors = scaleFactors
        let bottleneckChannels = hiddenDim / 2

        var stagesSamplingBuilt: [[FeatureSamplerStep]] = []
        var stagesBuilt: [ProjectorStage] = []

        for scale in scaleFactors {
            var scaleSamplers: [FeatureSamplerStep] = []
            for inCh in inChannelsList {
                if scale == 2.0 {
                    scaleSamplers.append(FeatureSamplerStep(
                        ConvTransposed2d(
                            inputChannels: inCh,
                            outputChannels: inCh / 2,
                            kernelSize: 2,
                            stride: 2
                        )
                    ))
                } else if scale == 0.5 {
                    scaleSamplers.append(FeatureSamplerStep(
                        ConvBN(inChannels: inCh, outChannels: inCh, kernelSize: 3, stride: 2, padding: 1)
                    ))
                } else {
                    scaleSamplers.append(FeatureSamplerStep(nil))
                }
            }
            stagesSamplingBuilt.append(scaleSamplers)

            let inDim = inChannelsList.reduce(0) { $0 + $1 / max(1, Int(scale)) }
            stagesBuilt.append(
                ProjectorStage(
                    inChannels: inDim,
                    outChannels: hiddenDim,
                    numBottlenecks: numBottlenecks,
                    bottleneckChannels: bottleneckChannels
                )
            )
        }

        self._stagesSampling = ModuleInfo(wrappedValue: stagesSamplingBuilt, key: "stages_sampling")
        self._stages = ModuleInfo(wrappedValue: stagesBuilt, key: "stages")
        super.init()
    }

    /// - Parameter features: list of `(B, h, w, D)` backbone feature maps.
    /// - Returns: list of `(B, hi, wi, hiddenDim)` projected feature maps, one per scale.
    public func callAsFunction(_ features: [MLXArray]) -> [MLXArray] {
        var results: [MLXArray] = []
        for (si, stage) in stages.enumerated() {
            var featFuse: [MLXArray] = []
            for (fi, sampler) in stagesSampling[si].enumerated() {
                featFuse.append(sampler(features[fi]))
            }
            results.append(stage(concatenated(featFuse, axis: -1)))
        }
        return results
    }
}
