// DINOv2 backbone with windowed attention.
//
// Supports all RF-DETR detection model variants:
//  - ViT-S (384d, 6 heads): Nano, Small, Medium, Base, Large
//  - ViT-B (768d, 12 heads): Large (deprecated)
//
// Windowed attention pattern: layers in `feature_indices` (default 2,5,8,11)
// run global attention; all other layers use windowed attention with a
// configurable window grid size (num_windows × num_windows).
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/backbone.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - PatchEmbed

/// Convert image patches to embeddings via strided convolution.
/// Input is NHWC (MLX-native); output is `(patches, gridH, gridW)`.
public final class PatchEmbed: Module {
    @ModuleInfo(key: "proj") public var proj: Conv2d

    public let patchSize: Int
    public let numPatches: Int

    public init(imgSize: Int = 640, patchSize: Int = 16, inChans: Int = 3, embedDim: Int = 384) {
        self.patchSize = patchSize
        self.numPatches = (imgSize / patchSize) * (imgSize / patchSize)
        self._proj = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inChans,
                outputChannels: embedDim,
                kernelSize: IntOrPair(patchSize),
                stride: IntOrPair(patchSize)
            ),
            key: "proj"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, Int, Int) {
        let y = proj(x)
        let N = y.dim(0); let H = y.dim(1); let W = y.dim(2); let C = y.dim(3)
        return (y.reshaped([N, H * W, C]), H, W)
    }
}

// MARK: - LayerScale

/// Learnable per-channel scale on residuals.
public final class LayerScale: Module {
    @ParameterInfo(key: "gamma") public var gamma: MLXArray

    public init(dim: Int) {
        self._gamma = ParameterInfo(wrappedValue: MLXArray.ones([dim]), key: "gamma")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

// MARK: - Attention

/// Multi-head self-attention with separate Q/K/V projections.
public final class Attention: Module {
    @ModuleInfo(key: "q") public var q: Linear
    @ModuleInfo(key: "k") public var k: Linear
    @ModuleInfo(key: "v") public var v: Linear
    @ModuleInfo(key: "out") public var out: Linear

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(dim: Int, numHeads: Int = 6, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / sqrt(Float(self.headDim))
        self._q = ModuleInfo(wrappedValue: Linear(dim, dim, bias: qkvBias), key: "q")
        self._k = ModuleInfo(wrappedValue: Linear(dim, dim, bias: qkvBias), key: "k")
        self._v = ModuleInfo(wrappedValue: Linear(dim, dim, bias: qkvBias), key: "v")
        self._out = ModuleInfo(wrappedValue: Linear(dim, dim), key: "out")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let N = x.dim(0); let L = x.dim(1); let C = x.dim(2)
        let H = numHeads; let D = headDim

        let qx = q(x).reshaped([N, L, H, D]).transposed(0, 2, 1, 3)
        let kx = k(x).reshaped([N, L, H, D]).transposed(0, 2, 1, 3)
        let vx = v(x).reshaped([N, L, H, D]).transposed(0, 2, 1, 3)

        var o = MLXFast.scaledDotProductAttention(
            queries: qx, keys: kx, values: vx, scale: scale, mask: nil
        )
        o = o.transposed(0, 2, 1, 3).reshaped([N, L, C])
        return out(o)
    }
}

// MARK: - MLP

/// Two-layer MLP with GELU activation.
public final class MLP: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inFeatures: Int, hiddenFeatures: Int, outFeatures: Int) {
        self._fc1 = ModuleInfo(wrappedValue: Linear(inFeatures, hiddenFeatures), key: "fc1")
        self._fc2 = ModuleInfo(wrappedValue: Linear(hiddenFeatures, outFeatures), key: "fc2")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

// MARK: - Block

/// Transformer block with layer scale.
///
/// Windowing is handled at the backbone level. When `runFullAttention` is true
/// the block merges windows before attention then unmerges; otherwise it
/// operates on whatever sequence it receives.
public final class Block: Module {
    @ModuleInfo(key: "norm1") public var norm1: LayerNorm
    @ModuleInfo(key: "attn") public var attn: Attention
    @ModuleInfo(key: "ls1") public var ls1: LayerScale
    @ModuleInfo(key: "norm2") public var norm2: LayerNorm
    @ModuleInfo(key: "mlp") public var mlp: MLP
    @ModuleInfo(key: "ls2") public var ls2: LayerScale

    public let numWindows: Int

    public init(dim: Int, numHeads: Int, numWindows: Int, mlpRatio: Float = 4.0) {
        self.numWindows = numWindows
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: 1e-6), key: "norm1")
        self._attn = ModuleInfo(wrappedValue: Attention(dim: dim, numHeads: numHeads), key: "attn")
        self._ls1 = ModuleInfo(wrappedValue: LayerScale(dim: dim), key: "ls1")
        self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: 1e-6), key: "norm2")
        self._mlp = ModuleInfo(
            wrappedValue: MLP(inFeatures: dim, hiddenFeatures: Int(Float(dim) * mlpRatio), outFeatures: dim),
            key: "mlp"
        )
        self._ls2 = ModuleInfo(wrappedValue: LayerScale(dim: dim), key: "ls2")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, runFullAttention: Bool = false) -> MLXArray {
        let nW2 = numWindows * numWindows

        let attnOut: MLXArray
        if runFullAttention && nW2 > 1 {
            let B = x.dim(0); let HW = x.dim(1); let C = x.dim(2)
            let merged = x.reshaped([B / nW2, nW2 * HW, C])
            let a = attn(norm1(merged))
            attnOut = a.reshaped([B, HW, C])
        } else {
            attnOut = attn(norm1(x))
        }

        var y = x + ls1(attnOut)
        y = y + ls2(mlp(norm2(y)))
        return y
    }
}

// MARK: - DINOv2Backbone

/// DINOv2 backbone with windowed attention for RF-DETR.
public final class DINOv2Backbone: Module {
    @ModuleInfo(key: "patch_embed") public var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token") public var clsToken: MLXArray
    @ParameterInfo(key: "register_tokens") public var registerTokens: MLXArray
    @ParameterInfo(key: "pos_embed") public var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") public var blocks: [Block]
    @ModuleInfo(key: "norm") public var norm: LayerNorm

    public let patchSize: Int
    public let embedDim: Int
    public let numWindows: Int
    public let depth: Int
    public let featureIndices: [Int]
    public let fullAttnLayers: Set<Int>
    public let numRegisterTokens: Int

    public init(
        imgSize: Int = 640,
        patchSize: Int = 16,
        embedDim: Int = 384,
        depth: Int = 12,
        numHeads: Int = 6,
        numWindows: Int = 2,
        featureIndices: [Int]? = nil,
        mlpRatio: Float = 4.0,
        numRegisterTokens: Int = 0
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numWindows = numWindows
        self.depth = depth
        self.numRegisterTokens = numRegisterTokens

        let fi = featureIndices ?? [2, 5, 8, 11]
        self.featureIndices = fi
        // Full attention at the same layers where features are extracted.
        // Mirrors rf-detr-mlx: window_block_indexes excludes out_feature_indexes,
        // so those layers always run global attention.
        self.fullAttnLayers = Set(fi)

        let pe = PatchEmbed(imgSize: imgSize, patchSize: patchSize, inChans: 3, embedDim: embedDim)
        self._patchEmbed = ModuleInfo(wrappedValue: pe, key: "patch_embed")
        let numPatches = pe.numPatches

        self._clsToken = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, embedDim]), key: "cls_token")
        self._registerTokens = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, max(numRegisterTokens, 1), embedDim]),
            key: "register_tokens"
        )
        self._posEmbed = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, 1 + numPatches, embedDim]),
            key: "pos_embed"
        )

        let built = (0..<depth).map { _ in
            Block(dim: embedDim, numHeads: numHeads, numWindows: numWindows, mlpRatio: mlpRatio)
        }
        self._blocks = ModuleInfo(wrappedValue: built, key: "blocks")
        self._norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: embedDim, eps: 1e-6), key: "norm")
        super.init()
    }

    /// (N, H*W, C) → (N*nW^2, win_h*win_w, C).
    func windowPartition(_ patches: MLXArray, H: Int, W: Int, N: Int) -> MLXArray {
        let nW = numWindows
        let C = patches.dim(2)
        let winH = H / nW
        let winW = W / nW
        var p = patches.reshaped([N, nW, winH, nW, winW, C])
        p = p.transposed(0, 1, 3, 2, 4, 5).reshaped([N * nW * nW, winH * winW, C])
        return p
    }

    /// (N*nW^2, win_h*win_w, C) → (N, H, W, C).
    private func unwindow(_ patches: MLXArray, H: Int, W: Int, N: Int) -> MLXArray {
        let nW = numWindows
        let nW2 = nW * nW
        let C = patches.dim(2)
        let winH = H / nW
        let winW = W / nW
        var p = patches.reshaped([N * nW2, winH * winW, C])
        p = p.reshaped([N, nW2 * winH * winW, C])
        p = p.reshaped([N * nW, nW, winH, winW, C])
        p = p.transposed(0, 2, 1, 3, 4)
        p = p.reshaped([N, H, W, C])
        return p
    }

    /// Forward pass extracting multi-scale features.
    /// - Parameter x: input image (N, H, W, C) NHWC, normalized.
    /// - Returns: feature maps at configured layer indices, each (N, gridH, gridW, embedDim).
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let N = x.dim(0)
        let (patches, H, W) = patchEmbed(x)

        let cls = MLX.broadcast(clsToken, to: [N, 1, embedDim])
        var tokens = MLX.concatenated([cls, patches], axis: 1) + posEmbed

        let nW = numWindows
        let nW2 = nW * nW

        let clsTokenSlice = tokens[0..., 0..<1, 0...]
        let patchTokens = tokens[0..., 1..., 0...]
        let winPatches = windowPartition(patchTokens, H: H, W: W, N: N)
        let winClsBase = MLX.broadcast(clsTokenSlice, to: [N, 1, embedDim])
        let winCls = MLX.concatenated(Array(repeating: winClsBase, count: nW2), axis: 0)
        tokens = MLX.concatenated([winCls, winPatches], axis: 1)

        // Insert register tokens between CLS and patches: [CLS, REG×R, patches].
        // Registers are added AFTER windowing (per Python: same registers replicated
        // across all nW² windows via expand on the batch axis).
        let nR = numRegisterTokens
        if nR > 0 {
            let cls = tokens[0..., 0..<1, 0...]
            let rest = tokens[0..., 1..., 0...]
            let regs = MLX.broadcast(registerTokens, to: [N * nW2, nR, embedDim])
            tokens = MLX.concatenated([cls, regs, rest], axis: 1)
        }

        var features: [MLXArray] = []
        features.reserveCapacity(featureIndices.count)
        let featureSet = Set(featureIndices)

        // Patches start at index (1 + nR): skip CLS and registers when extracting.
        let patchStart = 1 + nR

        for (i, block) in blocks.enumerated() {
            let runFull = fullAttnLayers.contains(i)
            tokens = block(tokens, runFullAttention: runFull)

            if featureSet.contains(i) {
                let normed = norm(tokens)
                let patchOnly = normed[0..., patchStart..., 0...]
                features.append(unwindow(patchOnly, H: H, W: W, N: N))
            }
        }

        return features
    }
}
