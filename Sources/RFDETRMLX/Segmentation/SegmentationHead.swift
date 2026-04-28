// RF-DETR Segmentation Head.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/segmentation.py

import Foundation
import MLX
import MLXNN

// MARK: - DepthwiseConvBlock

/// ConvNeXt-style depthwise conv block: dwconv → LayerNorm → pointwise → GELU + residual.
public final class DepthwiseConvBlock: Module {
    @ModuleInfo(key: "dwconv") public var dwconv: Conv2d
    @ModuleInfo(key: "norm") public var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") public var pwconv1: Linear

    public init(dim: Int) {
        self._dwconv = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: dim,
                outputChannels: dim,
                kernelSize: 3,
                padding: 1,
                groups: dim
            ),
            key: "dwconv"
        )
        self._norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: dim, eps: 1e-6),
            key: "norm"
        )
        self._pwconv1 = ModuleInfo(
            wrappedValue: Linear(dim, dim),
            key: "pwconv1"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = dwconv(x)
        y = norm(y)
        y = gelu(pwconv1(y))
        return residual + y
    }
}

// MARK: - MLPBlock

/// MLP block with residual: LayerNorm → Linear(4×) → GELU → Linear → residual.
///
/// Python uses `self.layers = [Linear, None, Linear]` (index 1 = None placeholder for GELU).
/// We mirror with `[Linear?]` and `nil` at index 1 so safetensors key `layers.0.weight` /
/// `layers.2.weight` match the array after `NestedDictionary.unflattened()`.
public final class MLPBlock: Module {
    @ModuleInfo(key: "norm_in") public var normIn: LayerNorm
    @ModuleInfo(key: "layers") public var layers: [Linear?]

    public init(dim: Int) {
        self._normIn = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: dim),
            key: "norm_in"
        )
        self._layers = ModuleInfo(
            wrappedValue: [Linear(dim, dim * 4), nil, Linear(dim * 4, dim)],
            key: "layers"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = normIn(x)
        y = layers[0]!(y) // fc1
        y = gelu(y)
        y = layers[2]!(y) // fc2
        return residual + y
    }
}

// MARK: - Bilinear interpolation for (B, H, W, C) tensors

/// Bilinear interpolation matching the Python `_interpolate_spatial` helper.
///
/// Uses linspace-based pixel coords (no align_corners adjustment), which is
/// equivalent to `align_corners=True` over `[0, H-1]`.
public func interpolateSpatial(_ x: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
    let B = x.dim(0); let H = x.dim(1); let W = x.dim(2); let C = x.dim(3)
    if H == targetH && W == targetW { return x }

    // Coordinates in source pixel space
    let yCoords = MLXArray.linspace(0.0, Float(H - 1), count: targetH) // (targetH,)
    let xCoords = MLXArray.linspace(0.0, Float(W - 1), count: targetW) // (targetW,)

    let yy = broadcast(yCoords.expandedDimensions(axis: 1), to: [targetH, targetW])
    let xx = broadcast(xCoords.expandedDimensions(axis: 0), to: [targetH, targetW])

    let y0 = clip(floor(yy).asType(.int32), min: 0, max: H - 1)
    let y1 = clip(y0 + 1, min: 0, max: H - 1)
    let x0 = clip(floor(xx).asType(.int32), min: 0, max: W - 1)
    let x1 = clip(x0 + 1, min: 0, max: W - 1)

    let fy = (yy - y0.asType(yy.dtype)).expandedDimensions(axis: -1) // (tH, tW, 1)
    let fx = (xx - x0.asType(xx.dtype)).expandedDimensions(axis: -1)

    // Gather x[:, y_i, x_j, :] for each (i, j) — flatten (tH, tW) into a single index list.
    // Index into the H×W axis pair via flat indices y * W + x, then `take` along axis 1
    // after reshaping (B, H*W, C).
    let xFlat = x.reshaped([B, H * W, C])

    func gather(_ yi: MLXArray, _ xi: MLXArray) -> MLXArray {
        let idx = (yi * W + xi).reshaped([targetH * targetW]) // (tH*tW,)
        let g = take(xFlat, idx, axis: 1) // (B, tH*tW, C)
        return g.reshaped([B, targetH, targetW, C])
    }

    let v00 = gather(y0, x0)
    let v01 = gather(y0, x1)
    let v10 = gather(y1, x0)
    let v11 = gather(y1, x1)

    return v00 * (1 - fy) * (1 - fx)
        + v01 * (1 - fy) * fx
        + v10 * fy * (1 - fx)
        + v11 * fy * fx
}

// MARK: - SegmentationHead

/// Per-query mask predictions from spatial features × decoder query features.
public final class SegmentationHead: Module {
    @ModuleInfo(key: "blocks") public var blocks: [DepthwiseConvBlock]
    @ModuleInfo(key: "spatial_features_proj") public var spatialFeaturesProj: Conv2d
    @ModuleInfo(key: "query_features_block") public var queryFeaturesBlock: MLPBlock
    @ModuleInfo(key: "query_features_proj") public var queryFeaturesProj: Linear

    /// Mirrors the Python `self.bias = mx.zeros((1,))` learnable scalar.
    @ParameterInfo(key: "bias") public var bias: MLXArray

    public let downsampleRatio: Int
    public let interactionDim: Int

    public init(
        inDim: Int = 256,
        numBlocks: Int = 4,
        bottleneckRatio: Int = 1,
        downsampleRatio: Int = 4
    ) {
        self.downsampleRatio = downsampleRatio
        self.interactionDim = inDim / bottleneckRatio

        self._blocks = ModuleInfo(
            wrappedValue: (0..<numBlocks).map { _ in DepthwiseConvBlock(dim: inDim) },
            key: "blocks"
        )
        self._spatialFeaturesProj = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inDim,
                outputChannels: interactionDim,
                kernelSize: 1
            ),
            key: "spatial_features_proj"
        )
        self._queryFeaturesBlock = ModuleInfo(
            wrappedValue: MLPBlock(dim: inDim),
            key: "query_features_block"
        )
        self._queryFeaturesProj = ModuleInfo(
            wrappedValue: Linear(inDim, interactionDim),
            key: "query_features_proj"
        )

        self._bias = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "bias")
        super.init()
    }

    /// - Parameters:
    ///   - spatialFeatures: `(B, H, W, C)` backbone features (channel-last).
    ///   - queryFeatures: `(B, N, C)` decoder hidden states.
    ///   - imageSize: original image `(H, W)`.
    /// - Returns: mask logits `(B, N, H', W')` with `H' = H / downsampleRatio`.
    public func callAsFunction(
        _ spatialFeatures: MLXArray,
        queryFeatures: MLXArray,
        imageSize: (Int, Int)
    ) -> MLXArray {
        let targetH = imageSize.0 / downsampleRatio
        let targetW = imageSize.1 / downsampleRatio

        var sf = interpolateSpatial(spatialFeatures, targetH: targetH, targetW: targetW)
        for block in blocks { sf = block(sf) }

        let sfProj = spatialFeaturesProj(sf) // (B, H', W', interactionDim)

        let qf = queryFeaturesBlock(queryFeatures)
        let qfProj = queryFeaturesProj(qf) // (B, N, interactionDim)

        // einsum "bhwc,bnc->bnhw"
        let maskLogits = einsum("bhwc,bnc->bnhw", sfProj, qfProj)
        return maskLogits + bias
    }
}
