// Conditional keypoint query initializer (GroupPose).
//
// Seeds per-instance keypoint query tokens by modulating a learned bank of
// keypoint queries with AdaLN parameters predicted from the detection query
// features.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/heads/keypoints.py
//   (ConditionalQueryInitializer)

import Foundation
import MLX
import MLXNN

/// Initialize keypoint query tokens via AdaLN-style modulation.
///
/// Forward maps detection query features `(B, N, dim)` to keypoint queries
/// `(B, N, totalKeypoints, outDim)`.
public final class ConditionalQueryInitializer: Module {
    /// Learned keypoint query bank `(totalKeypoints, outDim)`.
    @ParameterInfo(key: "queries") public var queries: MLXArray
    /// Two Linears wrapping a GELU. The PyTorch `nn.Sequential` indices are
    /// `0` (Linear) and `2` (Linear); the GELU at index `1` carries no weights,
    /// so the loader remaps `adaLN_modulation.2` → `adaLN_modulation.1` to fit
    /// this 2-element array.
    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: [Linear]
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    /// `query_norm` is a non-affine LayerNorm (no learnable weights).
    let queryNorm: LayerNorm

    public init(dim: Int, numQueries: Int, outDim: Int) {
        self._queries = ParameterInfo(
            wrappedValue: MLXArray.zeros([numQueries, outDim]), key: "queries"
        )
        self._adaLNModulation = ModuleInfo(
            wrappedValue: [Linear(dim, dim), Linear(dim, outDim * 3)],
            key: "adaLN_modulation"
        )
        self._outProj = ModuleInfo(wrappedValue: Linear(outDim, outDim), key: "out_proj")
        self.queryNorm = LayerNorm(dimensions: outDim, eps: 1e-5, affine: false)
        super.init()
    }

    /// - Parameter queryFeatures: detection query features `(B, N, dim)`.
    /// - Returns: keypoint queries `(B, N, totalKeypoints, outDim)`.
    public func callAsFunction(_ queryFeatures: MLXArray) -> MLXArray {
        let normed = queryNorm(queries)  // (K, outDim)

        var mod = queryFeatures.expandedDimensions(axis: -2)  // (B, N, 1, dim)
        mod = adaLNModulation[0](mod)
        mod = gelu(mod)
        mod = adaLNModulation[1](mod)  // (B, N, 1, outDim*3)

        let parts = mod.split(parts: 3, axis: -1)  // each (B, N, 1, outDim)
        let scale = parts[0]
        let shift = parts[1]
        let gate = parts[2]

        // modulate(normed, scale, shift) = (scale + 1) * normed + shift
        // normed (K, outDim) broadcasts against (B, N, 1, outDim) → (B, N, K, outDim)
        let modulated = (scale + 1.0) * normed + shift
        return outProj(modulated) * gate + queries
    }
}
