// Multi-Scale Deformable Attention using Metal grid_sample kernel.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/transformer.py (MSDeformableAttention)

import Foundation
import MLX
import MLXNN

/// Multi-Scale Deformable Attention with bilinear sampling.
///
/// Supports 2D and 4D reference points (bbox_reparam mode).
/// Currently assumes `nLevels == 1` (single-scale feature map).
public final class MSDeformableAttention: Module {
    @ModuleInfo(key: "sampling_offsets") public var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") public var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") public var valueProj: Linear
    @ModuleInfo(key: "output_proj") public var outputProj: Linear

    public let dModel: Int
    public let nHeads: Int
    public let nLevels: Int
    public let nPoints: Int
    public let headDim: Int

    public init(
        dModel: Int = 256,
        nHeads: Int = 16,
        nLevels: Int = 1,
        nPoints: Int = 2
    ) {
        self.dModel = dModel
        self.nHeads = nHeads
        self.nLevels = nLevels
        self.nPoints = nPoints
        self.headDim = dModel / nHeads

        self._samplingOffsets = ModuleInfo(
            wrappedValue: Linear(dModel, nHeads * nLevels * nPoints * 2),
            key: "sampling_offsets"
        )
        self._attentionWeights = ModuleInfo(
            wrappedValue: Linear(dModel, nHeads * nLevels * nPoints),
            key: "attention_weights"
        )
        self._valueProj = ModuleInfo(
            wrappedValue: Linear(dModel, dModel),
            key: "value_proj"
        )
        self._outputProj = ModuleInfo(
            wrappedValue: Linear(dModel, dModel),
            key: "output_proj"
        )
        super.init()
    }

    public func callAsFunction(
        _ query: MLXArray,
        referencePoints: MLXArray,
        value: MLXArray,
        spatialShapes: [(Int, Int)]
    ) -> MLXArray {
        let B = query.dim(0); let Q = query.dim(1)

        // Project values: (B, S, nHeads, headDim) where S = sum of Hi*Wi
        let vProj = valueProj(value).reshaped([B, -1, nHeads, headDim])

        // Compute sampling offsets: (B, Q, nHeads, nLevels, nPoints, 2)
        var offsets = samplingOffsets(query)
        offsets = offsets.reshaped([B, Q, nHeads, nLevels, nPoints, 2])

        // Compute attention weights: (B, Q, nHeads, nLevels, nPoints)
        var attnWeights_ = attentionWeights(query)
        attnWeights_ = attnWeights_.reshaped([B, Q, nHeads, nLevels * nPoints])
        attnWeights_ = softmax(attnWeights_, axis: -1)
        let attnWeights = attnWeights_.reshaped([B, Q, nHeads, nLevels, nPoints])

        // Compute sampling locations: (B, Q, nHeads, nLevels, nPoints, 2)
        let samplingLocations: MLXArray
        if referencePoints.ndim == 3 {
            // (B, Q, 2) — broadcast over heads, levels, points
            let ref = referencePoints
                .expandedDimensions(axis: 2)
                .expandedDimensions(axis: 2)
                .expandedDimensions(axis: 2) // (B, Q, 1, 1, 1, 2)
            let (H0, W0) = spatialShapes[0]
            let offsetNormalizer = MLXArray([Float(W0), Float(H0)]).reshaped([1, 1, 1, 1, 1, 2])
            samplingLocations = ref + offsets / offsetNormalizer
        } else if referencePoints.dim(-1) == 2 {
            // (B, Q, nLevels, 2)
            let ref = referencePoints[0..., 0..., .newAxis, 0..., .newAxis, 0...] // (B, Q, 1, nLevels, 1, 2)
            let (H0, W0) = spatialShapes[0]
            let offsetNormalizer = MLXArray([Float(W0), Float(H0)]).reshaped([1, 1, 1, 1, 1, 2])
            samplingLocations = ref + offsets / offsetNormalizer
        } else if referencePoints.dim(-1) == 4 {
            // (B, Q, nLevels, 4) — bbox_reparam mode
            let refCenter = referencePoints[0..., 0..., .newAxis, 0..., .newAxis, 0..<2]
            let refWH = referencePoints[0..., 0..., .newAxis, 0..., .newAxis, 2...]
            samplingLocations = refCenter + offsets / Float(nPoints) * refWH * 0.5
        } else {
            fatalError("referencePoints last dim must be 2 or 4")
        }

        // Accumulate weighted samples across all levels
        var output: MLXArray? = nil
        var offset = 0
        for (lvl, (H, W)) in spatialShapes.enumerated() {
            let sz = H * W
            // Slice value for this level: (B, H*W, nHeads, headDim)
            let vLvl = vProj[0..., offset..<(offset + sz), 0..., 0...]
            // Reshape for grid_sample: (B*nHeads, H, W, headDim)
            let vSpatial = vLvl.reshaped([B, H, W, nHeads, headDim])
                                .transposed(0, 3, 1, 2, 4)
                                .reshaped([B * nHeads, H, W, headDim])

            // Sampling locations for this level: (B, Q, nHeads, nPoints, 2)
            let sampLoc = samplingLocations[0..., 0..., 0..., lvl, 0..., 0...]
            // Convert [0,1] → [-1,1] and reshape for grid_sample: (B*nHeads, Q, nPoints, 2)
            let gridCoords = (sampLoc * 2 - 1).transposed(0, 2, 1, 3, 4).reshaped([B * nHeads, Q, nPoints, 2])

            // Grid sample: (B*nHeads, Q, nPoints, headDim)
            let sampled = gridSample([vSpatial, gridCoords])[0]
            let sampledR = sampled.reshaped([B, nHeads, Q, nPoints, headDim])

            // Attention weights for this level: (B, Q, nHeads, nPoints) → (B, nHeads, Q, nPoints, 1)
            let w = attnWeights[0..., 0..., 0..., lvl, 0...]
                        .transposed(0, 2, 1, 3)
                        .expandedDimensions(axis: -1)

            // Weighted sum over points: (B, nHeads, Q, headDim)
            let contrib = (sampledR * w).sum(axis: 3)
            output = output.map { $0 + contrib } ?? contrib

            offset += sz
        }

        // Reshape to (B, Q, dModel)
        let out = output!.transposed(0, 2, 1, 3).reshaped([B, Q, dModel])
        return outputProj(out)
    }
}
