// Single decoder layer: self-attn → cross-attn (deformable) → FFN.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/transformer.py (DecoderLayer)

import Foundation
import MLX
import MLXNN

/// One decoder layer with self-attention, deformable cross-attention, and FFN.
public final class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: DecoderSelfAttention
    @ModuleInfo(key: "norm1") public var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn") public var crossAttn: MSDeformableAttention
    @ModuleInfo(key: "norm2") public var norm2: LayerNorm
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear
    @ModuleInfo(key: "norm3") public var norm3: LayerNorm

    public init(config: RFDETRConfig) {
        let d = config.hiddenDim
        self._selfAttn = ModuleInfo(
            wrappedValue: DecoderSelfAttention(dModel: d, nHeads: config.saNheads),
            key: "self_attn"
        )
        self._norm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps),
            key: "norm1"
        )
        self._crossAttn = ModuleInfo(
            wrappedValue: MSDeformableAttention(
                dModel: d, nHeads: config.caNheads,
                nLevels: config.nLevels, nPoints: config.decNPoints
            ),
            key: "cross_attn"
        )
        self._norm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps),
            key: "norm2"
        )
        self._linear1 = ModuleInfo(
            wrappedValue: Linear(d, config.dimFeedforward),
            key: "linear1"
        )
        self._linear2 = ModuleInfo(
            wrappedValue: Linear(config.dimFeedforward, d),
            key: "linear2"
        )
        self._norm3 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps),
            key: "norm3"
        )
        super.init()
    }

    public func callAsFunction(
        _ tgt: MLXArray,
        memory: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: [(Int, Int)],
        queryPos: MLXArray? = nil
    ) -> MLXArray {
        // Self-attention
        let posEmbed = queryPos ?? MLXArray.zeros(tgt.shape)
        var out = tgt + selfAttn(tgt, queryPos: posEmbed)
        out = norm1(out)

        // Deformable cross-attention
        let crossQuery = queryPos != nil ? (out + queryPos!) : out
        out = out + crossAttn(crossQuery, referencePoints: referencePoints, value: memory, spatialShapes: spatialShapes)
        out = norm2(out)

        // FFN
        let ffn = linear2(relu(linear1(out)))
        out = out + ffn
        out = norm3(out)

        return out
    }
}
