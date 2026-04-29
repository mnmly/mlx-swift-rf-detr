// Decoder stack with lite_refpoint_refine.
//
// Port of the RF-DETR decoder that generates query_pos from reference points
// once (lite_refpoint_refine) and iterates through decoder layers.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/transformer.py (Decoder)

import Foundation
import MLX
import MLXNN

/// Decoder that refines object queries through self-attn and deformable cross-attn.
public final class Decoder: Module {
    @ModuleInfo(key: "layers") public var layers: [DecoderLayer]
    @ModuleInfo(key: "norm") public var norm: LayerNorm
    @ModuleInfo(key: "ref_point_head") public var refPointHead: DecoderMLP

    public let config: RFDETRConfig

    public init(config: RFDETRConfig) {
        self.config = config
        self._layers = ModuleInfo(
            wrappedValue: (0..<config.decLayers).map { _ in DecoderLayer(config: config) },
            key: "layers"
        )
        self._norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.hiddenDim, eps: config.layerNormEps),
            key: "norm"
        )
        // RefPointHead: 4 coords * 256/2 = 512 input features → hidden_dim → hidden_dim
        self._refPointHead = ModuleInfo(
            wrappedValue: DecoderMLP(
                inputDim: config.hiddenDim * 2,
                hiddenDim: config.hiddenDim,
                outputDim: config.hiddenDim,
                numLayers: 2
            ),
            key: "ref_point_head"
        )
        super.init()
    }

    public func callAsFunction(
        _ tgt: MLXArray,
        memory: MLXArray,
        referencePointsUnsigmoid: MLXArray,
        spatialShapes: [(Int, Int)],
        bboxEmbed: DecoderMLP
    ) -> (MLXArray, MLXArray) {
        var output = tgt
        let refCoords = referencePointsUnsigmoid
        let dHalf = config.hiddenDim / 2 // 128
        let nLvl = spatialShapes.count

        // Compute query_pos ONCE from reference points (lite_refpoint_refine)
        let refSine = genSineembedForPosition(refCoords, dModel: dHalf)
        let queryPos = refPointHead(refSine) // (B, Q, D)

        for layer in layers {
            // Broadcast reference points across levels: (B, Q, nLvl, 4)
            let B = refCoords.dim(0); let Q = refCoords.dim(1)
            let refpointsInput = broadcast(
                refCoords.expandedDimensions(axis: 2),
                to: [B, Q, nLvl, 4]
            )
            output = layer(output, memory: memory, referencePoints: refpointsInput, spatialShapes: spatialShapes, queryPos: queryPos)
        }

        output = norm(output)
        return (output, refCoords)
    }
}
