// Two-stage encoder + Decoder (Transformer).
//
// The two-stage encoder selects top-K queries from the projected feature map,
// refines box proposals, and feeds them into the decoder with learnable queries.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/transformer.py (Transformer)

import Foundation
import MLX
import MLXNN

/// Two-stage encoder selection + Decoder.
public final class Transformer: Module {
    @ModuleInfo(key: "enc_output") public var encOutput: [Linear]
    @ModuleInfo(key: "enc_output_norm") public var encOutputNorm: [LayerNorm]
    @ModuleInfo(key: "enc_out_class_embed") public var encOutClassEmbed: [Linear]
    @ModuleInfo(key: "enc_out_bbox_embed") public var encOutBBoxEmbed: [DecoderMLP]
    @ModuleInfo(key: "decoder") public var decoder: Decoder

    public let config: RFDETRConfig

    public init(config: RFDETRConfig) {
        self.config = config
        let d = config.hiddenDim
        let nGroups = config.groupDetr

        self._encOutput = ModuleInfo(
            wrappedValue: (0..<nGroups).map { _ in Linear(d, d) },
            key: "enc_output"
        )
        self._encOutputNorm = ModuleInfo(
            wrappedValue: (0..<nGroups).map { _ in LayerNorm(dimensions: d) },
            key: "enc_output_norm"
        )
        self._encOutClassEmbed = ModuleInfo(
            wrappedValue: (0..<nGroups).map { _ in Linear(d, config.numClasses) },
            key: "enc_out_class_embed"
        )
        self._encOutBBoxEmbed = ModuleInfo(
            wrappedValue: (0..<nGroups).map { _ in DecoderMLP(inputDim: d, hiddenDim: d, outputDim: 4, numLayers: 3) },
            key: "enc_out_bbox_embed"
        )

        self._decoder = ModuleInfo(
            wrappedValue: Decoder(config: config),
            key: "decoder"
        )
        super.init()
    }

    // MARK: - Two-stage query selection

    public func twoStageSelect(
        _ memory: MLXArray,
        spatialShape: (Int, Int),
        groupIdx: Int = 0
    ) -> (MLXArray, MLXArray) {
        let B = memory.dim(0)
        let nq = config.numQueries
        let H = spatialShape.0; let W = spatialShape.1

        // Generate grid proposals in [0,1] coordinate space
        let gridProposals = genEncoderOutputProposals(H: H, W: W) // (HW, 4)

        // Project encoder features
        let output = encOutputNorm[groupIdx](encOutput[groupIdx](memory))

        // Classify all positions
        let clsLogits = encOutClassEmbed[groupIdx](output) // (B, HW, numClasses)

        // Predict box refinements
        let bboxDelta = encOutBBoxEmbed[groupIdx](output) // (B, HW, 4)
        let proposals = gridProposals.expandedDimensions(axis: 0) // (1, HW, 4)

        let encOutputsCoord: MLXArray
        if config.bboxReparam {
            // Parametric: delta_cxcy * proposal_wh + proposal_center (last-axis split)
            let dParts = bboxDelta.split(parts: 2, axis: -1) // [(B,HW,2), (B,HW,2)]
            let pParts = proposals.split(parts: 2, axis: -1) // [(1,HW,2), (1,HW,2)]
            let encCxcy = dParts[0] * pParts[1] + pParts[0]
            let encWh = exp(dParts[1]) * pParts[1]
            encOutputsCoord = concatenated([encCxcy, encWh], axis: -1)
        } else {
            encOutputsCoord = bboxDelta + inverseSigmoid(proposals)
        }

        // Top-K selection by max class score
        let maxScores = clsLogits.max(axis: -1) // (B, HW)
        var topkIndices = argPartition(-maxScores, kth: nq - 1, axis: -1)
        topkIndices = topkIndices[0..., 0..<nq] // (B, nq)

        // Sort by score descending (per-batch gather → take_along_axis)
        let topkScores = takeAlong(maxScores, topkIndices, axis: -1)
        let sortIdx = argSort(-topkScores, axis: -1)
        topkIndices = takeAlong(topkIndices, sortIdx, axis: -1)

        // Gather selected features and boxes
        let topkExp = topkIndices.expandedDimensions(axis: -1)
        let indicesFeat = broadcast(topkExp, to: [B, nq, output.dim(-1)])
        let selectedFeat = takeAlong(output, indicesFeat, axis: 1)
        let indicesBox = broadcast(topkExp, to: [B, nq, 4])
        let selectedBoxes = takeAlong(encOutputsCoord, indicesBox, axis: 1)

        let refpointEmbedTS = stopGradient(selectedBoxes) // (B, nq, 4)
        let memoryTS = selectedFeat

        return (refpointEmbedTS, memoryTS)
    }

    // MARK: - Full forward

    public func callAsFunction(
        _ memory: MLXArray,
        spatialShape: (Int, Int),
        queryFeat: MLXArray,
        refpointEmbed: MLXArray,
        bboxEmbed: DecoderMLP
    ) -> (MLXArray, MLXArray) {
        let B = memory.dim(0)
        let nq = config.numQueries
        let d = config.hiddenDim

        // At inference, use only group 0
        let qf = queryFeat[0..<nq, 0...] // (nq, D)
        let rp = refpointEmbed[0..<nq, 0...] // (nq, 4)

        // Two-stage query selection
        let (refpointEmbedTS, _) = twoStageSelect(memory, spatialShape: spatialShape, groupIdx: 0)

        // Combine learnable refpoint with two-stage proposals
        let combinedRefpoints: MLXArray
        if config.bboxReparam {
            let rpBroad = rp.expandedDimensions(axis: 0) // (1, nq, 4)
            let rpParts = rpBroad.split(parts: 2, axis: -1) // [(1,nq,2), (1,nq,2)]
            let tsParts = refpointEmbedTS.split(parts: 2, axis: -1) // [(B,nq,2), (B,nq,2)]
            let refCxcy = rpParts[0] * tsParts[1] + tsParts[0]
            let refWH = exp(rpParts[1]) * tsParts[1]
            combinedRefpoints = concatenated([refCxcy, refWH], axis: -1)
        } else {
            combinedRefpoints = rp.expandedDimensions(axis: 0) + refpointEmbedTS
        }

        // tgt is JUST the query features (not enriched with encoder features)
        let tgt = broadcast(qf.expandedDimensions(axis: 0), to: [B, nq, d])

        let (hs, refUnsig) = decoder(tgt, memory: memory, referencePointsUnsigmoid: combinedRefpoints, spatialShape: spatialShape, bboxEmbed: bboxEmbed)

        return (hs, refUnsig)
    }
}
