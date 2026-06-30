// Two-stage encoder + Decoder (Transformer).
//
// The two-stage encoder selects top-K queries from the projected feature map,
// refines box proposals, and feeds them into the decoder with learnable queries.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/transformer.py (Transformer)

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

    // MARK: Keypoint modules (present only when keypoints are enabled)
    /// Seeds decoder keypoint queries from detection queries (used at inference).
    @ModuleInfo(key: "keypoint_query_initializer") public var keypointQueryInitializer: ConditionalQueryInitializer?
    /// Encoder-stage keypoint initializer + heads. These feed the two-stage encoder
    /// keypoint predictions (training aux only) and are not run at inference; they are
    /// constructed so their checkpoint weights load cleanly.
    @ModuleInfo(key: "keypoint_query_initializer_enc") public var keypointQueryInitializerEnc: ConditionalQueryInitializer?
    @ModuleInfo(key: "enc_out_keypoint_embed") public var encOutKeypointEmbed: [DecoderMLP]?

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

        if config.useGroupposeKeypoints {
            let kpDim = config.keypointDim
            let K = config.totalKeypoints
            self._keypointQueryInitializer = ModuleInfo(
                wrappedValue: ConditionalQueryInitializer(dim: d, numQueries: K, outDim: kpDim),
                key: "keypoint_query_initializer"
            )
            self._keypointQueryInitializerEnc = ModuleInfo(
                wrappedValue: ConditionalQueryInitializer(dim: d, numQueries: K, outDim: kpDim),
                key: "keypoint_query_initializer_enc"
            )
            // Copies of the keypoint head: MLP(kpDim → kpDim → kpDim → 8), 3 layers.
            self._encOutKeypointEmbed = ModuleInfo(
                wrappedValue: (0..<nGroups).map { _ in
                    DecoderMLP(inputDim: kpDim, hiddenDim: kpDim, outputDim: 8, numLayers: 3)
                },
                key: "enc_out_keypoint_embed"
            )
        }
        super.init()
    }

    // MARK: - Two-stage query selection

    public func twoStageSelect(
        _ memory: MLXArray,
        spatialShapes: [(Int, Int)],
        groupIdx: Int = 0
    ) -> (MLXArray, MLXArray) {
        let B = memory.dim(0)
        let nq = config.numQueries

        // Generate grid proposals in [0,1] coordinate space (multi-level)
        let gridProposals = genEncoderOutputProposals(spatialShapes: spatialShapes) // (sum(Hi*Wi), 4)

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
        spatialShapes: [(Int, Int)],
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
        let (refpointEmbedTS, _) = twoStageSelect(memory, spatialShapes: spatialShapes, groupIdx: 0)

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

        let (hs, refUnsig) = decoder(tgt, memory: memory, referencePointsUnsigmoid: combinedRefpoints, spatialShapes: spatialShapes, bboxEmbed: bboxEmbed)

        return (hs, refUnsig)
    }

    /// Forward with the keypoint subnetwork active.
    ///
    /// - Parameters:
    ///   - memory: flattened encoder/projector memory `(B, HW, D)` for detection cross-attention.
    ///   - keypointMemory: flattened dual-projector memory `(B, HW, D)`.
    ///   - spatialShapes: per-level feature-map `(height, width)` sizes.
    ///   - queryFeat: learned query features `(numQueries, D)`.
    ///   - refpointEmbed: learned reference-point embeddings `(numQueries, 4)`.
    ///   - bboxEmbed: shared box-regression head used for iterative refinement.
    ///   - perLayer: optional per-layer callback `(layerIndex, hs, refPoints)` for diagnostics.
    /// - Returns: `(hs, refUnsig, keypointHs)` for the final decoder layer.
    public func callWithKeypoints(
        _ memory: MLXArray,
        keypointMemory: MLXArray,
        spatialShapes: [(Int, Int)],
        queryFeat: MLXArray,
        refpointEmbed: MLXArray,
        bboxEmbed: DecoderMLP,
        perLayer: ((Int, MLXArray, MLXArray) -> Void)? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = memory.dim(0)
        let nq = config.numQueries
        let d = config.hiddenDim

        let qf = queryFeat[0..<nq, 0...]
        let rp = refpointEmbed[0..<nq, 0...]

        let (refpointEmbedTS, _) = twoStageSelect(memory, spatialShapes: spatialShapes, groupIdx: 0)

        let combinedRefpoints: MLXArray
        if config.bboxReparam {
            let rpBroad = rp.expandedDimensions(axis: 0)
            let rpParts = rpBroad.split(parts: 2, axis: -1)
            let tsParts = refpointEmbedTS.split(parts: 2, axis: -1)
            let refCxcy = rpParts[0] * tsParts[1] + tsParts[0]
            let refWH = exp(rpParts[1]) * tsParts[1]
            combinedRefpoints = concatenated([refCxcy, refWH], axis: -1)
        } else {
            combinedRefpoints = rp.expandedDimensions(axis: 0) + refpointEmbedTS
        }

        let tgt = broadcast(qf.expandedDimensions(axis: 0), to: [B, nq, d])

        // Seed keypoint queries from the (pre-decoder) detection queries.
        let tgtKeypoints = keypointQueryInitializer!(tgt)  // (B, nq, K, kpDim)

        return decoder.callWithKeypoints(
            tgt,
            memory: memory,
            keypointMemory: keypointMemory,
            tgtKeypoints: tgtKeypoints,
            referencePointsUnsigmoid: combinedRefpoints,
            spatialShapes: spatialShapes,
            bboxEmbed: bboxEmbed,
            perLayer: perLayer
        )
    }
}
