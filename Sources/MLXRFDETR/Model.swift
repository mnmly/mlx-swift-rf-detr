// RF-DETR top-level model: backbone → projector → transformer → detection heads.
//
// Also supports optional segmentation head for mask prediction.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/lwdetr.py (LWDETR class)

import Foundation
import MLX
import MLXNN

/// Full RF-DETR detection (and optionally segmentation) model.
public final class RFDETRModel: Module {
    @ModuleInfo(key: "backbone") public var backbone: DINOv2Backbone
    @ModuleInfo(key: "projector") public var projector: MultiScaleProjector
    @ModuleInfo(key: "transformer") public var transformer: Transformer
    @ModuleInfo(key: "class_embed") public var classEmbed: Linear
    @ModuleInfo(key: "bbox_embed") public var bboxEmbed: DecoderMLP
    @ParameterInfo(key: "query_feat") public var queryFeat: MLXArray
    @ParameterInfo(key: "refpoint_embed") public var refpointEmbed: MLXArray
    @ModuleInfo(key: "segmentation_head") public var segmentationHead: SegmentationHead?

    // MARK: Keypoint heads (present only when keypoints are enabled)
    /// Second projector feeding the keypoint cross-attention (`backbone.0.cross_attn_projector`).
    @ModuleInfo(key: "cross_attn_projector") public var keypointProjector: MultiScaleProjector?
    /// Keypoint prediction head: MLP(kpDim → kpDim → kpDim → 8).
    @ModuleInfo(key: "keypoint_embed") public var keypointEmbed: DecoderMLP?

    public let config: RFDETRConfig

    /// Build a full RF-DETR model.
    ///
    /// - Parameters:
    ///   - config: transformer/decoder configuration
    ///   - backbone: DINOv2 backbone (pre-configured)
    ///   - projector: multi-scale projector (inChannels must match backbone output)
    ///   - segmentationHead: optional segmentation head (nil = detection only)
    ///   - keypointProjector: optional second projector feeding the keypoint cross-attention (nil = no keypoints)
    public init(
        config: RFDETRConfig,
        backbone: DINOv2Backbone,
        projector: MultiScaleProjector,
        segmentationHead: SegmentationHead? = nil,
        keypointProjector: MultiScaleProjector? = nil
    ) {
        self.config = config
        let d = config.hiddenDim

        self._backbone = ModuleInfo(wrappedValue: backbone, key: "backbone")
        self._projector = ModuleInfo(wrappedValue: projector, key: "projector")
        self._transformer = ModuleInfo(wrappedValue: Transformer(config: config), key: "transformer")

        // Detection heads
        self._classEmbed = ModuleInfo(wrappedValue: Linear(d, config.numClasses), key: "class_embed")
        self._bboxEmbed = ModuleInfo(
            wrappedValue: DecoderMLP(inputDim: d, hiddenDim: d, outputDim: 4, numLayers: 3),
            key: "bbox_embed"
        )

        // Learnable queries and reference points (one per query * group, shared across batch)
        let totalQueries = config.numQueries * config.groupDetr
        self._queryFeat = ParameterInfo(wrappedValue: MLXArray.zeros([totalQueries, d]), key: "query_feat")
        self._refpointEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([totalQueries, 4]), key: "refpoint_embed")

        // Optional segmentation head
        self._segmentationHead = ModuleInfo(wrappedValue: segmentationHead, key: "segmentation_head")

        // Optional keypoint heads
        self._keypointProjector = ModuleInfo(wrappedValue: keypointProjector, key: "cross_attn_projector")
        let kpEmbed: DecoderMLP? = config.useGroupposeKeypoints
            ? DecoderMLP(inputDim: config.keypointDim, hiddenDim: config.keypointDim, outputDim: 8, numLayers: 3)
            : nil
        self._keypointEmbed = ModuleInfo(wrappedValue: kpEmbed, key: "keypoint_embed")

        super.init()
    }

    /// Run detection (and optionally segmentation) on a batch of images.
    ///
    /// - Parameter pixelValues: `(B, H, W, 3)` channel-last normalized image(s).
    /// - Returns: dictionary with keys `"pred_logits"`, `"pred_boxes"`, and optionally `"pred_masks"`.
    public func callAsFunction(_ pixelValues: MLXArray) -> [String: MLXArray] {
        let imageH = pixelValues.dim(1)
        let imageW = pixelValues.dim(2)

        // 1. Backbone: extract multi-scale features
        let features = backbone(pixelValues)

        // 2. Projector: produce one feature map per scale
        let memories = projector(features)  // [(B, hi, wi, D)]
        let spatialShapes = memories.map { ($0.dim(1), $0.dim(2)) }
        let memoryFlat = concatenated(memories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)

        // 3. Transformer: two-stage selection + decoder (optionally with keypoints)
        let hs: MLXArray
        let refPoints: MLXArray
        var keypointHs: MLXArray? = nil
        if config.useGroupposeKeypoints, let kpProjector = keypointProjector {
            // Dual projector: second feature map dedicated to keypoint cross-attention.
            let kpMemories = kpProjector(features)
            let kpMemoryFlat = concatenated(
                kpMemories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1
            )
            let (h, ref, kp) = transformer.callWithKeypoints(
                memoryFlat,
                keypointMemory: kpMemoryFlat,
                spatialShapes: spatialShapes,
                queryFeat: queryFeat,
                refpointEmbed: refpointEmbed,
                bboxEmbed: bboxEmbed
            )
            hs = h; refPoints = ref; keypointHs = kp
        } else {
            (hs, refPoints) = transformer(
                memoryFlat,
                spatialShapes: spatialShapes,
                queryFeat: queryFeat,
                refpointEmbed: refpointEmbed,
                bboxEmbed: bboxEmbed
            )
        }

        // 4. Detection heads on final decoder output
        var predLogits = classEmbed(hs)  // (B, Q, numClasses)

        let predBoxes: MLXArray
        if config.bboxReparam {
            let delta = bboxEmbed(hs)  // (B, Q, 4)
            let dcParts = delta.split(parts: 2, axis: -1)  // [(B,Q,2), (B,Q,2)]
            let rpParts = refPoints.split(parts: 2, axis: -1)
            let predCxcy = dcParts[0] * rpParts[1] + rpParts[0]
            let predWH = exp(dcParts[1]) * rpParts[1]
            predBoxes = concatenated([predCxcy, predWH], axis: -1)
        } else {
            predBoxes = sigmoid(bboxEmbed(hs) + inverseSigmoid(refPoints))
        }

        var result: [String: MLXArray] = [
            "pred_boxes": predBoxes,
        ]

        // 5. Optional keypoints (decode → pad → class boost)
        if config.useGroupposeKeypoints, let keypointHs, let keypointEmbed {
            let delta = keypointEmbed(keypointHs)  // (B, Q, K, 8)
            // Decode xy against the final bbox reference (ref_unsigmoid).
            let refWH = refPoints[0..., 0..., .newAxis, 2...]   // (B, Q, 1, 2)
            let refXY = refPoints[0..., 0..., .newAxis, 0..<2]  // (B, Q, 1, 2)
            let kpXY = delta[0..., 0..., 0..., 0..<2] * refWH + refXY
            let kpOther = delta[0..., 0..., 0..., 2...]
            let compact = concatenated([kpXY, kpOther], axis: -1)  // (B, Q, K, 8)

            let (padded, classBoost) = formatKeypointsAndBoost(compact)
            result["pred_keypoints"] = padded
            predLogits = predLogits + classBoost
        }

        result["pred_logits"] = predLogits

        // 6. Optional segmentation
        if let segHead = segmentationHead {
            let predMasks = segHead(memories[0], queryFeatures: hs, imageSize: (imageH, imageW))
            result["pred_masks"] = predMasks
        }

        return result
    }

    /// Pad compact per-class keypoints `(B, Q, totalKeypoints, 8)` to the dense
    /// `(B, Q, numClasses * maxKeypoints, 8)` layout and aggregate the per-keypoint
    /// class-logit contribution (channel 7) into a `(B, Q, numClasses)` boost.
    ///
    /// PORT FROM: lwdetr.py `_format_keypoint_output` + `_aggregate_keypoint_class_logits`.
    private func formatKeypointsAndBoost(_ compact: MLXArray) -> (MLXArray, MLXArray) {
        let B = compact.dim(0); let Q = compact.dim(1)
        let counts = config.numKeypointsPerClass
        let numClasses = counts.count
        let maxK = counts.max() ?? 0

        var blocks: [MLXArray] = []
        var idx = 0
        for count in counts {
            if count > 0 {
                blocks.append(compact[0..., 0..., idx..<(idx + count), 0...])
                idx += count
            }
            if count < maxK {
                blocks.append(MLXArray.zeros([B, Q, maxK - count, 8], dtype: compact.dtype))
            }
        }
        let padded = concatenated(blocks, axis: 2)  // (B, Q, numClasses*maxK, 8)

        // Active mask (numClasses, maxK): first `count` keypoints per class are active.
        var maskRows = [Float]()
        maskRows.reserveCapacity(numClasses * maxK)
        for count in counts {
            for j in 0..<maxK { maskRows.append(j < count ? 1 : 0) }
        }
        let activeMask = MLXArray(maskRows).reshaped([numClasses, maxK]).asType(compact.dtype)

        // class_boost = sum_k (class_contrib * active_mask) over keypoints.
        let contrib = padded[0..., 0..., 0..., 7].reshaped([B, Q, numClasses, maxK])
        var classBoost = (contrib * activeMask).sum(axis: -1)  // (B, Q, numClasses)

        // Zero-pad to detection class count if the keypoint schema is narrower.
        let detClasses = config.numClasses
        if numClasses < detClasses {
            let pad = MLXArray.zeros([B, Q, detClasses - numClasses], dtype: compact.dtype)
            classBoost = concatenated([classBoost, pad], axis: -1)
        }
        return (padded, classBoost)
    }
}
