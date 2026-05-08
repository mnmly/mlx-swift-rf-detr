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

    public let config: RFDETRConfig

    /// Build a full RF-DETR model.
    ///
    /// - Parameters:
    ///   - config: transformer/decoder configuration
    ///   - backbone: DINOv2 backbone (pre-configured)
    ///   - projector: multi-scale projector (inChannels must match backbone output)
    ///   - segmentationHead: optional segmentation head (nil = detection only)
    public init(
        config: RFDETRConfig,
        backbone: DINOv2Backbone,
        projector: MultiScaleProjector,
        segmentationHead: SegmentationHead? = nil
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
        let B = memories[0].dim(0)
        let D = memories[0].dim(-1)
        let memoryFlat = concatenated(memories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)

        // 3. Transformer: two-stage selection + decoder
        let (hs, refPoints) = transformer(
            memoryFlat,
            spatialShapes: spatialShapes,
            queryFeat: queryFeat,
            refpointEmbed: refpointEmbed,
            bboxEmbed: bboxEmbed
        )

        // 4. Detection heads on final decoder output
        let predLogits = classEmbed(hs)  // (B, Q, numClasses)

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
            "pred_logits": predLogits,
            "pred_boxes": predBoxes,
        ]

        // 5. Optional segmentation
        if let segHead = segmentationHead {
            let predMasks = segHead(memories[0], queryFeatures: hs, imageSize: (imageH, imageW))
            result["pred_masks"] = predMasks
        }

        return result
    }
}
