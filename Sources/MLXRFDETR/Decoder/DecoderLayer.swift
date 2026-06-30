// Single decoder layer: self-attn → cross-attn (deformable) → FFN,
// with an optional GroupPose keypoint subnetwork.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/transformer.py (TransformerDecoderLayer)

import Foundation
import MLX
import MLXNN

/// One decoder layer with self-attention, deformable cross-attention, and FFN.
///
/// When `config.useGroupposeKeypoints` is set, the layer additionally carries a
/// keypoint subnetwork (joint instance+keypoint self-attention, a keypoint
/// deformable cross-attention, and a keypoint FFN) and `forwardWithKeypoints`
/// returns the updated detection and keypoint features.
///
/// Only `grouppose_keypoint_dim_downscale == 1` is supported (the preview model),
/// so the per-layer instance/memory projections are identities and carry no weights.
public final class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: DecoderSelfAttention
    @ModuleInfo(key: "norm1") public var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn") public var crossAttn: MSDeformableAttention
    @ModuleInfo(key: "norm2") public var norm2: LayerNorm
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear
    @ModuleInfo(key: "norm3") public var norm3: LayerNorm

    // MARK: Keypoint subnetwork (present only when keypoints are enabled)
    @ModuleInfo(key: "kp_inst_self_attn") public var kpInstSelfAttn: DecoderSelfAttention?
    @ModuleInfo(key: "kp_inst_norm") public var kpInstNorm: LayerNorm?
    @ModuleInfo(key: "kp_norm") public var kpNorm: LayerNorm?
    @ModuleInfo(key: "kp_cross_attn") public var kpCrossAttn: MSDeformableAttention?
    @ModuleInfo(key: "kp_cross_attn_norm") public var kpCrossAttnNorm: LayerNorm?
    @ModuleInfo(key: "kp_linear1") public var kpLinear1: Linear?
    @ModuleInfo(key: "kp_linear3") public var kpLinear3: Linear?
    @ModuleInfo(key: "kp_norm5") public var kpNorm5: LayerNorm?
    @ParameterInfo(key: "instance_kp_layer_scale") public var instanceKpLayerScale: MLXArray?

    let config: RFDETRConfig

    public init(config: RFDETRConfig) {
        self.config = config
        let d = config.hiddenDim
        self._selfAttn = ModuleInfo(
            wrappedValue: DecoderSelfAttention(dModel: d, nHeads: config.saNheads),
            key: "self_attn"
        )
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "norm1")
        self._crossAttn = ModuleInfo(
            wrappedValue: MSDeformableAttention(
                dModel: d, nHeads: config.caNheads,
                nLevels: config.nLevels, nPoints: config.decNPoints
            ),
            key: "cross_attn"
        )
        self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "norm2")
        self._linear1 = ModuleInfo(wrappedValue: Linear(d, config.dimFeedforward), key: "linear1")
        self._linear2 = ModuleInfo(wrappedValue: Linear(config.dimFeedforward, d), key: "linear2")
        self._norm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "norm3")

        // Keypoint subnetwork
        if config.useGroupposeKeypoints {
            let kp = config.keypointDim  // == d for downscale==1
            self._kpInstSelfAttn = ModuleInfo(
                wrappedValue: DecoderSelfAttention(dModel: kp, nHeads: config.saNheads / max(1, config.keypointDimDownscale)),
                key: "kp_inst_self_attn"
            )
            // kp_inst_norm normalizes the detection stream (d_model); kp_norm the keypoints (kpDim).
            self._kpInstNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "kp_inst_norm")
            self._kpNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: kp, eps: config.layerNormEps), key: "kp_norm")
            if config.keypointCrossAttn {
                self._kpCrossAttn = ModuleInfo(
                    wrappedValue: MSDeformableAttention(
                        dModel: kp, nHeads: config.caNheads / max(1, config.keypointDimDownscale),
                        nLevels: config.nLevels, nPoints: config.decNPoints
                    ),
                    key: "kp_cross_attn"
                )
                self._kpCrossAttnNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: kp, eps: config.layerNormEps), key: "kp_cross_attn_norm")
            }
            self._kpLinear1 = ModuleInfo(wrappedValue: Linear(kp, d * 4 / max(1, config.keypointDimDownscale)), key: "kp_linear1")
            self._kpLinear3 = ModuleInfo(wrappedValue: Linear(d * 4 / max(1, config.keypointDimDownscale), kp), key: "kp_linear3")
            self._kpNorm5 = ModuleInfo(wrappedValue: LayerNorm(dimensions: kp, eps: config.layerNormEps), key: "kp_norm5")
            self._instanceKpLayerScale = ParameterInfo(wrappedValue: MLXArray.ones([1]), key: "instance_kp_layer_scale")
        }
        super.init()
    }

    // MARK: - Detection path

    /// Detection-only forward (self-attn → deformable cross-attn → FFN).
    private func detectionForward(
        _ tgt: MLXArray,
        memory: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: [(Int, Int)],
        queryPos: MLXArray?
    ) -> MLXArray {
        let posEmbed = queryPos ?? MLXArray.zeros(tgt.shape)
        var out = tgt + selfAttn(tgt, queryPos: posEmbed)
        out = norm1(out)

        let crossQuery = queryPos != nil ? (out + queryPos!) : out
        out = out + crossAttn(crossQuery, referencePoints: referencePoints, value: memory, spatialShapes: spatialShapes)
        out = norm2(out)

        let ffn = linear2(relu(linear1(out)))
        out = out + ffn
        out = norm3(out)
        return out
    }

    public func callAsFunction(
        _ tgt: MLXArray,
        memory: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: [(Int, Int)],
        queryPos: MLXArray? = nil
    ) -> MLXArray {
        detectionForward(tgt, memory: memory, referencePoints: referencePoints, spatialShapes: spatialShapes, queryPos: queryPos)
    }

    // MARK: - Keypoint path

    /// Detection + keypoint forward.
    ///
    /// - Parameters:
    ///   - tgt: detection queries `(B, N, D)`.
    ///   - keypointTgt: keypoint queries `(B, N, K, kpDim)`.
    ///   - memory: detection cross-attention memory `(B, HW, D)`.
    ///   - keypointMemory: dual-projector memory used by the keypoint cross-attention.
    ///   - referencePoints: bbox reference points `(B, N, nLvl, 4)`; reused for the
    ///     keypoint cross-attention (broadcast across keypoints).
    ///   - spatialShapes: per-level feature-map `(height, width)` sizes.
    ///   - queryPos: detection query positional embeddings `(B, N, D)`.
    ///   - keypointPos: keypoint positional embeddings `(B, N, K, kpDim)`.
    /// - Returns: updated `(tgt, keypointTgt)`.
    public func forwardWithKeypoints(
        _ tgt: MLXArray,
        keypointTgt: MLXArray,
        memory: MLXArray,
        keypointMemory: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: [(Int, Int)],
        queryPos: MLXArray,
        keypointPos: MLXArray
    ) -> (MLXArray, MLXArray) {
        var tgt = detectionForward(tgt, memory: memory, referencePoints: referencePoints, spatialShapes: spatialShapes, queryPos: queryPos)
        var kpt = keypointTgt

        let B = tgt.dim(0); let N = tgt.dim(1); let kp = kpt.dim(3); let K = kpt.dim(2)

        // --- Joint instance + keypoint self-attention ---
        // (downscale==1: inst_in_proj / inst_pos_in_proj are identities)
        let tgtExpanded = tgt.expandedDimensions(axis: 2)                  // (B, N, 1, kp)
        let queryExpanded = MLXArray.zeros(tgtExpanded.shape)              // zero pos for instance slot
        let combinedFeat = concatenated([tgtExpanded, kpt], axis: 2)       // (B, N, 1+K, kp)
        let combinedPos = concatenated([queryExpanded, keypointPos], axis: 2)

        let flatFeat = combinedFeat.reshaped([B * N, 1 + K, kp])
        let flatPos = combinedPos.reshaped([B * N, 1 + K, kp])
        // keypoint_class_mask is all-False for the preview schema → standard attention.
        let attended = kpInstSelfAttn!(flatFeat, queryPos: flatPos).reshaped([B, N, 1 + K, kp])

        let tgt2 = attended[0..., 0..., 0, 0...]            // (B, N, kp)
        let keypointTgt2 = attended[0..., 0..., 1..., 0...] // (B, N, K, kp)

        // inst_out_proj is identity; dropout is identity at inference.
        tgt = tgt + tgt2 * instanceKpLayerScale!
        tgt = kpInstNorm!(tgt)
        kpt = kpNorm!(kpt + keypointTgt2)

        // --- Keypoint deformable cross-attention (samples dual-projector memory at bbox refs) ---
        if config.keypointCrossAttn {
            // keypoint_query = kpt + query_pos broadcast to each keypoint
            let kpQueryPos = queryPos.expandedDimensions(axis: 2)          // (B, N, 1, kp)
            let keypointQuery = (kpt + kpQueryPos).reshaped([B, N * K, kp])

            // bbox refs broadcast to each keypoint: (B, N, nLvl, 4) → (B, N*K, nLvl, 4)
            let nLvl = referencePoints.dim(2)
            let refExpanded = broadcast(
                referencePoints.expandedDimensions(axis: 2),
                to: [B, N, K, nLvl, 4]
            ).reshaped([B, N * K, nLvl, 4])

            let kpOut = kpCrossAttn!(
                keypointQuery, referencePoints: refExpanded, value: keypointMemory, spatialShapes: spatialShapes
            ).reshaped([B, N, K, kp])
            kpt = kpCrossAttnNorm!(kpt + kpOut)
        }

        // --- Keypoint FFN ---
        kpt = kpNorm5!(kpt + kpLinear3!(relu(kpLinear1!(kpt))))

        return (tgt, kpt)
    }
}
