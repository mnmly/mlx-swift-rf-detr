// RF-DETR configuration.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/config.py

import Foundation

/// Transformer/decoder configuration for RF-DETR.
public struct RFDETRConfig: Sendable {
    public var hiddenDim: Int
    public var decLayers: Int
    public var saNheads: Int
    public var caNheads: Int
    public var dimFeedforward: Int
    public var decNPoints: Int
    public var nLevels: Int
    public var numQueries: Int
    public var groupDetr: Int
    public var numClasses: Int
    public var twoStage: Bool
    public var bboxReparam: Bool
    public var liteRefpointRefine: Bool
    public var layerNormEps: Float

    // MARK: - Keypoint (GroupPose) configuration
    /// Enable the GroupPose keypoint decoding path. When false the model is
    /// detection/segmentation only and no keypoint modules are constructed.
    public var useGroupposeKeypoints: Bool
    /// Per-class keypoint counts, e.g. `[0, 17]` (class 0 = background/0 kpts,
    /// class 1 = person/17 kpts). Empty when keypoints are disabled.
    public var numKeypointsPerClass: [Int]
    /// Whether each decoder layer runs a keypoint-specific deformable cross-attention.
    public var keypointCrossAttn: Bool
    /// Use a second ("cross-attn") projector dedicated to keypoint cross-attention.
    public var dualProjector: Bool
    /// In dual-projector mode, route only the keypoint cross-attention to the
    /// second projector (detection keeps the main projector).
    public var dualProjectorKpOnly: Bool
    /// Keypoint embedding dim downscale factor (1 = keypoints use full `hiddenDim`).
    public var keypointDimDownscale: Int
    /// Inter-instance keypoint attention (disabled for the preview model).
    public var interInstanceKpAttn: Bool

    /// Total keypoints carried by the decoder (sum over classes).
    public var totalKeypoints: Int { numKeypointsPerClass.reduce(0, +) }
    /// Keypoint feature dimension (`hiddenDim / keypointDimDownscale`).
    public var keypointDim: Int { hiddenDim / max(1, keypointDimDownscale) }

    /// RF-DETR Small: resolution=512, patch_size=16, feature_indices=[3,6,9,12]
    public static let small = RFDETRConfig(numClasses: 90)

    public init(
        hiddenDim: Int = 256,
        decLayers: Int = 3,
        saNheads: Int = 8,
        caNheads: Int = 16,
        dimFeedforward: Int = 2048,
        decNPoints: Int = 2,
        nLevels: Int = 1,
        numQueries: Int = 300,
        groupDetr: Int = 13,
        numClasses: Int = 91,
        twoStage: Bool = true,
        bboxReparam: Bool = true,
        liteRefpointRefine: Bool = true,
        layerNormEps: Float = 1e-5,
        useGroupposeKeypoints: Bool = false,
        numKeypointsPerClass: [Int] = [],
        keypointCrossAttn: Bool = true,
        dualProjector: Bool = false,
        dualProjectorKpOnly: Bool = false,
        keypointDimDownscale: Int = 1,
        interInstanceKpAttn: Bool = false
    ) {
        self.hiddenDim = hiddenDim
        self.decLayers = decLayers
        self.saNheads = saNheads
        self.caNheads = caNheads
        self.dimFeedforward = dimFeedforward
        self.decNPoints = decNPoints
        self.nLevels = nLevels
        self.numQueries = numQueries
        self.groupDetr = groupDetr
        self.numClasses = numClasses + 1 // +1 for background
        self.twoStage = twoStage
        self.bboxReparam = bboxReparam
        self.liteRefpointRefine = liteRefpointRefine
        self.layerNormEps = layerNormEps
        self.useGroupposeKeypoints = useGroupposeKeypoints
        self.numKeypointsPerClass = numKeypointsPerClass
        self.keypointCrossAttn = keypointCrossAttn
        self.dualProjector = dualProjector
        self.dualProjectorKpOnly = dualProjectorKpOnly
        self.keypointDimDownscale = keypointDimDownscale
        self.interInstanceKpAttn = interInstanceKpAttn
    }
}
