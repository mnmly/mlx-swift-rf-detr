// RF-DETR variant presets. Each variant maps directly to a converted
// model directory converted from the python rf-detr repo (see ../../python/rf-detr).
//
// Backbone hyperparameters (embedDim/depth/numHeads) are derived from
// the `encoder` field of `config.json`; the variant cases here exist
// mainly for ergonomics and disambiguation.

import Foundation

public enum RFDETRVariant: String, CaseIterable, Sendable {
    case base
    case small
    /// Pre-2026 large checkpoint (`rf-detr-large.pth`): 560 resolution, `dinov2_windowed_base` backbone, hidden dim 384.
    case large
    /// Current 2026 large checkpoint (`rf-detr-large-2026.pth`): 704 resolution, small backbone, 4 decoder layers — a different architecture from `large`.
    case large2026 = "large-2026"
    case segSmall = "seg-small"
    case segLarge = "seg-large"
    case segXLarge = "seg-xlarge"
    case seg2XLarge = "seg-2xlarge"
    case keypointPreview = "keypoint-preview"

    public var hasSegmentation: Bool {
        switch self {
        case .segSmall, .segLarge, .segXLarge, .seg2XLarge: return true
        default: return false
        }
    }

    public var hasKeypoints: Bool { self == .keypointPreview }

    /// Match the converter's MODEL_VARIANTS table by signature.
    static func detect(resolution: Int, decLayers: Int, segmentation: Bool, numQueries: Int, hiddenDim: Int = 256) -> RFDETRVariant? {
        switch (resolution, decLayers, segmentation, numQueries, hiddenDim) {
        case (560, 3, false, 300, 256): return .base
        case (512, 3, false, 300, _):   return .small
        case (560, 3, false, 300, 384): return .large
        case (704, 4, false, 300, 256): return .large2026
        case (576, 4, false, 100, 256): return .keypointPreview
        case (384, 4, true,  100, _):   return .segSmall
        case (504, 5, true,  300, _):   return .segLarge
        case (624, 6, true,  300, _):   return .segXLarge
        case (768, 6, true,  300, _):   return .seg2XLarge
        default: return nil
        }
    }
}

/// Embed dimensions for each `encoder` string used by the converter.
struct EncoderSpec {
    let embedDim: Int
    let numHeads: Int
    let depth: Int

    static func from(_ encoder: String) -> EncoderSpec {
        switch encoder {
        case "dinov2_windowed_base":
            return .init(embedDim: 768, numHeads: 12, depth: 12)
        case "dinov2_windowed_large":
            return .init(embedDim: 1024, numHeads: 16, depth: 12)
        default:
            // dinov2_windowed_small (and fallback)
            return .init(embedDim: 384, numHeads: 6, depth: 12)
        }
    }
}
