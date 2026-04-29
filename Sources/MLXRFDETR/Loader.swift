// High-level factory for loading a converted RF-DETR model directory.
//
// Expects the layout produced by `mlx_vlm.models.rfdetr.convert`:
//
//   <directory>/
//     config.json
//     preprocessor_config.json
//     model.safetensors

import Foundation
import MLX
import MLXNN

public enum RFDETR {
    /// Load a converted model directory into a ready-to-use model + processor.
    ///
    /// - Parameters:
    ///   - directory: path to a directory produced by the mlx-vlm rfdetr converter.
    ///   - dtype: parameter dtype (default `.float16`).
    /// - Returns: configured model, matching processor, and detected variant.
    public static func load(
        directory: URL,
        dtype: DType = .float16
    ) throws -> (model: RFDETRModel, processor: RFDETRProcessor, variant: RFDETRVariant?) {
        let cfg = try ModelJSON.load(directory: directory)
        let prep = try? PreprocessorJSON.load(directory: directory)

        let enc = EncoderSpec.from(cfg.encoder)
        let nFeatures = cfg.outFeatureIndexes.count

        // mlx-vlm config supports both 0-indexed and HF/1-indexed `out_feature_indexes`.
        // If any index is >= depth, treat the list as 1-indexed and shift.
        let isHFIndexed = cfg.outFeatureIndexes.contains { $0 >= enc.depth }
        let featureIndices = isHFIndexed
            ? cfg.outFeatureIndexes.map { $0 - 1 }
            : cfg.outFeatureIndexes

        let backbone = DINOv2Backbone(
            imgSize: cfg.resolution,
            patchSize: cfg.patchSize,
            embedDim: enc.embedDim,
            depth: enc.depth,
            numHeads: enc.numHeads,
            numWindows: cfg.numWindows,
            featureIndices: featureIndices,
            numRegisterTokens: 0
        )

        let levelToScale: [String: Float] = ["P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25]
        let scaleFactors = (cfg.projectorScale ?? ["P4"]).compactMap { levelToScale[$0] }
        let inChannelsList = Array(repeating: enc.embedDim, count: nFeatures)
        let nLevels = scaleFactors.count

        let projector = MultiScaleProjector(
            scaleFactors: scaleFactors,
            inChannelsList: inChannelsList,
            hiddenDim: cfg.hiddenDim
        )

        let segHead: SegmentationHead? = (cfg.segmentation ?? false)
            ? SegmentationHead(inDim: cfg.hiddenDim, numBlocks: cfg.segNumBlocks ?? 4)
            : nil

        let modelConfig = RFDETRConfig(
            hiddenDim: cfg.hiddenDim,
            decLayers: cfg.decLayers,
            saNheads: cfg.saNheads,
            caNheads: cfg.caNheads,
            decNPoints: cfg.decNPoints,
            nLevels: nLevels,
            numQueries: cfg.numQueries,
            groupDetr: cfg.groupDetr,
            numClasses: cfg.numClasses,
            twoStage: cfg.twoStage,
            bboxReparam: cfg.bboxReparam,
            liteRefpointRefine: cfg.liteRefpointRefine
        )

        let model = RFDETRModel(
            config: modelConfig,
            backbone: backbone,
            projector: projector,
            segmentationHead: segHead
        )

        let weightsURL = directory.appendingPathComponent("model.safetensors")
        try loadWeights(url: weightsURL, into: model, dtype: dtype)
        eval(model)

        let processor = RFDETRProcessor(
            resolution: cfg.resolution,
            imageMean: prep?.config.imageMean ?? [0.485, 0.456, 0.406],
            imageStd: prep?.config.imageStd ?? [0.229, 0.224, 0.225],
            numSelect: prep?.postProcessConfig?.numSelect ?? 300
        )

        let variant = RFDETRVariant.detect(
            resolution: cfg.resolution,
            decLayers: cfg.decLayers,
            segmentation: cfg.segmentation ?? false,
            numQueries: cfg.numQueries,
            hiddenDim: cfg.hiddenDim
        )

        return (model, processor, variant)
    }
}

// MARK: - JSON schemas

struct ModelJSON: Decodable {
    let modelType: String?
    let encoder: String
    let hiddenDim: Int
    let resolution: Int
    let decLayers: Int
    let numQueries: Int
    let numClasses: Int
    let patchSize: Int
    let numWindows: Int
    let groupDetr: Int
    let saNheads: Int
    let caNheads: Int
    let decNPoints: Int
    let twoStage: Bool
    let bboxReparam: Bool
    let liteRefpointRefine: Bool
    let outFeatureIndexes: [Int]
    let projectorScale: [String]?
    let segmentation: Bool?
    let segNumBlocks: Int?
    let positionalEncodingSize: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoder
        case hiddenDim = "hidden_dim"
        case resolution
        case decLayers = "dec_layers"
        case numQueries = "num_queries"
        case numClasses = "num_classes"
        case patchSize = "patch_size"
        case numWindows = "num_windows"
        case groupDetr = "group_detr"
        case saNheads = "sa_nheads"
        case caNheads = "ca_nheads"
        case decNPoints = "dec_n_points"
        case twoStage = "two_stage"
        case bboxReparam = "bbox_reparam"
        case liteRefpointRefine = "lite_refpoint_refine"
        case outFeatureIndexes = "out_feature_indexes"
        case projectorScale = "projector_scale"
        case segmentation
        case segNumBlocks = "seg_num_blocks"
        case positionalEncodingSize = "positional_encoding_size"
    }

    static func load(directory: URL) throws -> ModelJSON {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ModelJSON.self, from: data)
    }
}

struct PreprocessorJSON: Decodable {
    struct Inner: Decodable {
        let imageMean: [Float]?
        let imageStd: [Float]?
        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
        }
    }
    struct PostProcess: Decodable {
        let numSelect: Int
        enum CodingKeys: String, CodingKey {
            case numSelect = "num_select"
        }
    }

    let config: Inner
    let postProcessConfig: PostProcess?

    enum CodingKeys: String, CodingKey {
        case config
        case postProcessConfig = "post_process_config"
    }

    static func load(directory: URL) throws -> PreprocessorJSON {
        let url = directory.appendingPathComponent("preprocessor_config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PreprocessorJSON.self, from: data)
    }
}
