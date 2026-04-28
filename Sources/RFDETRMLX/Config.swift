// RF-DETR configuration.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/config.py

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
        layerNormEps: Float = 1e-5
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
    }
}
