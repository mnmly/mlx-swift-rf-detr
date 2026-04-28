import MLX
import MLXNN
import XCTest

@testable import RFDETRMLX

final class RFDETRMLXTests: XCTestCase {

    func testBackboneShapes() {
        let bb = DINOv2Backbone(
            imgSize: 384, patchSize: 16, embedDim: 384,
            depth: 12, numHeads: 6, numWindows: 2,
            featureIndices: [2, 5, 8, 11]
        )
        let x = MLXArray.zeros([1, 384, 384, 3])
        let feats = bb(x)
        XCTAssertEqual(feats.count, 4)
        for f in feats {
            XCTAssertEqual(f.shape, [1, 24, 24, 384])
        }
    }

    // MARK: - Decoder component tests

    func testGridSampleShapes() {
        let x = MLXArray.zeros([2, 7, 7, 128])
        let grid = MLXArray.zeros([2, 5, 3, 2])
        let out = gridSample([x, grid])[0]
        XCTAssertEqual(out.shape, [2, 5, 3, 128])
    }

    func testDecoderSelfAttentionShapes() {
        let attn = DecoderSelfAttention(dModel: 256, nHeads: 8)
        let x = MLXArray.zeros([1, 300, 256])
        let qpos = MLXArray.zeros([1, 300, 256])
        let out = attn(x, queryPos: qpos)
        XCTAssertEqual(out.shape, [1, 300, 256])
    }

    func testMSDeformableAttentionShapes() {
        let attn = MSDeformableAttention(dModel: 256, nHeads: 16, nLevels: 1, nPoints: 2)
        let query = MLXArray.zeros([1, 300, 256])
        let refPoints = MLXArray.zeros([1, 300, 1, 4]) // 4D ref points (bbox_reparam)
        let value = MLXArray.zeros([1, 576, 256])
        let out = attn(query, referencePoints: refPoints, value: value, spatialShape: (24, 24))
        XCTAssertEqual(out.shape, [1, 300, 256])
    }

    func testMSDeformable2DRefPoints() {
        let attn = MSDeformableAttention(dModel: 256, nHeads: 16, nLevels: 1, nPoints: 2)
        let query = MLXArray.zeros([1, 300, 256])
        let refPoints = MLXArray.zeros([1, 300, 2]) // 2D ref points
        let value = MLXArray.zeros([1, 576, 256])
        let out = attn(query, referencePoints: refPoints, value: value, spatialShape: (24, 24))
        XCTAssertEqual(out.shape, [1, 300, 256])
    }

    func testDecoderLayerShapes() {
        let config = RFDETRConfig()
        let layer = DecoderLayer(config: config)
        let tgt = MLXArray.zeros([1, 300, 256])
        let memory = MLXArray.zeros([1, 576, 256])
        let refPoints = MLXArray.zeros([1, 300, 1, 4])
        let queryPos = MLXArray.zeros([1, 300, 256])
        let out = layer(tgt, memory: memory, referencePoints: refPoints, spatialShape: (24, 24), queryPos: queryPos)
        XCTAssertEqual(out.shape, [1, 300, 256])
    }

    func testDecodeRepro() {
        let config = RFDETRConfig()
        let decoder = Decoder(config: config)
        let tgt = MLXArray.zeros([1, 300, 256])
        let memory = MLXArray.zeros([1, 576, 256])
        let refUnsig = MLXArray.zeros([1, 300, 4])
        let bboxEmbed = DecoderMLP(inputDim: 256, hiddenDim: 256, outputDim: 4, numLayers: 3)

        let dHalf = config.hiddenDim / 2
        let refSine = genSineembedForPosition(refUnsig, dModel: dHalf)
        let queryPos = decoder.refPointHead(refSine)

        let layer = decoder.layers[0]
        let selfAttn = layer.selfAttn

        let B = tgt.dim(0); let N = tgt.dim(1)
        let H = selfAttn.numHeads; let d = selfAttn.headDim

        let qkInput = tgt + queryPos
        let q = selfAttn.qProj(qkInput)

        print("REPRO: B=\(B) N=\(N) H=\(H) d=\(d)")
        print("REPRO: q.shape=\(q.shape) q.size=\(q.shape.reduce(1, *))")
        print("REPRO: target=[\([B, N, H, d])] size=\(B*N*H*d)")

        eval(q)
        print("REPRO: after eval q.shape=\(q.shape)")
        XCTAssertEqual(q.shape, [B, N, config.hiddenDim])
    }

    func testDecoderShapes() {
        let config = RFDETRConfig()
        let decoder = Decoder(config: config)
        let tgt = MLXArray.zeros([1, 300, 256])
        let memory = MLXArray.zeros([1, 576, 256])
        let refUnsig = MLXArray.zeros([1, 300, 4])
        let bboxEmbed = DecoderMLP(inputDim: 256, hiddenDim: 256, outputDim: 4, numLayers: 3)
        let (hs, refOut) = decoder(tgt, memory: memory, referencePointsUnsigmoid: refUnsig, spatialShape: (24, 24), bboxEmbed: bboxEmbed)
        XCTAssertEqual(hs.shape, [1, 300, 256])
        XCTAssertEqual(refOut.shape, [1, 300, 4])
    }

    func testTransformerShapes() {
        let config = RFDETRConfig(hiddenDim: 256, decLayers: 3, numQueries: 300, groupDetr: 13, numClasses: 91)
        let transformer = Transformer(config: config)
        let memory = MLXArray.zeros([1, 576, 256])
        let queryFeat = MLXArray.zeros([13 * 300, 256])
        let refpointEmbed = MLXArray.zeros([13 * 300, 4])
        let bboxEmbed = DecoderMLP(inputDim: 256, hiddenDim: 256, outputDim: 4, numLayers: 3)
        let (hs, refOut) = transformer(memory, spatialShape: (24, 24), queryFeat: queryFeat, refpointEmbed: refpointEmbed, bboxEmbed: bboxEmbed)
        XCTAssertEqual(hs.shape, [1, 300, 256])
        XCTAssertEqual(refOut.shape, [1, 300, 4])
    }

    // MARK: - Segmentation tests

    func testInterpolateSpatialShapes() {
        let x = MLXArray.zeros([1, 24, 24, 256])
        let y = interpolateSpatial(x, targetH: 96, targetW: 96)
        XCTAssertEqual(y.shape, [1, 96, 96, 256])
    }

    func testInterpolateSpatialIdentity() {
        let x = MLXArray.zeros([1, 24, 24, 256])
        let y = interpolateSpatial(x, targetH: 24, targetW: 24)
        XCTAssertEqual(y.shape, [1, 24, 24, 256])
    }

    func testDepthwiseConvBlockShapes() {
        let block = DepthwiseConvBlock(dim: 256)
        let x = MLXArray.zeros([1, 96, 96, 256])
        let y = block(x)
        XCTAssertEqual(y.shape, [1, 96, 96, 256])
    }

    func testMLPBlockShapes() {
        let block = MLPBlock(dim: 256)
        let x = MLXArray.zeros([1, 300, 256])
        let y = block(x)
        XCTAssertEqual(y.shape, [1, 300, 256])
    }

    func testSegmentationHeadShapes() {
        let head = SegmentationHead(inDim: 256, numBlocks: 4, bottleneckRatio: 1, downsampleRatio: 4)
        let spatial = MLXArray.zeros([1, 24, 24, 256]) // backbone feature map
        let queries = MLXArray.zeros([1, 300, 256]) // decoder hidden states
        let masks = head(spatial, queryFeatures: queries, imageSize: (384, 384))
        XCTAssertEqual(masks.shape, [1, 300, 96, 96]) // 384/4 = 96
    }

    func testTwoStageSelectShapes() {
        let config = RFDETRConfig(hiddenDim: 256, numQueries: 300, groupDetr: 1, numClasses: 91)
        let transformer = Transformer(config: config)
        let memory = MLXArray.zeros([1, 576, 256])
        let (refTS, memTS) = transformer.twoStageSelect(memory, spatialShape: (24, 24), groupIdx: 0)
        XCTAssertEqual(refTS.shape, [1, 300, 4])
        XCTAssertEqual(memTS.shape, [1, 300, 256])
    }
}
