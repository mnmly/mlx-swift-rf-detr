import MLX
import MLXNN
import XCTest

@testable import MLXRFDETR

final class MLXRFDETRTests: XCTestCase {

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

    // MARK: - Projector tests

    func testConvBNShapes() {
        let cb = ConvBN(inChannels: 384, outChannels: 256)
        let x = MLXArray.zeros([1, 24, 24, 384])
        let y = cb(x)
        XCTAssertEqual(y.shape, [1, 24, 24, 256])
    }

    func testBottleneckShapes() {
        let bn = Bottleneck(channels: 128)
        let x = MLXArray.zeros([1, 24, 24, 128])
        let y = bn(x)
        XCTAssertEqual(y.shape, [1, 24, 24, 128])
    }

    func testC2fShapes() {
        // DINOv2 small: 4 feature maps × 384 = 1536 input channels
        let c2f = C2f(inChannels: 1536, outChannels: 256, numBottlenecks: 3, bottleneckChannels: 128)
        let x = MLXArray.zeros([1, 24, 24, 1536])
        let y = c2f(x)
        XCTAssertEqual(y.shape, [1, 24, 24, 256])
    }

    func testMultiScaleProjectorShapes() {
        // Backbone outputs 4 × [1, 24, 24, 384]
        let feats: [MLXArray] = [
            MLXArray.zeros([1, 24, 24, 384]),
            MLXArray.zeros([1, 24, 24, 384]),
            MLXArray.zeros([1, 24, 24, 384]),
            MLXArray.zeros([1, 24, 24, 384]),
        ]
        let proj = MultiScaleProjector(inChannels: 1536, hiddenDim: 256)
        let mem = proj(feats)
        XCTAssertEqual(mem.shape, [1, 24, 24, 256])
    }

    // MARK: - Full model test

    func testModelShapes() {
        let config = RFDETRConfig(hiddenDim: 256, decLayers: 3, numQueries: 300, groupDetr: 13, numClasses: 91)
        let bb = DINOv2Backbone(
            imgSize: 384, patchSize: 16, embedDim: 384,
            depth: 12, numHeads: 6, numWindows: 2,
            featureIndices: [2, 5, 8, 11]
        )
        let proj = MultiScaleProjector(inChannels: 1536, hiddenDim: 256)
        let model = RFDETRModel(config: config, backbone: bb, projector: proj)

        let x = MLXArray.zeros([1, 384, 384, 3])
        let out = model(x)

        XCTAssertEqual(out["pred_logits"]?.shape, [1, 300, 92])
        XCTAssertEqual(out["pred_boxes"]?.shape, [1, 300, 4])
        XCTAssertNil(out["pred_masks"]) // no seg head
    }

    func testModelWithSegmentationShapes() {
        let config = RFDETRConfig(hiddenDim: 256, decLayers: 3, numQueries: 300, groupDetr: 13, numClasses: 91)
        let bb = DINOv2Backbone(
            imgSize: 384, patchSize: 16, embedDim: 384,
            depth: 12, numHeads: 6, numWindows: 2,
            featureIndices: [2, 5, 8, 11]
        )
        let proj = MultiScaleProjector(inChannels: 1536, hiddenDim: 256)
        let segHead = SegmentationHead(inDim: 256, numBlocks: 4, bottleneckRatio: 1, downsampleRatio: 4)
        let model = RFDETRModel(config: config, backbone: bb, projector: proj, segmentationHead: segHead)

        let x = MLXArray.zeros([1, 384, 384, 3])
        let out = model(x)

        XCTAssertEqual(out["pred_logits"]?.shape, [1, 300, 92])
        XCTAssertEqual(out["pred_boxes"]?.shape, [1, 300, 4])
        XCTAssertEqual(out["pred_masks"]?.shape, [1, 300, 96, 96])
    }

    // MARK: - Weight sanitizer tests

    func testSanitizeBackboneLayer() {
        let v = MLXArray.zeros([384, 384])
        // HF key: model.backbone.0.encoder.encoder.encoder.layer.5.attention.attention.query.weight
        let results = sanitized(
            key: "model.backbone.0.encoder.encoder.encoder.layer.5.attention.attention.query.weight",
            value: v
        )
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].0, "backbone.blocks.5.attn.q.weight")
    }

    func testSanitizeConvTranspose() {
        // Conv2d: 4D tensor needs transposition (out,in,kH,kW) → (out,kH,kW,in)
        let v = MLXArray.zeros([384, 3, 16, 16]) // torch format
        let results = sanitized(
            key: "model.backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
            value: v
        )
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].0, "backbone.patch_embed.proj.weight")
        XCTAssertEqual(results[0].1.shape, [384, 16, 16, 3]) // NHWC
    }

    func testSanitizeLayerScale() {
        let v = MLXArray.zeros([384])
        let results = sanitized(
            key: "model.backbone.0.encoder.encoder.encoder.layer.3.layer_scale1.lambda1",
            value: v
        )
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].0, "backbone.blocks.3.ls1.gamma")
    }

    func testSanitizeFusedQKV() {
        // shape: (3*256, 256)
        let v = MLXArray.zeros([768, 256])
        let results = sanitized(
            key: "model.transformer.decoder.layers.0.self_attn.in_proj_weight",
            value: v
        )
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].0, "transformer.decoder.layers.0.self_attn.q_proj.weight")
        XCTAssertEqual(results[0].1.shape, [256, 256])
        XCTAssertEqual(results[1].0, "transformer.decoder.layers.0.self_attn.k_proj.weight")
        XCTAssertEqual(results[1].1.shape, [256, 256])
        XCTAssertEqual(results[2].0, "transformer.decoder.layers.0.self_attn.v_proj.weight")
        XCTAssertEqual(results[2].1.shape, [256, 256])
    }

    func testSanitizeProjector() {
        let v = MLXArray.zeros([256, 1536, 1, 1])
        let results = sanitized(
            key: "model.backbone.0.projector.stages.0.0.cv1.conv.weight",
            value: v
        )
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].0, "projector.c2f.cv1.conv.weight")
        XCTAssertEqual(results[0].1.shape, [256, 1, 1, 1536])
    }

    func testSanitizeSkipMaskToken() {
        let v = MLXArray.zeros([1, 1, 384])
        let results = sanitized(
            key: "model.backbone.0.encoder.encoder.embeddings.mask_token",
            value: v
        )
        XCTAssertEqual(results.count, 0)
    }

    // MARK: - Full pipeline test (no weights)

    func testFullPipelineWithRandomWeights() {
        // Build model with seg head
        let config = RFDETRConfig(hiddenDim: 256, decLayers: 3, numQueries: 300, groupDetr: 1, numClasses: 91)
        let bb = DINOv2Backbone(
            imgSize: 384, patchSize: 16, embedDim: 384,
            depth: 12, numHeads: 6, numWindows: 2,
            featureIndices: [2, 5, 8, 11]
        )
        let proj = MultiScaleProjector(inChannels: 1536, hiddenDim: 256)
        let segHead = SegmentationHead(inDim: 256, numBlocks: 4, bottleneckRatio: 1, downsampleRatio: 4)
        let model = RFDETRModel(config: config, backbone: bb, projector: proj, segmentationHead: segHead)

        let x = MLXArray.zeros([1, 384, 384, 3])
        let out = model(x)

        // Verify output structure
        XCTAssertEqual(out["pred_logits"]?.shape, [1, 300, 92])
        XCTAssertEqual(out["pred_boxes"]?.shape, [1, 300, 4])
        XCTAssertEqual(out["pred_masks"]?.shape, [1, 300, 96, 96])

        // Verify post-processing runs without crashing (random weights → likely no detections)
        eval(out["pred_logits"]!, out["pred_boxes"]!, out["pred_masks"]!)

        let result = postProcess(
            predLogits: out["pred_logits"]!,
            predBoxes: out["pred_boxes"]!,
            originalSize: (384, 384),
            scoreThreshold: 0.5,
            predMasks: out["pred_masks"]!,
            nmsThreshold: 0.5
        )
        // Random wei ghts → no detections above 0.5 threshold
        XCTAssertGreaterThanOrEqual(result.count, 0) // at minimum, no crash
    }
}
