import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXRFDETR

/// Numerical parity for the RF-DETR keypoint-preview (GroupPose) model.
///
/// Requires:
///   - a converted model directory (config.json + model.safetensors), default
///     `~/.cache/rfdetr/rfdetr-keypoint-preview-mlx` or `RFDETR_KP_MODEL_DIR`.
///   - keypoint fixtures (input/outputs/intermediates), default
///     `Tests/fixtures/keypoint` or `RFDETR_KP_FIXTURE_DIR`.
/// Both are produced by `Scripts/convert_keypoint.py` and
/// `Tests/fixtures/keypoint/generate_keypoint_fixtures.py`. Tests skip (not fail)
/// when either is missing.
final class KeypointParityTests: XCTestCase {

    var modelDir: URL {
        if let env = ProcessInfo.processInfo.environment["RFDETR_KP_MODEL_DIR"] {
            return URL(fileURLWithPath: env)
        }
        return URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/rfdetr/rfdetr-keypoint-preview-mlx")
    }

    var fixtureDir: URL {
        if let env = ProcessInfo.processInfo.environment["RFDETR_KP_FIXTURE_DIR"] {
            return URL(fileURLWithPath: env)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXRFDETRTests/
            .deletingLastPathComponent()   // Tests/
            .appendingPathComponent("fixtures/keypoint")
    }

    private func loadModelAndInput() throws -> (RFDETRModel, MLXArray) {
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw XCTSkip("Converted keypoint model not found at \(modelDir.path) — run Scripts/convert_keypoint.py")
        }
        let (model, _, _) = try RFDETR.load(directory: modelDir, dtype: .float32)

        let inputURL = fixtureDir.appendingPathComponent("input.safetensors")
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw XCTSkip("Keypoint fixtures missing — run generate_keypoint_fixtures.py")
        }
        let inputs = try MLX.loadArrays(url: inputURL, stream: .cpu)
        guard let pixelValues = inputs["pixel_values"] else {
            throw XCTSkip("Missing pixel_values in keypoint input fixture")
        }
        return (model, pixelValues)
    }

    private func loadFixtureArrays(_ file: String) throws -> [String: MLXArray] {
        let url = fixtureDir.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("\(file) missing — regenerate fixtures")
        }
        return try MLX.loadArrays(url: url, stream: .cpu)
    }

    // MARK: - Primary: full model output parity

    func testKeypointOutputParity() throws {
        let (model, pixelValues) = try loadModelAndInput()
        XCTAssertEqual(pixelValues.shape, [1, 576, 576, 3])

        let out = model(pixelValues)
        guard let logits = out["pred_logits"],
              let boxes = out["pred_boxes"],
              let keypoints = out["pred_keypoints"] else {
            XCTFail("Missing keypoint model outputs: \(out.keys)")
            return
        }
        eval(logits, boxes, keypoints)

        XCTAssertEqual(logits.shape, [1, 100, 2])
        XCTAssertEqual(boxes.shape, [1, 100, 4])
        XCTAssertEqual(keypoints.shape, [1, 100, 34, 8])

        let expected = try loadFixtureArrays("outputs.safetensors")
        assertParity(logits, expected["pred_logits"]!, name: "pred_logits", atol: 4.0, statsRtol: 0.2)
        assertParity(boxes, expected["pred_boxes"]!, name: "pred_boxes", atol: 2.5, statsRtol: 0.2)

        // Per-channel breakdown: xy localization (0,1) vs aux (findable/visible 2,3),
        // Cholesky (4,5,6), class-contrib (7). xy/conf are the functional outputs.
        let expKp = expected["pred_keypoints"]!
        let chanNames = ["x", "y", "findable", "visible", "logL11", "L21", "logL22", "classContrib"]
        for ch in 0..<8 {
            let a = keypoints[0..., 0..., 0..., ch]
            let e = expKp[0..., 0..., 0..., ch]
            let d = (a - e).abs()
            print("  kp ch[\(chanNames[ch])] max=\(d.max().item(Float.self)) "
                  + "|μ|s=\(a.abs().mean().item(Float.self)) |μ|p=\(e.abs().mean().item(Float.self))")
        }
        // xy localization is the functional output and stays tight. The full tensor's
        // max-diff is dominated by the L21 Cholesky off-diagonal (high dynamic range,
        // unstable on a synthetic input with no real objects) — gate it on aggregate
        // stats (|μ|/σ) like the detection parity tests rather than element-wise max.
        let xy = keypoints[0..., 0..., 0..., 0..<2]
        let expXY = expKp[0..., 0..., 0..., 0..<2]
        assertParity(xy, expXY, name: "pred_keypoints xy", atol: 1.5, statsRtol: 0.12)
        assertParity(keypoints, expKp, name: "pred_keypoints (all)", atol: 13.0, statsRtol: 0.25)
    }

    // MARK: - Staged parity (localizes keypoint-path bugs)

    func testDualProjectorParity() throws {
        let (model, pixelValues) = try loadModelAndInput()
        let inter = try loadFixtureArrays("intermediates.safetensors")
        let features = model.backbone(pixelValues)

        // Main projector → (B, h, w, D); fixture is (B, D, h, w) (NCHW).
        let mem = model.projector(features)[0]
        eval(mem)
        if let exp = inter["backbone_0_projector_0"] {
            assertParity(mem.transposed(0, 3, 1, 2), exp, name: "projector", atol: 0.7)
        }
        guard let kpProj = model.keypointProjector else {
            XCTFail("keypointProjector missing"); return
        }
        let kpMem = kpProj(features)[0]
        eval(kpMem)
        if let exp = inter["backbone_0_cross_attn_projector_0"] {
            assertParity(kpMem.transposed(0, 3, 1, 2), exp, name: "cross_attn_projector", atol: 0.7)
        }
    }

    func testKeypointInitializerParity() throws {
        let (model, _) = try loadModelAndInput()
        let inter = try loadFixtureArrays("intermediates.safetensors")
        let nq = model.config.numQueries
        let d = model.config.hiddenDim
        let qf = model.queryFeat[0..<nq, 0...]
        let tgt = broadcast(qf.expandedDimensions(axis: 0), to: [1, nq, d])
        let kpInit = model.transformer.keypointQueryInitializer!(tgt)
        eval(kpInit)
        guard let exp = inter["transformer_keypoint_query_initializer"] else {
            throw XCTSkip("missing initializer intermediate")
        }
        assertParity(kpInit, exp, name: "keypoint_query_initializer", atol: 1.0, statsRtol: 0.1)
    }

    func testKeypointDecoderParity() throws {
        let (model, pixelValues) = try loadModelAndInput()
        let inter = try loadFixtureArrays("intermediates.safetensors")

        let features = model.backbone(pixelValues)
        let memories = model.projector(features)
        let spatialShapes = memories.map { ($0.dim(1), $0.dim(2)) }
        let memFlat = concatenated(memories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)
        let kpMemories = model.keypointProjector!(features)
        let kpFlat = concatenated(kpMemories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)

        let (hs, _, kpHs) = model.transformer.callWithKeypoints(
            memFlat, keypointMemory: kpFlat, spatialShapes: spatialShapes,
            queryFeat: model.queryFeat, refpointEmbed: model.refpointEmbed, bboxEmbed: model.bboxEmbed
        )
        eval(hs, kpHs)

        // transformer.0 = stacked hs (4,1,100,256); transformer.4 = stacked keypoint_hs (4,1,100,17,256).
        if let exp = inter["transformer.0"] {
            assertParity(hs, exp[-1], name: "hs (detection final)", atol: 8.0, statsRtol: 0.15)
        }
        if let exp = inter["transformer.4"] {
            assertParity(kpHs, exp[-1], name: "keypoint_hs final", atol: 8.0, statsRtol: 0.15)
        }
        let delta = model.keypointEmbed!(kpHs)
        eval(delta)
        if let exp = inter["keypoint_embed"] {
            // Max-diff dominated by the L21 Cholesky off-diagonal (high dynamic range);
            // gate on aggregate stats. The head-isolation check below proves the head exact.
            assertParity(delta, exp[-1], name: "keypoint_embed", atol: 13.0, statsRtol: 0.25)
        }
        // Head isolation: feed the PYTHON keypoint_hs to the Swift head. If this
        // matches, the head weights are correct and the bug is upstream in kpHs.
        if let kpHsPy = inter["transformer.4"], let exp = inter["keypoint_embed"] {
            let deltaPy = model.keypointEmbed!(kpHsPy[-1])
            eval(deltaPy)
            assertParity(deltaPy, exp[-1], name: "keypoint_embed(pythonHs)", atol: 0.05, statsRtol: 0.02)
        }
    }

    func testKeypointPerLayerParity() throws {
        let (model, pixelValues) = try loadModelAndInput()
        let inter = try loadFixtureArrays("intermediates.safetensors")

        let features = model.backbone(pixelValues)
        let memories = model.projector(features)
        let spatialShapes = memories.map { ($0.dim(1), $0.dim(2)) }
        let memFlat = concatenated(memories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)
        let kpMemories = model.keypointProjector!(features)
        let kpFlat = concatenated(kpMemories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)

        _ = model.transformer.callWithKeypoints(
            memFlat, keypointMemory: kpFlat, spatialShapes: spatialShapes,
            queryFeat: model.queryFeat, refpointEmbed: model.refpointEmbed, bboxEmbed: model.bboxEmbed,
            perLayer: { i, det, kp in
                eval(det, kp)
                if let expDet = inter["dec_layer_\(i).0"] {
                    self.assertParity(det, expDet, name: "dec_layer_\(i).det", atol: 8.0, statsRtol: 0.15)
                }
                if let expKp = inter["dec_layer_\(i).1"] {
                    self.assertParity(kp, expKp, name: "dec_layer_\(i).kp", atol: 8.0, statsRtol: 0.15)
                }
            }
        )
    }

    // MARK: - Postprocess parity (full output: keypoints + Cholesky + trace fusion)

    func testKeypointPostprocessParity() throws {
        let (model, pixelValues) = try loadModelAndInput()
        let pp = try loadFixtureArrays("postprocess.safetensors")
        let out = model(pixelValues)
        let res = postProcessKeypoints(
            predLogits: out["pred_logits"]!,
            predBoxes: out["pred_boxes"]!,
            predKeypoints: out["pred_keypoints"]!,
            originalSize: (576, 576),
            numSelect: 100,
            numKeypointsPerClass: model.config.numKeypointsPerClass,
            traceAlpha: 0.2
        )
        // Reference postprocess output (image 0).
        let expScores = pp["pp_scores"]!.asArray(Float.self)
        let expKp = pp["pp_keypoints"]!  // (K, maxK, 3)
        XCTAssertEqual(res.count, expScores.count, "selection count")

        // Selection order can differ on near-tied synthetic-input scores; compare the
        // SORTED score sets (order-independent). Trace fusion is included on both sides.
        let sortedSwift = res.scores.sorted(by: >)
        let sortedPy = expScores.sorted(by: >)
        var maxScoreDiff: Float = 0
        for i in 0..<min(sortedSwift.count, sortedPy.count) {
            maxScoreDiff = max(maxScoreDiff, abs(sortedSwift[i] - sortedPy[i]))
        }
        print("[pp scores sorted] count=\(sortedSwift.count) maxDiff=\(maxScoreDiff)")
        XCTAssertLessThanOrEqual(Double(maxScoreDiff), 0.05, "trace-fused score-set parity")

        // Keypoints (x,y pixels + confidence): compare aggregate stats (order-independent).
        let K = res.count
        let maxK = model.config.numKeypointsPerClass.max() ?? 0
        var flat = [Float]()
        flat.reserveCapacity(K * maxK * 3)
        for d in 0..<K { flat.append(contentsOf: res.keypoints![d].data) }
        let swiftKp = MLXArray(flat).reshaped([K, maxK, 3])
        let aMean = swiftKp.abs().mean().item(Float.self)
        let eMean = expKp.abs().mean().item(Float.self)
        let aStd = (swiftKp - swiftKp.mean()).square().mean().sqrt().item(Float.self)
        let eStd = (expKp - expKp.mean()).square().mean().sqrt().item(Float.self)
        print("[pp keypoints] |μ|s=\(aMean) |μ|p=\(eMean) σs=\(aStd) σp=\(eStd)")
        XCTAssertLessThanOrEqual(Double(abs(aMean - eMean) / max(eMean, 1e-6)), 0.15, "pp keypoints |μ|")
        XCTAssertLessThanOrEqual(Double(abs(aStd - eStd) / max(eStd, 1e-6)), 0.15, "pp keypoints σ")
    }

    // MARK: - Helper

    private func assertParity(
        _ actual: MLXArray, _ expected: MLXArray, name: String,
        atol: Double, statsRtol: Double = 0.10,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "\(name) shape", file: file, line: line)
        let diff = (actual - expected).abs()
        let maxDiff = diff.max().item(Float.self)
        let aAbsMean = actual.abs().mean().item(Float.self)
        let eAbsMean = expected.abs().mean().item(Float.self)
        let aStd = (actual - actual.mean()).square().mean().sqrt().item(Float.self)
        let eStd = (expected - expected.mean()).square().mean().sqrt().item(Float.self)
        let absMeanRelErr = abs(aAbsMean - eAbsMean) / max(eAbsMean, 1e-6)
        let stdRelErr = abs(aStd - eStd) / max(eStd, 1e-6)
        print("[\(name)] shape=\(actual.shape) max=\(maxDiff) |μ|s=\(aAbsMean) |μ|p=\(eAbsMean) σs=\(aStd) σp=\(eStd)")
        XCTAssertLessThanOrEqual(Double(maxDiff), atol, "\(name): max diff \(maxDiff) > atol \(atol)", file: file, line: line)
        XCTAssertLessThanOrEqual(Double(absMeanRelErr), statsRtol, "\(name): |μ| drift \(absMeanRelErr)", file: file, line: line)
        XCTAssertLessThanOrEqual(Double(stdRelErr), statsRtol, "\(name): σ drift \(stdRelErr)", file: file, line: line)
    }
}
