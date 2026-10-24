import MLX
import XCTest

@testable import MLXRFDETR

/// Unit tests for the keypoint post-process score fusion and box clamping
/// (rf-detr ≥ 1.9.0 semantics). Synthetic single-query inputs — no model or fixtures.
final class KeypointPostProcessTests: XCTestCase {

    /// One query, background-first `[0, 17]` schema, with every person keypoint at the
    /// image center, findable logit 0, and an isotropic Cholesky `log_l11 = log_l22 = logL`.
    private func run(
        personLogit: Float, logL: Float, box: [Float], size: (Int, Int) = (100, 200)
    ) -> DetectionResult {
        let schema = [0, 17]
        let maxK = 17, D = 8
        var kps = [Float](repeating: 0, count: schema.count * maxK * D)
        for kk in 0..<maxK {
            let base = (maxK + kk) * D  // class 1 slots
            kps[base + 0] = 0.5
            kps[base + 1] = 0.5
            kps[base + 4] = logL
            kps[base + 6] = logL
        }
        return postProcessKeypoints(
            predLogits: MLXArray([-10, personLogit] as [Float]).reshaped([1, 1, 2]),
            predBoxes: MLXArray(box).reshaped([1, 1, 4]),
            predKeypoints: MLXArray(kps).reshaped([1, 1, schema.count * maxK, D]),
            originalSize: size,
            numSelect: 1,
            numKeypointsPerClass: schema,
            traceAlpha: 0.2
        )
    }

    func testTraceFusionIsLogSpaceSigmoid() {
        let res = run(personLogit: 2, logL: -1, box: [0.5, 0.5, 0.2, 0.2])
        XCTAssertEqual(res.labels, [1])
        let s: Float = 1 / (1 + exp(-2))
        // trace(Σ) per keypoint = 2·exp(-2·logL) (l21 = 0); all keypoints equal → mean = itself.
        let logMeanTrace = log(Float(2)) + 2
        let expected = 1 / (1 + exp(-(log(s) - 0.2 * logMeanTrace)))
        XCTAssertEqual(res.scores[0], expected, accuracy: 1e-4)
        // Distinguish from the pre-1.9 multiplicative penalty `s · exp(-α·logMeanTrace)`.
        XCTAssertGreaterThan(abs(res.scores[0] - s * exp(-0.2 * logMeanTrace)), 0.1)
    }

    func testSaturatedFusedScoreStaysBelowOne() {
        // Confident + extremely sharp keypoints → fused logit far above float sigmoid saturation.
        let res = run(personLogit: 20, logL: 200, box: [0.5, 0.5, 0.2, 0.2])
        XCTAssertLessThan(res.scores[0], 1)
        XCTAssertTrue(res.scores[0].isFinite)
    }

    func testBoxesClampedToImageBounds() {
        // cx ± w/2 = [-0.1, 1.1] (x), cy ± h/2 = [0.8, 1.2] (y) — overflows on three sides.
        let res = run(personLogit: 2, logL: 0, box: [0.5, 1.0, 1.2, 0.4], size: (100, 200))
        XCTAssertEqual(res.boxes[0], [0, 80, 200, 100])
    }
}
