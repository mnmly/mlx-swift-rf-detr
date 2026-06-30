import XCTest

@testable import MLXRFDETR

/// Unit tests for OKS-based pose NMS. Pure CPU math — no model or fixtures, so
/// these always run.
final class KeypointNMSTests: XCTestCase {

    /// A 17-keypoint COCO-person pose laid out as a simple vertical strip anchored
    /// at `(originX, originY)`. Only the relative geometry matters for OKS.
    private func pose(originX: Float, originY: Float, conf: Float = 0.9) -> Array2D<Float> {
        var data = [Float]()
        data.reserveCapacity(17 * 3)
        for i in 0..<17 {
            data.append(originX + Float(i % 3) * 10)  // small horizontal spread
            data.append(originY + Float(i) * 15)      // down the body
            data.append(conf)
        }
        return Array2D(data: data, rows: 17, cols: 3)
    }

    func testOksIdenticalIsOneAndDisjointIsZero() {
        let a = pose(originX: 100, originY: 100)
        let identical = objectKeypointSimilarity(a, areaA: 18000, a, areaB: 18000, count: 17)
        XCTAssertEqual(identical, 1.0, accuracy: 1e-5)

        let far = pose(originX: 1000, originY: 100)
        let disjoint = objectKeypointSimilarity(a, areaA: 18000, far, areaB: 18000, count: 17)
        XCTAssertLessThan(disjoint, 0.01)
    }

    /// The regression: a duplicate/offset skeleton on the same person must be
    /// removed, while a distinct nearby person is preserved.
    func testOksNmsRemovesDuplicateSkeleton() {
        let a = pose(originX: 100, originY: 100)  // person 1
        let b = pose(originX: 104, originY: 100)  // 4px-offset duplicate of person 1
        let c = pose(originX: 500, originY: 100)  // distinct person 2

        let boxes: [[Float]] = [
            [100, 100, 160, 400],  // A
            [104, 100, 164, 400],  // B (overlaps A)
            [500, 100, 560, 400],  // C
        ]
        let scores: [Float] = [0.9, 0.6, 0.7]
        let labels = [1, 1, 1]

        let keep = oksNmsKeep(
            boxes: boxes, scores: scores, labels: labels, keypoints: [a, b, c],
            numKeypointsPerClass: [0, 17], oksThreshold: 0.7
        )

        // B (the duplicate) is suppressed; A and C survive, score-ordered.
        XCTAssertEqual(keep, [0, 2])
    }

    func testOksNmsThresholdOfOneKeepsOffsetDetections() {
        let a = pose(originX: 100, originY: 100)
        let b = pose(originX: 104, originY: 100)
        let keep = oksNmsKeep(
            boxes: [[100, 100, 160, 400], [104, 100, 164, 400]],
            scores: [0.9, 0.6], labels: [1, 1], keypoints: [a, b],
            numKeypointsPerClass: [0, 17], oksThreshold: 1.0
        )
        XCTAssertEqual(Set(keep), Set([0, 1]))
    }

    /// Classes with no keypoints can't be compared on OKS and must pass through.
    func testKeypointlessClassPassesThrough() {
        let empty = Array2D(data: [Float](repeating: 0, count: 17 * 3), rows: 17, cols: 3)
        let keep = oksNmsKeep(
            boxes: [[0, 0, 10, 10], [0, 0, 10, 10]],
            scores: [0.9, 0.8], labels: [0, 0], keypoints: [empty, empty],
            numKeypointsPerClass: [0, 17], oksThreshold: 0.7
        )
        XCTAssertEqual(Set(keep), Set([0, 1]))
    }
}
