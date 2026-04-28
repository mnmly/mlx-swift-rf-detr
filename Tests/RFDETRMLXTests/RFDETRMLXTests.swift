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
}
