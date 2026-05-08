import MLX
import XCTest

@testable import MLXRFDETR

/// End-to-end load + forward tests against real converted checkpoints.
///
/// Skipped when the fixture directory is not present. Set the env var
/// `RFDETR_FIXTURES` to override the default search path.
final class LoaderTests: XCTestCase {

    private var fixturesRoot: URL? {
        if let env = ProcessInfo.processInfo.environment["RFDETR_FIXTURES"] {
            return URL(fileURLWithPath: env)
        }
        // Fallback: sibling python checkout used during development.
        let candidate = URL(fileURLWithPath: "/Users/mnmly/Development-local/GitHub/python/rf-detr")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    private func loadIfAvailable(_ subdir: String) throws -> (RFDETRModel, RFDETRProcessor, RFDETRVariant?)? {
        guard let root = fixturesRoot else {
            throw XCTSkip("Set RFDETR_FIXTURES to a directory containing converted models.")
        }
        let dir = root.appendingPathComponent(subdir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Missing fixture: \(dir.path)")
        }
        return try RFDETR.load(directory: dir)
    }

    func testLoadAndForwardSmall() throws {
        guard let (model, processor, variant) = try loadIfAvailable("rfdetr-small-mlx") else { return }
        XCTAssertEqual(variant, .small)
        XCTAssertEqual(processor.resolution, 512)

        let res = processor.resolution
        let pixels = MLXArray.zeros([1, res, res, 3])
        let out = model(pixels)
        eval(out["pred_logits"]!, out["pred_boxes"]!)

        XCTAssertEqual(out["pred_logits"]?.shape, [1, 300, 91])
        XCTAssertEqual(out["pred_boxes"]?.shape, [1, 300, 4])
        XCTAssertNil(out["pred_masks"])
    }

    func testLoadAndForwardSegSmall() throws {
        guard let (model, processor, variant) = try loadIfAvailable("rfdetr-seg-small-mlx") else { return }
        XCTAssertEqual(variant, .segSmall)
        XCTAssertEqual(processor.resolution, 384)

        let res = processor.resolution
        let pixels = MLXArray.zeros([1, res, res, 3])
        let out = model(pixels)
        eval(out["pred_logits"]!, out["pred_boxes"]!)

        XCTAssertEqual(out["pred_logits"]?.shape, [1, 100, 91])
        XCTAssertEqual(out["pred_boxes"]?.shape, [1, 100, 4])
        let masks = try XCTUnwrap(out["pred_masks"])
        eval(masks)
        // (B, Q, H', W') with H' = res / downsampleRatio (default 4).
        XCTAssertEqual(masks.shape, [1, 100, res / 4, res / 4])
    }

    func testLoadAndForwardSegLarge() throws {
        guard let (model, processor, variant) = try loadIfAvailable("rfdetr-seg-large-mlx") else { return }
        XCTAssertEqual(variant, .segLarge)
        XCTAssertEqual(processor.resolution, 504)

        let res = processor.resolution
        let pixels = MLXArray.zeros([1, res, res, 3])
        let out = model(pixels)
        eval(out["pred_logits"]!, out["pred_boxes"]!)

        XCTAssertEqual(out["pred_logits"]?.shape, [1, 300, 91])
        XCTAssertEqual(out["pred_boxes"]?.shape, [1, 300, 4])
        let masks = try XCTUnwrap(out["pred_masks"])
        eval(masks)
        XCTAssertEqual(masks.shape, [1, 300, res / 4, res / 4])
    }

    func testSegPredictorReturnsMasks() throws {
        guard let (model, processor, _) = try loadIfAvailable("rfdetr-seg-small-mlx") else { return }
        let predictor = RFDETRPipeline(
            model: model, processor: processor,
            scoreThreshold: 0.0, nmsThreshold: 1.0  // keep everything to verify the mask path
        )
        let res = processor.resolution
        let pixels = MLXArray.zeros([1, res, res, 3])
        let result = try predictor.predict(pixelValues: pixels, originalSize: (res, res))
        XCTAssertGreaterThan(result.count, 0)
        let masks = try XCTUnwrap(result.masks)
        XCTAssertEqual(masks.count, result.count)
        XCTAssertEqual(masks[0].rows, res / 4)
        XCTAssertEqual(masks[0].cols, res / 4)
    }

    func testPredictorOnZeros() throws {
        guard let (model, processor, _) = try loadIfAvailable("rfdetr-small-mlx") else { return }
        let predictor = RFDETRPipeline(
            model: model, processor: processor,
            scoreThreshold: 0.5, nmsThreshold: 0.5
        )
        let res = processor.resolution
        let pixels = MLXArray.zeros([1, res, res, 3])
        let result = try predictor.predict(pixelValues: pixels, originalSize: (res, res))
        // No detections expected on a zero input, but call should not crash and
        // result fields should all align in length.
        XCTAssertEqual(result.boxes.count, result.scores.count)
        XCTAssertEqual(result.boxes.count, result.classNames.count)
    }
}
