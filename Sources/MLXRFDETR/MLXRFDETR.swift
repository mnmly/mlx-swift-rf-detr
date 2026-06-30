// MLXRFDETR — pure mlx-swift port of RF-DETR.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/__init__.py

import Foundation
import MLX

public enum MLXRFDETR {
    public static let version = "0.2.0"

    /// One-line convenience: load a converted model directory and return a
    /// ready-to-use predictor. Mirrors `RFDETR.load(directory:dtype:)` and
    /// wraps the result in a default `RFDETRPipeline`.
    public static func fromPretrained(
        _ directory: URL,
        dtype: DType = .float16,
        scoreThreshold: Float = 0.5,
        nmsThreshold: Float = 0.5,
        classNames: [String]? = nil
    ) throws -> RFDETRPipeline {
        let (model, processor, _) = try RFDETR.load(directory: directory, dtype: dtype)
        return RFDETRPipeline(
            model: model,
            processor: processor,
            scoreThreshold: scoreThreshold,
            nmsThreshold: nmsThreshold,
            classNames: classNames
        )
    }
}
