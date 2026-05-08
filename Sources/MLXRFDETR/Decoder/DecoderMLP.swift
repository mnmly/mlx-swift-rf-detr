// Variable-depth MLP with ReLU activation.
// Used for bbox_embed, ref_point_head, enc_out_bbox_embed.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/transformer.py (MLP class)

import Foundation
import MLX
import MLXNN

/// Variable-depth MLP with ReLU between layers.
/// Last layer has no activation.
public final class DecoderMLP: Module {
    @ModuleInfo(key: "layers") public var layers: [Linear]

    public init(inputDim: Int, hiddenDim: Int, outputDim: Int, numLayers: Int) {
        var dims: [Int] = [inputDim]
        dims.append(contentsOf: Array(repeating: hiddenDim, count: numLayers - 1))
        dims.append(outputDim)
        self._layers = ModuleInfo(
            wrappedValue: (0..<numLayers).map { Linear(dims[$0], dims[$0 + 1]) },
            key: "layers"
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for (i, layer) in layers.enumerated() {
            y = layer(y)
            if i < layers.count - 1 {
                y = relu(y)
            }
        }
        return y
    }
}
