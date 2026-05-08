// Multi-head self-attention for decoder queries.
//
// Position embedding is added to q and k only (not v).
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/transformer.py (decoder self-attn)

import Foundation
import MLX
import MLXFast
import MLXNN

/// Multi-head self-attention with position added to q,k only.
public final class DecoderSelfAttention: Module {
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(dModel: Int, nHeads: Int) {
        self.numHeads = nHeads
        self.headDim = dModel / nHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "v_proj")
        self._outProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "out_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, queryPos: MLXArray) -> MLXArray {
        let B = x.dim(0); let N = x.dim(1); let D = x.dim(2)
        let H = numHeads; let d = headDim

        let qkInput = x + queryPos
        let q = qProj(qkInput).reshaped([B, N, H, d]).transposed(0, 2, 1, 3)
        let k = kProj(qkInput).reshaped([B, N, H, d]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, N, H, d]).transposed(0, 2, 1, 3)

        var y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        y = y.transposed(0, 2, 1, 3).reshaped([B, N, D])
        return outProj(y)
    }
}
