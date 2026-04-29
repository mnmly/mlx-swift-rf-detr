// Utility functions for the RF-DETR decoder.
//
// PORT FROM: rf-detr-mlx/src/rfdetr/mlx/transformer.py

import Foundation
import MLX

// MARK: - inverse_sigmoid

/// Inverse sigmoid with epsilon clipping.
public func inverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let clamped = clip(x, min: eps, max: 1 - eps)
    return log(clamped / (1 - clamped))
}

// MARK: - gen_sineembed_for_position

/// Generate interleaved sine/cosine positional embeddings from coordinates.
///
/// Matches PyTorch DETR: interleaves sin (even) / cos (odd) per frequency,
/// and concatenates coordinates in (y, x, [w, h]) order.
///
/// - Parameters:
///   - pos: `(..., numCoords)` coordinates in [0, 1] or similar range.
///   - dModel: embedding dimension **per coordinate** (default 128).
/// - Returns: `(..., dModel * numCoords)` interleaved sine/cosine embeddings.
public func genSineembedForPosition(_ pos: MLXArray, dModel: Int = 128) -> MLXArray {
    let temperature: Float = 10000.0
    let scale: Float = 2 * Float.pi
    let numCoords = pos.dim(-1)

    // dim_t = temperature ** (2 * (dim_t // 2) / d_model) — note floor division
    let dimT = MLXArray(stride(from: 0, through: dModel - 1, by: 1), [dModel]).asType(.float32)
    let dimTFloor = floor(dimT / 2)
    let dimTExp = MLX.pow(MLXArray([temperature]), (2 * dimTFloor) / Float(dModel))

    // dimTExp shape (dModel,) broadcasts against coord (..., 1) to give (..., dModel)
    let dimTS = dimTExp

    func embedCoord(_ coord: MLXArray) -> MLXArray {
        // coord shape: (..., 1) — a single coord with last dim kept
        let embed = coord * scale / dimTS // (..., dModel)
        // Interleave [sin(even), cos(odd), sin(even), cos(odd), ...]
        let halves = embed.reshaped(Array(embed.shape.dropLast()) + [dModel / 2, 2])
        let sinPart = sin(halves[.ellipsis, 0])
        let cosPart = cos(halves[.ellipsis, 1])
        let interleaved = stacked([sinPart, cosPart], axis: -1)
        return interleaved.reshaped(Array(embed.shape.dropLast()) + [dModel])
    }

    // Split the last axis into individual coords, each (..., 1)
    let coordParts = pos.split(parts: numCoords, axis: -1)
    let embeds = coordParts.map { embedCoord($0) }
    if numCoords == 2 {
        return concatenated([embeds[1], embeds[0]], axis: -1)
    } else if numCoords == 4 {
        return concatenated([embeds[1], embeds[0], embeds[3], embeds[2]], axis: -1)
    } else {
        return concatenated(embeds, axis: -1)
    }
}

// MARK: - gen_encoder_output_proposals

/// Generate grid of anchor proposals in [0, 1] coordinate space.
///
/// - Parameters:
///   - H: feature map height.
///   - W: feature map width.
///   - scale: initial box size (fraction of image).
/// - Returns: `(H * W, 4)` proposals in `[cx, cy, w, h]` format, values in (0, 1).
public func genEncoderOutputProposals(H: Int, W: Int, scale: Float = 0.05) -> MLXArray {
    let gridY = (MLXArray(stride(from: 0, through: H - 1, by: 1), [H]).asType(.float32) + 0.5) / Float(H)
    let gridX = (MLXArray(stride(from: 0, through: W - 1, by: 1), [W]).asType(.float32) + 0.5) / Float(W)

    let yy = broadcast(gridY.expandedDimensions(axis: 1), to: [H, W])
    let xx = broadcast(gridX.expandedDimensions(axis: 0), to: [H, W])

    let ww = MLXArray.full([H, W], values: MLXArray(scale))
    let hh = MLXArray.full([H, W], values: MLXArray(scale))

    let stacked_ = stacked([xx, yy, ww, hh], axis: -1) // (H, W, 4)
    return stacked_.reshaped([-1, 4])
}
