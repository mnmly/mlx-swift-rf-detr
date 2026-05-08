// Weight loading and key sanitization for RF-DETR.
//
// Converts HuggingFace PyTorch safetensors keys to MLX-Swift module tree keys,
// including Conv2d weight transposition and fused QKV splitting.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/lwdetr.py (state_dict keys)

import Foundation
import MLX
import MLXNN

/// Load RF-DETR weights from a safetensors file and update the model in place.
///
/// - Parameters:
///   - url: path to `model.safetensors` file
///   - model: the RFDETRModel instance
///   - dtype: target dtype for all parameters (e.g. `.float16` or `.float32`)
///   - verify: update verification flags (default `.noUnusedKeys` catches remapping errors)
///
/// - Throws: MLX load error, or updath error if keys don't match the module tree.
public func loadWeights(url: URL, into model: RFDETRModel, dtype: DType = .float16, verify: Module.VerifyUpdate = [.noUnusedKeys]) throws {
    let weights = try loadArrays(url: url)

    var remapped = [(String, MLXArray)]()
    remapped.reserveCapacity(weights.count)

    for (key, value) in weights {
        // Transpose + remap in one pass (may produce multiple entries for fused QKV)
        let results = sanitized(key: key, value: value.asType(dtype))
        remapped.append(contentsOf: results)
    }

    // RF-DETR checkpoints inherit DINOv2's pretrained pos_embed, whose grid size
    // can differ from `(resolution / patch_size)²`. Python rfdetr interpolates
    // at every forward pass; we do it once at load time to match the backbone's
    // static buffer shape.
    let targetPosLen = model.backbone.posEmbed.dim(1)
    for i in 0..<remapped.count {
        let (k, v) = remapped[i]
        guard k == "backbone.pos_embed", v.dim(1) != targetPosLen else { continue }
        remapped[i] = (k, resamplePosEmbed(v, targetTokens: targetPosLen))
    }

    try model.update(
        parameters: ModuleParameters.unflattened(remapped),
        verify: verify
    )
}

/// Bilinearly resample `pos_embed` of shape `(1, 1 + storedSide², D)` to
/// `(1, targetTokens, D)` where `targetTokens - 1` is a perfect square.
private func resamplePosEmbed(_ posEmbed: MLXArray, targetTokens: Int) -> MLXArray {
    let D = posEmbed.dim(2)
    let storedPatches = posEmbed.dim(1) - 1
    let targetPatches = targetTokens - 1
    let storedSide = Int(Double(storedPatches).squareRoot().rounded())
    let targetSide = Int(Double(targetPatches).squareRoot().rounded())

    let dtype = posEmbed.dtype
    let cls = posEmbed[0..., ..<1, 0...]                        // (1, 1, D)
    let patch = posEmbed[0..., 1..., 0...].asType(.float32)     // upcast for resample
    let grid = patch.reshaped([1, storedSide, storedSide, D])   // (1, sH, sW, D)
    let resized = interpolateSpatial(grid, targetH: targetSide, targetW: targetSide)
    let resizedFlat = resized.reshaped([1, targetSide * targetSide, D]).asType(dtype)
    return concatenated([cls, resizedFlat], axis: 1)
}

// MARK: - Key sanitization

/// Remap a single HF safetensors key → (possibly multiple) Swift key(s).
/// Conv2d weights are transposed in-place. Fused QKV splits produce 3 entries.
///
/// Returns empty array for keys that should be skipped.
public func sanitized(key: String, value: MLXArray) -> [(String, MLXArray)] {
    // 1. Strip model. prefix
    guard key.hasPrefix("model.") else { return [] }
    var k = String(key.dropFirst("model.".count))

    // 2. Skip mask_token
    if k.contains("mask_token") { return [] }

    // 3. Remap backbone key structure
    k = remapBackbone(k)

    // 4. Remap projector key structure
    k = remapProjector(k)

    // 4b. nn.Embedding → bare ParameterInfo (Swift stores these as MLXArray, not Module)
    if k == "refpoint_embed.weight" { k = "refpoint_embed" }
    if k == "query_feat.weight" { k = "query_feat" }

    // 5. Handle fused QKV split for decoder self-attention
    if k.contains("in_proj_weight") || k.contains("in_proj_bias") {
        return splitFusedQKV(key: k, value: value)
    }

    // 6. Transpose Conv weights (PyTorch NCHW → MLX NHWC)
    var v = value
    if v.ndim == 4 {
        // ConvTranspose2d in stages_sampling: PyTorch (in, out, kH, kW) → MLX (out, kH, kW, in)
        if k.contains("stages_sampling") && !k.contains(".conv.") && k.hasSuffix(".weight") {
            v = v.transposed(1, 2, 3, 0)
        } else if k.lowercased().contains("conv") || k.contains("spatial_features_proj") || k.contains("patch_embed.proj") {
            v = v.transposed(0, 2, 3, 1)  // Conv2d: (O, I, kH, kW) → (O, kH, kW, I)
        }
    }

    return [(k, v)]
}

// MARK: - Backbone remapping

private func remapBackbone(_ key: String) -> String {
    var k = key
    // DINOv2 backbone structural remapping
    // cls_token / position_embeddings live at backbone level in Swift, not patch_embed
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.cls_token",
        with: "backbone.cls_token"
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.position_embeddings",
        with: "backbone.pos_embed"
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.register_tokens",
        with: "backbone.register_tokens"
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.",
        with: "backbone.patch_embed."
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.encoder.layer.",
        with: "backbone.blocks."
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.layernorm.",
        with: "backbone.norm."
    )
    // PatchEmbed: patch_embeddings.projection → proj
    k = k.replacingOccurrences(of: ".patch_embeddings.projection", with: ".proj")
    // Attention key remapping (HF → MLX)
    k = k.replacingOccurrences(of: ".attention.attention.query.", with: ".attn.q.")
    k = k.replacingOccurrences(of: ".attention.attention.key.", with: ".attn.k.")
    k = k.replacingOccurrences(of: ".attention.attention.value.", with: ".attn.v.")
    k = k.replacingOccurrences(of: ".attention.output.dense.", with: ".attn.out.")
    // LayerScale: strip only .lambda1 suffix, then rename .layer_scale1 → .ls1.gamma
    k = k.replacingOccurrences(of: ".layer_scale1.lambda1", with: ".layer_scale1")
    k = k.replacingOccurrences(of: ".layer_scale2.lambda1", with: ".layer_scale2")
    k = k.replacingOccurrences(of: ".layer_scale1", with: ".ls1.gamma")
    k = k.replacingOccurrences(of: ".layer_scale2", with: ".ls2.gamma")
    return k
}

// MARK: - Projector remapping

private func remapProjector(_ key: String) -> String {
    var k = key
    k = k.replacingOccurrences(of: "backbone.0.projector.", with: "projector.")

    // MLX Swift treats integer key segments as array indices, so we remap integer
    // sub-module keys to named strings to match our @ModuleInfo declarations.
    //
    // stages.N.0.{rest}  → stages.N.c2f.{rest}   (ProjectorStage.c2f)
    // stages.N.1.{rest}  → stages.N.norm.{rest}   (ProjectorStage.norm)
    // stages_sampling.N.M.0.{rest} → stages_sampling.N.M.op.{rest}  (FeatureSamplerStep.op)
    var parts = k.components(separatedBy: ".")
    if parts.count >= 4 && parts[0] == "projector" && parts[1] == "stages"
        && Int(parts[2]) != nil && (parts[3] == "0" || parts[3] == "1") {
        parts[3] = parts[3] == "0" ? "c2f" : "norm"
        k = parts.joined(separator: ".")
    } else if parts.count >= 5 && parts[0] == "projector" && parts[1] == "stages_sampling"
        && Int(parts[2]) != nil && Int(parts[3]) != nil && parts[4] == "0" {
        parts[4] = "op"
        k = parts.joined(separator: ".")
    }
    return k
}

// MARK: - Fused QKV split

/// Split fused `in_proj_weight` or `in_proj_bias` into separate q/k/v entries.
private func splitFusedQKV(key: String, value: MLXArray) -> [(String, MLXArray)] {
    let isWeight = key.hasSuffix("weight")
    let base = key.replacingOccurrences(of: "in_proj_weight", with: "")
        .replacingOccurrences(of: "in_proj_bias", with: "")

    if isWeight {
        // value shape: (3*d, d) where d = hidden_dim
        let d = value.dim(1)  // second dim is the feature dim
        return [
            ("\(base)q_proj.weight", value[..<d, 0...]),
            ("\(base)k_proj.weight", value[d..<(2 * d), 0...]),
            ("\(base)v_proj.weight", value[(2 * d)..., 0...]),
        ]
    } else {
        // value shape: (3*d,)
        let d = value.dim(0) / 3
        return [
            ("\(base)q_proj.bias", value[..<d]),
            ("\(base)k_proj.bias", value[d..<(2 * d)]),
            ("\(base)v_proj.bias", value[(2 * d)...]),
        ]
    }
}
