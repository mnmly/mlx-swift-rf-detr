// Bilinear grid_sample Metal kernel with CustomFunction wrapper.
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/ops/functions/ms_deform_attn_func.py
//            (bilinear grid_sample inside ms_deform_attn_core_pytorch)
// ADAPTED FROM: mlx-swift CustomFunctionExample.swift

import Foundation
import MLX
import MLXFast

// MARK: - Metal forward kernel

private let forwardSource = """
    uint elem = thread_position_in_grid.x;

    int B = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];
    int C = x_shape[3];
    int gH = grid_shape[1];
    int gW = grid_shape[2];

    int w_stride = C;
    int h_stride = W * w_stride;
    int b_stride = H * h_stride;

    // Decode (b, h, w, c) from flat elem
    int c = elem % C;
    int w = (elem / C) % gW;
    int h = (elem / (C * gW)) % gH;
    int b = elem / (C * gW * gH);

    if (b >= B) return;

    uint grid_idx = ((b * gH + h) * gW + w) * 2;
    // align_corners=False (matches torch.nn.functional.grid_sample): ((g+1)*W - 1) / 2
    float ix = ((grid[grid_idx] + 1) * W - 1) / 2.0;
    float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2.0;

    int ix_nw = floor(ix);
    int iy_nw = floor(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    float nw = (ix_se - ix)    * (iy_se - iy);
    float ne = (ix    - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix)    * (iy    - iy_ne);
    float se = (ix    - ix_nw) * (iy    - iy_nw);

    int base_idx = b * b_stride + c;
    float I_nw = 0.0;
    float I_ne = 0.0;
    float I_sw = 0.0;
    float I_se = 0.0;

    if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W)
        I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
    if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W)
        I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
    if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W)
        I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
    if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W)
        I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

    int out_idx = ((b * gH + h) * gW + w) * C + c;
    out[out_idx] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
    """

// MARK: - Metal VJP (backward) kernel

private let vjpSource = """
    uint elem = thread_position_in_grid.x;

    int B = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];
    int C = x_shape[3];
    int gH = grid_shape[1];
    int gW = grid_shape[2];

    int w_stride = C;
    int h_stride = W * w_stride;
    int b_stride = H * h_stride;

    int c = elem % C;
    int w = (elem / C) % gW;
    int h = (elem / (C * gW)) % gH;
    int b = elem / (C * gW * gH);

    if (b >= B) return;

    uint grid_idx = ((b * gH + h) * gW + w) * 2;
    // align_corners=False (matches torch.nn.functional.grid_sample): ((g+1)*W - 1) / 2
    float ix = ((grid[grid_idx] + 1) * W - 1) / 2.0;
    float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2.0;

    int ix_nw = floor(ix);
    int iy_nw = floor(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    float dx = ix - ix_nw;
    float dy = iy - iy_nw;

    float nw = (1 - dx) * (1 - dy);
    float ne = dx * (1 - dy);
    float sw = (1 - dx) * dy;
    float se = dx * dy;

    int base_idx = b * b_stride + c;

    // x_grad: scatter cotangent back to source pixels
    if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W)
        atomic_fetch_add_explicit(&x_grad[base_idx + iy_nw * h_stride + ix_nw * w_stride], nw, memory_order_relaxed);
    if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W)
        atomic_fetch_add_explicit(&x_grad[base_idx + iy_ne * h_stride + ix_ne * w_stride], ne, memory_order_relaxed);
    if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W)
        atomic_fetch_add_explicit(&x_grad[base_idx + iy_sw * h_stride + ix_sw * w_stride], sw, memory_order_relaxed);
    if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W)
        atomic_fetch_add_explicit(&x_grad[base_idx + iy_se * h_stride + ix_se * w_stride], se, memory_order_relaxed);

    // grid_grad: compute dL/dgrid at this (x,y) position
    float gix = 0.0;
    float giy = 0.0;

    if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W) {
        float val = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        gix += -(1 - dy) * val;
        giy += -(1 - dx) * val;
    }
    if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W) {
        float val = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        gix += (1 - dy) * val;
        giy += -dx * val;
    }
    if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W) {
        float val = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        gix += -dy * val;
        giy += (1 - dx) * val;
    }
    if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W) {
        float val = x[base_idx + iy_se * h_stride + ix_se * w_stride];
        gix += dy * val;
        giy += dx * val;
    }

    // Normalize to [-1,1] coordinate gradient (align_corners=False: dix/dgrid = W/2)
    gix *= float(W) / 2.0;
    giy *= float(H) / 2.0;

    atomic_fetch_add_explicit(&grid_grad[grid_idx], gix, memory_order_relaxed);
    atomic_fetch_add_explicit(&grid_grad[grid_idx + 1], giy, memory_order_relaxed);
    """

// MARK: - Lazy kernel compilation

private let forwardKernel: MLXFast.MLXFastKernel = {
    MLXFast.metalKernel(
        name: "grid_sample_forward",
        inputNames: ["x", "grid"],
        outputNames: ["out"],
        source: forwardSource
    )
}()

private let vjpKernel: MLXFast.MLXFastKernel = {
    MLXFast.metalKernel(
        name: "grid_sample_vjp",
        inputNames: ["x", "grid", "cotangent"],
        outputNames: ["x_grad", "grid_grad"],
        source: vjpSource,
        atomicOutputs: true
    )
}()

// MARK: - CustomFunction wrapper

/// Bilinear grid sampling wrapped in a `CustomFunction` for autograd support.
///
/// Matches `mx.fast.metal_kernel("grid_sample", ...)` from the Python port.
///
/// - Parameters:
///   - x: Feature map `(B, H, W, C)` channel-last.
///   - grid: Sampling grid `(B, gH, gW, 2)` in normalized `[-1, 1]` coordinates.
/// - Returns: Sampled values `(B, gH, gW, C)`.
nonisolated(unsafe) public let gridSample = CustomFunction {
    Forward { inputs in
        let x = inputs[0]
        let grid = inputs[1]
        let totalElems = x.shape[0] * grid.shape[1] * grid.shape[2] * x.shape[3]

        return forwardKernel(
            [x, grid],
            grid: (totalElems, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [grid.shape.dropLast() + [x.shape[3]]],
            outputDTypes: [x.dtype]
        )
    }

    VJP { primals, cotangents in
        let x = primals[0]
        let grid = primals[1]
        let cot = cotangents[0]
        let totalElems = x.shape[0] * grid.shape[1] * grid.shape[2] * x.shape[3]

        return vjpKernel(
            [x, grid, cot],
            grid: (totalElems, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [x.shape, grid.shape],
            outputDTypes: [x.dtype, grid.dtype],
            initValue: 0
        )
    }
}
