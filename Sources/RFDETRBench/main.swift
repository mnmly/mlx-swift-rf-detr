import Foundation
import MLX
import MLXRFDETR

struct BenchmarkResult: Codable {
    let label: String
    let iterations: Int
    let warmup: Int
    let dtype: String
    let shape: [Int]
    let stageStatsMs: [String: StageStats]
    let totalStatsMs: StageStats
}

struct StageStats: Codable {
    let mean: Double
    let median: Double
    let min: Double
    let max: Double
    let stddev: Double
}

struct BenchConfig {
    var fixtures = "Tests/fixtures"
    var iterations = 20
    var warmup = 5
    var dtype = DType.float32
    var label = "swift-mlx-small"
    var diagnose = false
}

enum BenchError: Error, LocalizedError {
    case missingInput(String)

    var errorDescription: String? {
        switch self {
        case .missingInput(let name):
            return "Missing tensor '\(name)' in fixture file."
        }
    }
}

do {
    let config = try parseArgs(Array(CommandLine.arguments.dropFirst()))
    let fixtureURL = URL(fileURLWithPath: config.fixtures, isDirectory: true)
    let model = try loadFixtureModel(from: fixtureURL, dtype: config.dtype)
    let pixelValues = try loadFixtureInput(from: fixtureURL)

    if config.diagnose {
        runBackboneDiagnose(
            model: model,
            pixelValues: pixelValues,
            iterations: config.iterations,
            warmup: config.warmup
        )
    } else {
        let benchmark = runBenchmark(
            label: config.label,
            model: model,
            pixelValues: pixelValues,
            iterations: config.iterations,
            warmup: config.warmup,
            dtype: config.dtype
        )

        let data = try JSONEncoder.pretty.encode(benchmark)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
    exit(1)
}

private func parseArgs(_ args: [String]) throws -> BenchConfig {
    var config = BenchConfig()
    var index = 0

    while index < args.count {
        switch args[index] {
        case "--fixtures":
            index += 1
            config.fixtures = args[index]
        case "--iterations":
            index += 1
            config.iterations = Int(args[index]) ?? config.iterations
        case "--warmup":
            index += 1
            config.warmup = Int(args[index]) ?? config.warmup
        case "--dtype":
            index += 1
            config.dtype = try parseDType(args[index])
        case "--label":
            index += 1
            config.label = args[index]
        case "--diagnose":
            config.diagnose = true
        default:
            throw NSError(domain: "RFDETRBench", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Unknown argument: \(args[index])"
            ])
        }
        index += 1
    }

    return config
}

private func parseDType(_ raw: String) throws -> DType {
    switch raw.lowercased() {
    case "float16", "fp16":
        return .float16
    case "float32", "fp32":
        return .float32
    default:
        throw NSError(domain: "RFDETRBench", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "Unsupported dtype: \(raw)"
        ])
    }
}

private func loadFixtureModel(from fixtureURL: URL, dtype: DType) throws -> RFDETRModel {
    let bb = DINOv2Backbone(
        imgSize: 512,
        patchSize: 16,
        embedDim: 384,
        depth: 12,
        numHeads: 6,
        numWindows: 2,
        featureIndices: [2, 5, 8, 11]
    )
    let proj = MultiScaleProjector(
        scaleFactors: [1.0],
        inChannelsList: [384, 384, 384, 384],
        hiddenDim: 256
    )
    let model = RFDETRModel(config: .small, backbone: bb, projector: proj)
    let weightsURL = fixtureURL.appendingPathComponent("weights.safetensors")
    try loadWeights(url: weightsURL, into: model, dtype: dtype)
    eval(model)
    return model
}

private func loadFixtureInput(from fixtureURL: URL) throws -> MLXArray {
    let inputURL = fixtureURL.appendingPathComponent("input.safetensors")
    let inputTensors = try MLX.loadArrays(url: inputURL, stream: .cpu)
    guard let pixelValues = inputTensors["pixel_values"] else {
        throw BenchError.missingInput("pixel_values")
    }
    return pixelValues
}

private func runBenchmark(
    label: String,
    model: RFDETRModel,
    pixelValues: MLXArray,
    iterations: Int,
    warmup: Int,
    dtype: DType
) -> BenchmarkResult {
    var stageSamples: [String: [Double]] = [
        "backbone": [],
        "projector": [],
        "transformer": [],
        "heads": [],
    ]
    var totalSamples: [Double] = []

    for step in 0..<(warmup + iterations) {
        var currentStages: [String: Double] = [:]
        let totalStart = now()

        let backboneStart = now()
        let features = model.backbone(pixelValues)
        eval(features[0], features[1], features[2], features[3])
        currentStages["backbone"] = elapsedMs(since: backboneStart)

        let projectorStart = now()
        let memories = model.projector(features)
        eval(memories[0])
        currentStages["projector"] = elapsedMs(since: projectorStart)

        let transformerStart = now()
        let spatialShapes = memories.map { ($0.dim(1), $0.dim(2)) }
        let memoryFlat = concatenated(memories.map { $0.reshaped([$0.dim(0), -1, $0.dim(-1)]) }, axis: 1)
        let (hs, refPoints) = model.transformer(
            memoryFlat,
            spatialShapes: spatialShapes,
            queryFeat: model.queryFeat,
            refpointEmbed: model.refpointEmbed,
            bboxEmbed: model.bboxEmbed
        )
        eval(hs, refPoints)
        currentStages["transformer"] = elapsedMs(since: transformerStart)

        let headsStart = now()
        let predLogits = model.classEmbed(hs)
        let predBoxes: MLXArray
        if model.config.bboxReparam {
            let delta = model.bboxEmbed(hs)
            let dcParts = delta.split(parts: 2, axis: -1)
            let rpParts = refPoints.split(parts: 2, axis: -1)
            let predCxcy = dcParts[0] * rpParts[1] + rpParts[0]
            let predWH = exp(dcParts[1]) * rpParts[1]
            predBoxes = concatenated([predCxcy, predWH], axis: -1)
        } else {
            predBoxes = sigmoid(model.bboxEmbed(hs) + inverseSigmoid(refPoints))
        }
        eval(predLogits, predBoxes)
        currentStages["heads"] = elapsedMs(since: headsStart)

        let total = elapsedMs(since: totalStart)

        if step >= warmup {
            for (key, value) in currentStages {
                stageSamples[key, default: []].append(value)
            }
            totalSamples.append(total)
        }
    }

    let stageStats = Dictionary(uniqueKeysWithValues: stageSamples.map { key, values in
        (key, summarize(values))
    })

    return BenchmarkResult(
        label: label,
        iterations: iterations,
        warmup: warmup,
        dtype: String(describing: dtype),
        shape: pixelValues.shape.map { Int($0) },
        stageStatsMs: stageStats,
        totalStatsMs: summarize(totalSamples)
    )
}

private func runBackboneDiagnose(
    model: RFDETRModel,
    pixelValues: MLXArray,
    iterations: Int,
    warmup: Int
) {
    let bb = model.backbone
    let nW = bb.numWindows
    let nW2 = nW * nW
    let depth = bb.blocks.count

    var patchEmbedSamples: [Double] = []
    var tokenSetupSamples: [Double] = []
    var perBlockSamples: [[Double]] = Array(repeating: [], count: depth)
    var fullVsWindowed: [String: [Double]] = ["windowed": [], "full": []]

    for step in 0..<(warmup + iterations) {
        // Stage A: patch embed
        let aStart = now()
        let N = pixelValues.dim(0)
        let (patches, H, W) = bb.patchEmbed(pixelValues)
        eval(patches)
        let aMs = elapsedMs(since: aStart)

        // Stage B: token setup (cls/posEmbed/window-partition)
        let bStart = now()
        let cls = MLX.broadcast(bb.clsToken, to: [N, 1, bb.embedDim])
        var tokens = MLX.concatenated([cls, patches], axis: 1) + bb.posEmbed
        let clsSlice = tokens[0..., 0..<1, 0...]
        let patchTokens = tokens[0..., 1..., 0...]
        let winPatches = bb.windowPartition(patchTokens, H: H, W: W, N: N)
        let winClsBase = MLX.broadcast(clsSlice, to: [N, 1, bb.embedDim])
        let winCls = MLX.concatenated(Array(repeating: winClsBase, count: nW2), axis: 0)
        tokens = MLX.concatenated([winCls, winPatches], axis: 1)
        eval(tokens)
        let bMs = elapsedMs(since: bStart)

        // Stage C: per-block
        var blockMs = [Double](repeating: 0, count: depth)
        for i in 0..<depth {
            let cStart = now()
            let runFull = bb.fullAttnLayers.contains(i)
            tokens = bb.blocks[i](tokens, runFullAttention: runFull)
            eval(tokens)
            blockMs[i] = elapsedMs(since: cStart)
        }

        if step >= warmup {
            patchEmbedSamples.append(aMs)
            tokenSetupSamples.append(bMs)
            for i in 0..<depth {
                perBlockSamples[i].append(blockMs[i])
                if bb.fullAttnLayers.contains(i) {
                    fullVsWindowed["full", default: []].append(blockMs[i])
                } else {
                    fullVsWindowed["windowed", default: []].append(blockMs[i])
                }
            }
        }
    }

    func median(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        guard !s.isEmpty else { return 0 }
        return s.count.isMultiple(of: 2)
            ? (s[s.count / 2 - 1] + s[s.count / 2]) / 2.0
            : s[s.count / 2]
    }

    print("=== Backbone diagnose (\(iterations) iter, \(warmup) warmup) ===")
    print(String(format: "patch_embed:  %.3f ms median", median(patchEmbedSamples)))
    print(String(format: "token_setup:  %.3f ms median", median(tokenSetupSamples)))
    print()
    print("Per-block (median ms):")
    for i in 0..<depth {
        let kind = bb.fullAttnLayers.contains(i) ? "full" : "win "
        print(String(format: "  block %2d [%@]: %.3f", i, kind, median(perBlockSamples[i])))
    }
    print()
    let winSamples = fullVsWindowed["windowed"] ?? []
    let fullSamples = fullVsWindowed["full"] ?? []
    print(String(format: "windowed avg per block: %.3f ms (n=%d samples across %d blocks)",
                 winSamples.reduce(0, +) / Double(max(winSamples.count, 1)),
                 winSamples.count, depth - bb.fullAttnLayers.count))
    print(String(format: "full     avg per block: %.3f ms (n=%d samples across %d blocks)",
                 fullSamples.reduce(0, +) / Double(max(fullSamples.count, 1)),
                 fullSamples.count, bb.fullAttnLayers.count))
    let totalBlock = (0..<depth).map { median(perBlockSamples[$0]) }.reduce(0, +)
    print(String(format: "sum of per-block medians: %.3f ms", totalBlock))
    print(String(format: "patch_embed + token_setup + blocks ≈ %.3f ms",
                 median(patchEmbedSamples) + median(tokenSetupSamples) + totalBlock))
}

private func now() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

private func elapsedMs(since start: UInt64) -> Double {
    Double(now() - start) / 1_000_000.0
}

private func summarize(_ values: [Double]) -> StageStats {
    let sorted = values.sorted()
    let count = Double(values.count)
    let mean = values.reduce(0.0, +) / count
    let variance = values.reduce(0.0) { partial, value in
        let delta = value - mean
        return partial + (delta * delta)
    } / count
    let median: Double
    if sorted.count.isMultiple(of: 2) {
        median = (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2.0
    } else {
        median = sorted[sorted.count / 2]
    }
    return StageStats(
        mean: mean,
        median: median,
        min: sorted.first ?? 0,
        max: sorted.last ?? 0,
        stddev: sqrt(variance)
    )
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}
