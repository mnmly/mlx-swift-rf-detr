// swift-tools-version: 6.0
// Port of RF-DETR (PyTorch) → mlx-swift. Inference-only DINOv2-windowed
// backbone + RF-DETR decoder with deformable cross-attention.
// PORT FROM: https://github.com/roboflow/rf-detr (../../python/rf-detr)

import PackageDescription

let package = Package(
    name: "mlx-swift-rf-detr",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXRFDETR", targets: ["MLXRFDETR"]),
        .executable(name: "RFDETRBench", targets: ["RFDETRBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
    ],
    targets: [
        .target(
            name: "MLXRFDETR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXRFDETRTests",
            dependencies: ["MLXRFDETR"]
        ),
        .executableTarget(
            name: "RFDETRBench",
            dependencies: [
                "MLXRFDETR",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation, so normal builds
// and downstream consumers of MLXRFDETR don't have to resolve an extra dependency.
// Scripts/build_docs.sh exports BUILD_DOC=1; the Swift Package Index sets
// SPI_GENERATE_DOCS automatically.
if Context.environment["SPI_GENERATE_DOCS"] == "1"
    || Context.environment["BUILD_DOC"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
