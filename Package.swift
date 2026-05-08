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
