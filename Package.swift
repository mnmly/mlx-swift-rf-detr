// swift-tools-version: 6.0
// Port of rfdetr.mlx (Python) → mlx-swift. Inference-only DINOv2-windowed
// backbone + RF-DETR decoder with deformable cross-attention.
// PORT FROM: https://github.com/.../rf-detr-mlx (src/rfdetr/mlx)

import PackageDescription

let package = Package(
    name: "mlx-swift-rf-detr",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "RFDETRMLX", targets: ["RFDETRMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
    ],
    targets: [
        .target(
            name: "RFDETRMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "RFDETRMLXTests",
            dependencies: ["RFDETRMLX"]
        ),
    ]
)
