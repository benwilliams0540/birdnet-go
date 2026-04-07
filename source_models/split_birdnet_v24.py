#!/usr/bin/env python3
"""
Split BirdNET_V2.4.onnx at the exact preprocessing handoff used for the NCNN
and QNN CNN-only experiments.

Input:
  raw-audio BirdNET V2.4 ONNX with shape [batch, 144000]

Output:
  CNN-only ONNX that expects the tensor produced immediately after the source
  graph's spectrogram affine normalization + transpose:
    [batch, 2, 96, 511]

This keeps the resulting model grounded in source_models/BirdNET_V2.4.onnx
while removing the unsupported DFT/Gather front-end from the ONNX graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


START_TENSOR = "model/ACT_0/Relu;model/BNORM_0/FusedBatchNormV3;model/CONV_0/Conv2D__265:0"
OUTPUT_TENSOR = "output"
START_NODE_INDEX = 47
INPUT_SHAPE = [1, 2, 96, 511]


def build_cnn_only_model(source_model: onnx.ModelProto) -> onnx.ModelProto:
    keep_nodes = list(source_model.graph.node)[START_NODE_INDEX:]
    produced = {output for node in keep_nodes for output in node.output}

    external_inputs = set()
    for node in keep_nodes:
        for name in node.input:
            if name:
                external_inputs.add(name)
    external_inputs.discard(START_TENSOR)
    external_inputs -= produced

    initializers = [
        initializer
        for initializer in source_model.graph.initializer
        if initializer.name in external_inputs
    ]

    output_value = None
    for candidate in source_model.graph.output:
        if candidate.name == OUTPUT_TENSOR:
            output_value = candidate
            break
    if output_value is None:
        raise ValueError(f"could not find output tensor {OUTPUT_TENSOR!r}")

    value_info = [
        value
        for value in source_model.graph.value_info
        if value.name == START_TENSOR or value.name in produced
    ]

    graph = helper.make_graph(
        keep_nodes,
        "BirdNET_V2.4.cnn_only",
        [helper.make_tensor_value_info(START_TENSOR, TensorProto.FLOAT, INPUT_SHAPE)],
        [output_value],
        initializer=initializers,
        value_info=value_info,
    )

    model = helper.make_model(graph, producer_name="codex-split")
    model.ir_version = source_model.ir_version

    # Preserve the source model's released opset imports exactly. Avoid keeping
    # helper.make_model()'s default latest opset import because current PNNX
    # bundles an ONNX Runtime that rejects unreleased opsets such as 26.
    del model.opset_import[:]
    model.opset_import.extend(source_model.opset_import)

    onnx.checker.check_model(model)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="source_models/BirdNET_V2.4.onnx",
        help="Path to the source BirdNET_V2.4.onnx model",
    )
    parser.add_argument(
        "--output",
        default="source_models/BirdNET_V2.4.cnn_only.onnx",
        help="Path to write the split CNN-only ONNX model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    model = onnx.load(str(input_path))
    split_model = build_cnn_only_model(model)
    onnx.save(split_model, str(output_path))

    print(f"wrote {output_path}")
    print(f"input tensor:  {START_TENSOR}")
    print(f"output tensor: {OUTPUT_TENSOR}")
    print(f"shape:         {INPUT_SHAPE}")


if __name__ == "__main__":
    main()
