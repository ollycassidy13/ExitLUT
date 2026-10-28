# ExitLUT-Predictor

This directory contains the LUT-based exit predictor used by ExitLUT. It is built on [NeuraLUT-Assemble](https://github.com/MartaAndronic/NeuraLUT), a framework for training neural networks that map directly onto FPGA LUTs.

## What it is for

ATHEENA decides whether a sample exits early by running a floating-point softmax pipeline on the early exit logits. ExitLUT replaces this with a small LUT network that takes the quantized logits and outputs a single bit: exit or continue.

NeuraLUT-Assemble is what we use to train that network. It:

1. Trains a small quantized network with sparse connections (each neuron sees only a few inputs).
2. Enumerates every neuron into a truth table, so each neuron becomes one logical LUT (L-LUT).
3. Generates Verilog from those truth tables, with no DSPs or BRAMs.

The result is a DSP-free exit decision that takes 12 cycles (10 for dataflow + 2 for the network) instead of 180 for the softmax path.

## How it is used

The predictor is trained to follow ATHEENA's thresholded softmax decision (`max softmax >= threshold`) from the quantized logits. Training data comes from running the B-LeNet ONNX model on the MNIST training set (see [`evaluate_onnx.py`](../evaluate_onnx.py)). The network is fixed: six sparse layers with fan-in 2, ten logit inputs and two outputs (continue, exit).

The code is in [`examples/b-lenet`](./examples/b-lenet):

| File | Purpose |
|------|---------|
| `train.py` | Trains the predictor (dense first, then sparse) and saves the best checkpoint |
| `neq2lut.py` | Converts the checkpoint to truth tables and generates Verilog |
| `models.py`, `dataset.py` | Model definition and the logit dataset loader |

## Usage

Install the package once (see the [Requirements](../README.md#requirements) in the parent README):

```bash
pip install .
```

Then, visit [`examples/b-lenet`](./examples/b-lenet) for instructions on training the predictor and generating Verilog. The generated Verilog is what gets plugged into ATHEENA in place of the softmax block.

## Citation

ExitLUT-Predictor is a derivative of NeuraLUT-Assemble, which is itself based on [LogicNets](https://github.com/Xilinx/logicnets) (Apache License 2.0). Please cite:

```bibtex
@inproceedings{cassidy2026exitlut,
    author    = {Cassidy, Oliver and Andronic, Marta and Constantinides, George A.},
    title     = {{ExitLUT: Boolean Early Exits for Efficient FPGA-Based Neural Network Inference}},
    booktitle = {2026 International Conference on Field-Programmable Technology (ICFPT)},
    publisher = {IEEE},
    year      = {2026}
}

@inproceedings{andronic2025neuralut-assemble,
    author    = "Andronic, Marta and Constantinides, George A.",
    title     = "{NeuraLUT-Assemble: Hardware-Aware Assembling of Sub-Neural Networks for Efficient LUT Inference}",
    booktitle = "{2025 IEEE 33rd Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)}",
    pages     = "208-216",
    publisher = "IEEE",
    year      = 2025,
    note      = "doi: 10.1109/FCCM62733.2025.00077"
}
```
