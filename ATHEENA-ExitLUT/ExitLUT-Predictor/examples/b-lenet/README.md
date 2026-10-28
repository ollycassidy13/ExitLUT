## ExitLUT-Predictor for B-LeNet on MNIST

This folder provides the code and resources to reproduce our ExitLUT results in ATHEENA for B-LeNet on MNIST.

### Usage

```bash
# 1. Train the predictor on logits saved from the ATHEENA model
python train.py --arch b-lenet --log_dir exitlut_0.996 \
    --dataset_file /path/to/logits.npz --cuda --device 0

# 2. Convert to LUTs and generate Verilog
python neq2lut.py --arch b-lenet \
    --checkpoint test_exitlut_0.996/best_accuracy.pth \
    --imask test_exitlut_0.996/imask.pth \
    --dataset-file /path/to/logits.npz \
    --log-dir /path/to/verilog-out \
    --add-max-output --add-input-register --add-output-register \
    --cuda --device 0
```

The confidence threshold (`0.996` above) is set by `confidence_threshold` in the config at the top of `train.py`.

### Pretrained predictor

The predictor used for the paper's B-LeNet results is included in [`test_exitlut_0.996/`](./test_exitlut_0.996/):

- `best_accuracy.pth`: the trained, sparse ExitLUT predictor (highest validation exit-decision accuracy).
- `imask.pth`: its learned sparse connectivity masks.

To map it to LUTs and Verilog without retraining, run step 2 above with these files as `--checkpoint` and `--imask`. Step 2 needs the early exit logits (`--dataset-file`) to validate the converted model; `evaluate_onnx.py --save-logits` generates them from the ATHEENA ONNX model (see [`run.sh`](../../../run.sh)).

[`run_demo.sh`](../../../run_demo.sh) runs the flow using the pre-trained checkpoint. It generates the logits if they are missing, runs step 2, then evaluates the resulting LUT predictor against the ATHEENA softmax decision on the MNIST test set:

```bash
bash run_demo.sh [DATA_DIR]
```

> Note: the final stage of `neq2lut.py` is out-of-context Vivado synthesis, which raises an error if Vivado/oh-my-xilinx is not set up. This is expected; the Verilog is written before it and evaluation still runs.
>
> Running `train.py` with `--log_dir exitlut_0.996` writes to the same `test_exitlut_0.996/` directory and will overwrite the pretrained files. Use a different `--log_dir` to keep them.

## Citation
Should you find this work valuable, we kindly request that you reference our papers as below:
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