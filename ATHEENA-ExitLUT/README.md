# ATHEENA-ExitLUT

This directory contains the scripts for running ATHEENA DSE and ExitLUT training. The DSE scripts are based on the ATHEENA toolflow, which is a tool for automating the design of hardware early exit networks. The training scripts are based on the NeuraLUT-Assemble toolflow, which is a tool for training LUT-based neural networks.

<img src="../img/exitlut.png" alt="ExitLUT" width="800"/>

## B-LeNet Flow

With the [environment](#requirements) activated:

```bash
bash run.sh [DATA_DIR]
```

`DATA_DIR` is where MNIST is stored or downloaded (default: `./data` in this directory - it will download if not found). Training needs a GPU. Set `WANDB_MODE=disabled` first to skip Weights & Biases logging.

This will run the ATHEENA ONNX model and ExitLUT training for the B-LeNet model.

First the flow runs the B-LeNet ONNX model on the MNIST train set to generate the softmax training data. We then train the ExitLUT predictor on the softmax training data. Next, we map the trained model to LUTs and generate the Verilog code. Finally, we run the ExitLUT predictor on the B-LeNet ONNX model to measure the performance of the predictor on the test set.

> To test the flow without training use `bash run_demo.sh [DATA_DIR]`. This will run the B-LeNet ONNX model and pre-trained ExitLUT predictor on the MNIST test set.

## ATHEENA DSE

The ATHEENA design-space exploration (DSE) maps an early exit CNN onto an FPGA. Given an ONNX model, a target platform and the profiled continuation probability `p`, it uses simulated annealing to choose folding factors for each stage under the platform's resource budget, producing a streaming hardware description for a HLS backend. The DSE can be found in the [optimiser](./optimiser) directory. We include the DSE scripts for running with the ExitLUT predictor, meaning you can generate design based on the use of the ExitLUT predictor.

#### Running the ATHEENA DSE

Run all commands from the [optimiser](./optimiser) directory with the [environment](#requirements) activated:

```bash
cd optimiser
```

**1. Optimise each stage.** Simulated annealing is run on every stage of the early exit network under a range of resource budgets. Add `--exit_decision exitlut` to model the ExitLUT predictor as the exit decision; omit it (or use `softmax`) to reproduce ATHEENA.

```bash
python -m fpgaconvnet_optimiser.tools.dev_script \
    --expr opt_brn \
    --exit_decision exitlut \
    --save_name branchy_lenet_exitlut \
    -o outputs/branchy_lenet_exitlut \
    --model_path examples/models/atheena/branchy_lenet_20220902.onnx \
    --platform_path examples/platforms/zc706.json \
    --optimiser_path examples/optimiser_example.yml \
    -bs 1024
```

The result is a set of `post_optim-rsc(P)p/` folders in the `no-buff-shift/` subfolder of the output directory, each holding a hardware description (`.json`) and a performance/resource report for every stage at `P`% of the platform's resources. Change `--platform_path` (see [`optimiser/examples/platforms`](./optimiser/examples/platforms)) to target another board, and edit the [optimiser config](./optimiser/examples/optimiser_example.yml) to change the annealing settings.

**2. Combine stages into a Pareto front.** Give the profiled continuation probability `p` (the fraction of samples that do *not* exit early and go on to the late stage, i.e. `1 - exit probability`, measured on the training set; e.g. `0.25` means 75% exit early) with `--profiled_probability`:

```bash
python -m fpgaconvnet_optimiser.tools.dev_script \
    --expr gen_graph \
    --save_name branchy_lenet_exitlut_graph \
    -o outputs/branchy_lenet_exitlut/results/no-buff-shift \
    -i outputs/branchy_lenet_exitlut/no-buff-shift/ \
    --profiled_probability 0.25
```

This writes `combined_rpt_p25.csv` to the results folder (the number is `p` as a percentage). It lists the Pareto-optimal pairings of first- and second-stage designs with their expected throughput and resource usage. To compare against a baseline (no early exit) run, pass its output folder with `-bi`.

**3. Merge the chosen stages.** The stage merger builds a single full-network hardware description for every pairing in the combined report:

```bash
python -m fpgaconvnet_optimiser.tools.ee_stage_merger \
    -cp outputs/branchy_lenet_exitlut/results/no-buff-shift/combined_rpt_p25.csv \
    -jp outputs/branchy_lenet_exitlut/no-buff-shift/ \
    -on branchy_lenet_exitlut_merged \
    -op outputs/branchy_lenet_exitlut/merged/
```

Each output file is named `branchy_lenet_exitlut_merged_rsc(R)_thru(T).json`, giving the resource percentage and expected throughput of that design. Pick the one that fits your target and pass it to the HLS backend (see [Full-Accelerator Hardware Mapping](#full-accelerator-hardware-mapping)).

To use the ATHEENA exit decision, repeat steps 1-3 without `--exit_decision exitlut`.

> **Note**: only the B-LeNet ONNX models (ATHEENA and baseline) are included in [`optimiser/examples/models`](./optimiser/examples/models). The Triple-Wins and B-AlexNet models used in the paper can be found in the [ATHEENA](https://github.com/biggsbenjamin/ATHEENA) repository.

## Requirements

A single conda environment covers the ExitLUT predictor training, the ONNX evaluation scripts and the ATHEENA optimiser.

**1. Create the environment**

```bash
conda create -n exitlut python=3.8.19
conda activate exitlut
pip install torch==2.4.1 torchvision==0.19.1 brevitas==0.10.2 PyVerilator==0.7.0 wandb==0.24.2 \
    numpy==1.24.3 scipy==1.10.1 pandas==2.0.3 matplotlib==3.7.5 networkx==3.1 scikit-learn==1.3.2 \
    sympy==1.13.3 PyYAML==6.0.2 protobuf==3.20.3 pyparsing==3.1.4 onnx==1.17.0 onnxruntime==1.19.2
pip install onnxoptimizer==0.3.0 pydot graphviz ddt coverage untangle fpbinary
```

Training uses a CUDA GPU (we used CUDA 12.1). Log in to Weights & Biases with `wandb login`, or set `WANDB_MODE=disabled` to skip logging.

**2. Install the ExitLUT predictor**

```bash
cd ExitLUT-Predictor && pip install . && cd ..
```

> **Optional: Vivado tooling.** Only needed to synthesise the generated Verilog or run HLS; neither is required to train the predictor or run the DSE.
>
> Install the Vivado Design Suite. We used Vivado 2020.1 for out-of-context synthesis of the generated Verilog (following NeuraLUT) and Vivado HLS 2019.1 for HLS synthesis of the full accelerator. Then install oh-my-xilinx, which is used to call Vivado from the LUT flow:
>
> ```bash
> git clone https://github.com/ollycassidy13/oh-my-xilinx.git /path/to/local/dir
> export OHMYXILINX=/path/to/local/dir
> ```

## Full-Accelerator Hardware Mapping

Please refer to the [ATHEENA](https://github.com/biggsbenjamin/ATHEENA) and [fpgaConvNet](https://github.com/AlexMontgomerie/fpgaconvnet-hls) toolflows for details on how to map the trained models to FPGA hardware. The ExitLUT predictor will generate Verilog which can be combined with the existing HLS flows to produce a complete FPGA implementation of the early exit network. We used Vivado HLS 2019.1 to synthesise the full accelerator designs in the paper.

## Citation

If ExitLUT contributes to your research, please cite:

```bibtex
@inproceedings{cassidy2026exitlut,
    author    = {Cassidy, Oliver and Andronic, Marta and Constantinides, George A.},
    title     = {{ExitLUT: Boolean Early Exits for Efficient FPGA-Based Neural Network Inference}},
    booktitle = {2026 International Conference on Field-Programmable Technology (ICFPT)},
    publisher = {IEEE},
    year      = {2026}
}
```

ExitLUT builds on NeuraLUT-Assemble and ATHEENA; please also cite:

```bibtex
@inproceedings{andronic2025neuralut-assemble,
    author    = "Andronic, Marta and Constantinides, George A.",
    title     = "{NeuraLUT-Assemble: Hardware-Aware Assembling of Sub-Neural Networks for Efficient LUT Inference}",
    booktitle = "{2025 IEEE 33rd Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)}",
    pages     = "208-216",
    publisher = "IEEE",
    year      = 2025,
    note      = "doi: 10.1109/FCCM62733.2025.00077"
}

@inproceedings{bbiggs_ATHEENA_2023,
    title = {{ATHEENA: A Toolflow for Hardware Early-Exit Network Automation}},
    booktitle = {2023 IEEE 31st Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
    author = {Benjamin Biggs and Christos-Savvas Bouganis and George A. Constantinides},
    year = {2023},
}
```
