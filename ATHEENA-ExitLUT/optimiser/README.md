# ATHEENA-ExitLUT Optimiser

This repo contains code for optimising the mapping a Early Exit Convolutional Neural Network (CNN) model to an FPGA. Hardware-specific transforms are applied to the model, producing a hardware description that can be used by the hls backend. The generated architecture is streaming-based, and optimised for a specified hardware platform.

## Use in ExitLUT

The DSE is used to show the system-level benefit of the ExitLUT predictor. It sizes each stage of an early exit network for a platform and a profiled continuation probability `p`, so the resources saved by replacing the softmax exit decision are reinvested in the datapath (higher folding and parallelism for the same device budget). These designs were then mapped to hardware following the [ATHEENA](https://github.com/biggsbenjamin/ATHEENA) and [fpgaConvNet](https://github.com/AlexMontgomerie/fpgaconvnet-hls) HLS flows.

### Changes from ATHEENA

This copy differs from upstream ATHEENA in two ways. The exit-decision block is new, and the resource models are recalibrated against Vivado synthesis results. The algorithms (transforms, simulated annealing, stage merging) are otherwise unchanged. Every modified file carries a notice at the top, as required by the GPL.

**Exit decision (new).** Models the ExitLUT predictor alongside ATHEENA's softmax comparator:

- [`models/modules/NeuralUT.py`](./fpgaconvnet_optimiser/models/modules/NeuralUT.py): a hardware module for the predictor. It uses the measured resources of the generated LUT network (1241 LUTs, 460 FFs, 0 DSPs, 0 BRAMs) and a pipeline depth of 12 cycles. `functional_model` is the softmax-threshold decision the predictor is trained to reproduce.
- [`models/layers/NeuralUTCmpLayer.py`](./fpgaconvnet_optimiser/models/layers/NeuralUTCmpLayer.py): a layer that replaces `SoftMaxCmpLayer`, with the same interface. Its hardware is fixed, so it only supports `coarse_in = 1`.
- [`tools/parser.py`](./fpgaconvnet_optimiser/tools/parser.py) and [`tools/dev_script.py`](./fpgaconvnet_optimiser/tools/dev_script.py): the parser picks `NeuralUTCmpLayer` or `SoftMaxCmpLayer` for the exit decision. Choose it with `--exit_decision {softmax,exitlut}` or the `FPGACONVNET_EXIT_DECISION` environment variable. The default is `softmax`, which reproduces ATHEENA.

For example, to run the DSE with the ExitLUT predictor, add the flag to the command from [Optimiser Framework](#optimiser-framework):

```Shell
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

**Resource-model and search changes.**

- `BufferLayer.py`: optional coarse folding for buffers (`BufferLayer.allow_coarse`, set in `dev_script.py`).
- `InnerProductLayer.py`, `Buffer.py`, `Conv.py`, `Squeeze.py`: recalibrated LUT/FF utilisation models; `ReLULayer.py` and `SqueezeLayer.py`: explicit `resource()` methods; `SoftMaxCmpLayer.py`: measured per-lane resources of the softmax decision.
- `fpgaconvnet_optimiser/coefficients/*.npy`: refitted coefficients for `buffer_lut`, `conv_ff`, `conv_lut`, `relu_lut`, `squeeze_ff` and `squeeze_lut`.
- `dev_script.py` and `examples/optimiser_example.yml`: different resource limits and annealing settings, and `apply_transform` enabled.

The resource figures are fixed numbers for the B-LeNet predictor. If you train a different predictor, update the counts in `NeuralUT.py` from its synthesis report.


From ATHEENA (we recommend following the setup instructions in [our example](../README.md) to reproduce our results):
> ## Setup
>
> The optimiser (and HLS generation) has been verified using the following software:
>
> - `python=3.7`
> - `miniconda=4.10.1`
>
> To install this package, run from this directory the following:
>
> ```
> sudo apt install protobuf-compiler libprotoc-dev
> cd ./optimiser/
> conda env create -f atheena_opt_hls_p37.yml
> conda activate atheena_opt_hls_p37
> python atheena_setup.py install
> ```
>
> ## Optimiser Framework
>
> The main tool is the optimisation script which generates an optimised hardware topology for a given model and platform. There are several components needed for this: a model of the hardware, transforms that map the model to the hardware and an optimisation scheme that chooses the best mapping.
>
> ```Shell
> python -m fpgaconvnet_optimiser.tools.dev_script \
>     --expr opt_brn \
>     --save_name branchy_lenet \
>     -o outputs/branchy_lenet \
>     --model_path examples/models/atheena/branchy_lenet_20220902.onnx \
>     --platform_path examples/platforms/zc706.json \
>     --optimiser_path examples/optimiser_example.yml \
>     -bs 1024
> ```
>
> This will generate the following a series of json files describing the stages of the early-exit network under different resource utilisation contraints.
>
> - `(output_path)/no-buff-shift/post_optim-rsc(P)p/report_(save_name)-ee(stage)-rsc(P)p-iter(I).json`: A report file containing estimations of usage and performance at `(P)` percent of the boards available resources, for `(stage)` of the network, for the `(I)th` iteration of the optimiser..
> - `(output_path)/no-buff-shift/post_optim-rsc[P]p/(save_name)-ee(stage)-rsc(P)p-iter(I).json`: A hardware specification file containing estimations of usage and performance at `[P]` percent of the boards available resources, for `(stage)` of the network, for the `(I)th` iteration of the optimiser.
>
> ```Shell
> python -m fpgaconvnet_optimiser.tools.dev_script \
>     --expr gen_graph \
>     --save_name branchy_lenet_graph \
>     -o outputs/branchy_lenet/results/ \
>     -i outputs/branchy_lenet/no-buff-shift/ \
>     --profiled_probability 0.75
> ```
>
> Running the `gen_graph` experiment will generate a pareto front for the optimised combination of the networks as well as a list of the reports for each of the network stages so that the user can choose pairs of network stages on the pareto front.
>
> - `(output_path)/combined_rpt_p(PR).csv`, where `PR` is `--profiled_probability` as a percentage (the fraction of samples that reach the late stage, i.e. `1 - exit probability`): lists the pareto front combinations of network stages and their expected throughput and resource usage.
>
> > **Note**: for a graph containing baseline results, specify the path using `-bi`.
>
> ### Merging optimised stages
>
> Run the following command to perform a stage merge for all the results in the combined report.
>
> ```Shell
> python -m fpgaconvnet_optimiser.tools.ee_stage_merger \
>     -cp outputs/branchy_lenet/results/combined_rpt_p75.csv \
>     -jp outputs/branchy_lenet/no-buff-shift/ \
>     -on branchy_lenet_merged \
>     -op outputs/branchy_lenet/merged/
> ```
>
> - `(output_path)/branchy_lenet_merged(resources and throughput).json`: the final hardware specification ready to be implemented using the HLS backend.
>
> Copy this `.json` file into a folder in `hls/test/partitions/(example)/`.
>
> For example:
> ```
> mkdir -p ../hls/test/partitions/branchy_lenet_eg
>
> cp outputs/branchy_lenet/merged/branchy_lenet_merged_rsc80_thru95000.json ../hls/test/partitions/branchy_lenet_eg/
> ```
>
> ## Additional Information
>
> ### Modelling
>
> In order to do the CNN to hardware mapping, a model of the hardware is needed. There are four levels of abstraction for the final hardware: modules, layers, partitions and network. At each level of abstraction, there is an associated performance and resource estimate so that the constraints for the optimiser can be obtained.
>
> - __Module:__ These are the basic building blocks of the accelerator. The modules are the following:
>   - Accum
>   - Conv
>   - Glue
>   - SlidingWindow
>   - Fork
>   - Pool
>   - Squeeze
> - __EE Modules:__
>   - Exponential
>   - Buffer
>   - Compare
>   - ExitMerge
>   - ReduceMax
>   - SoftMaxSum
> - __Layer:__ Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model. The following layers are supported:
>   - Convolution
>   - Inner Product
>   - Pooling
>   - ReLU
> - __EE Layers:__
>   - Buffer
>   - SoftMaxCmp
>   - Split
>   - ExitMerge
>
> - __Partition:__ Partitions make up a sub-graph of the CNN model network. They are comprised of layers. A single partition fits on an FPGA at a time, and partitions are changed by reconfiguring the FPGA.
> - __Network:__ This is the entire CNN model described through hardware. A network contains partitions and information on how to execute them.
>
> ### Transforms
>
> In order to find the optimal mapping, transforms are used to manipulate the performance/resource trade-off of the hardware. The main transforms implemented currently are:
>
> | Transform | Level | Description |
> |-----------|:-----:|------------|
> | Fine | Module | Defines the parallelism for the kernel x kernel dot product of the `conv` module. |
> | Coarse | Layer | Input and output channel dimension parallelism of Layers. For a convolution layer, this is how filters are run in parallel. |
>
> ### Optimisation Schemes
>
> Finally, optimisations schemes are used to explore the transform design space and find an optimal mapping for a given hardware platform. The optimisation schemes implemented are the following:
>
> - __Simulated Annealing:__ Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function.
>
> ---

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
