#  This file is part of NeuraLUT.
#
#  NeuraLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralut.nn import (
    generate_truth_tables,
    lut_inference,
    module_list_to_verilog_module,
    QuantLayer,
    MaxOutputLayer,
)

from train import configs, model_config, dataset_config, test
from dataset import SoftmaxApproxDataset
from models import ExitLUTNeqModel, ExitLUTLutModel
from neuralut.synthesis import synthesize_and_get_resource_counts

other_options = {
    "cuda": None,
    "device": 1,
    "log_dir": None,
    "checkpoint": None,
    "imask": None,
    "add_registers": False,
    "add_input_register": False,
    "add_output_register": False,
    "add_max_output": False,
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Synthesize convert a PyTorch trained model into verilog"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="b-lenet",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch size for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=None,
        metavar="",
        help="Input feature length for synthetic softmax task (default: %(default)s)",
    )
    parser.add_argument(
        "--output_length",
        type=int,
        default=None,
        metavar="",
        help="Output class length for binary decision task; must be 2 (default: %(default)s)",
    )
    parser.add_argument(
        "--input-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--support_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the support layer (default: %(default)s)",
    )
    parser.add_argument(
        "--input-fanin",
        type=int,
        default=None,
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-fanin",
        type=int,
        default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--support_layers",
        nargs="+",
        type=int,
        default=None,
        metavar="",
        help="A boolean list of support layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-fanin",
        type=int,
        default=None,
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--width_n",
        type=int,
        default=None,
        metavar="",
        help="Width of sub-network(N) (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="Early-exit logits .npz saved by evaluate.py --save-logits on the training split",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        metavar="",
        help="Synthetic training-set size used for deterministic split generation (default: %(default)s)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=None,
        metavar="",
        help="Synthetic validation-set size used for deterministic split generation (default: %(default)s)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        metavar="",
        help="Synthetic training-set size (default: %(default)s)",
    )
    parser.add_argument(
        "--valid_size",
        type=int,
        default=None,
        metavar="",
        help="Synthetic validation-set size (default: %(default)s)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        metavar="",
        help="Synthetic test-set size (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        metavar="",
        help="Seed used to generate synthetic dataset splits (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        metavar="",
        help="Decision threshold on max(softmax(logits)) in (0, 1) (default: %(default)s)",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=None,
        metavar="",
        help="Scale for synthetic input logits (default: %(default)s)",
    )
    parser.add_argument(
        "--clock-period",
        type=float,
        default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="config/yaml_IP_OP_config.yml",
        help="The file to use to configure the input dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset to use for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device_id for GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint file which contains the model weights",
    )
    parser.add_argument(
        "--imask",
        type=str,
        required=True,
        help="The checkpoint file which contains the model weights",
    )
    parser.add_argument(
        "--add-registers",
        action="store_true",
        default=False,
        help="Add pipeline registers before layers 1..N-1 in generated verilog (default: %(default)s)",
    )
    parser.add_argument(
        "--add-input-register",
        action="store_true",
        default=False,
        help="Add a pipeline register before the first layer (on M0) in generated verilog (default: %(default)s)",
    )
    parser.add_argument(
        "--add-output-register",
        action="store_true",
        default=False,
        help="Add a pipeline register after the final layer (on M{N}) in generated verilog (default: %(default)s)",
    )
    parser.add_argument(
        "--support_fanin",  
        type=int,
        default=None,
        metavar="",
        help="Fanin to use for the support layers (default: %(default)s)",
    )
    parser.add_argument(
        "--add-max-output",
        action="store_true",
        default=False,
        help="Append a max-output LUT layer that reduces 2 output neurons to a 1-bit exit decision (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Train on a GPU (default: %(default)s)",
    )
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]
    config = {}
    for k in options.keys():
        config[k] = (
            options[k] if options[k] is not None else defaults[k]
        )

    if config["output_length"] != 2:
        raise ValueError("output_length must be exactly 2 for binary decision output")

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]
    model_cfg["cuda"] = options_cfg["cuda"]
    model_cfg["logit_scale"] = dataset_cfg["logit_scale"]
    
    if options_cfg["cuda"]:
        torch.cuda.set_device(options_cfg["device"])

    dataset = {}
    dataset["train"] = SoftmaxApproxDataset(
        dataset_cfg["dataset_file"],
        dataset_cfg["dataset_config"],
        split="train",
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
        train_size=dataset_cfg["train_size"],
        valid_size=dataset_cfg["valid_size"],
        test_size=dataset_cfg["test_size"],
        dataset_seed=dataset_cfg["dataset_seed"],
        confidence_threshold=dataset_cfg["confidence_threshold"],
        logit_scale=dataset_cfg["logit_scale"],
    )
    dataset["valid"] = SoftmaxApproxDataset(
        dataset_cfg["dataset_file"],
        dataset_cfg["dataset_config"],
        split="valid",
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
        train_size=dataset_cfg["train_size"],
        valid_size=dataset_cfg["valid_size"],
        test_size=dataset_cfg["test_size"],
        dataset_seed=dataset_cfg["dataset_seed"],
        confidence_threshold=dataset_cfg["confidence_threshold"],
        logit_scale=dataset_cfg["logit_scale"],
    )
    dataset["test"] = SoftmaxApproxDataset(
        dataset_cfg["dataset_file"],
        dataset_cfg["dataset_config"],
        split="test",
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
        train_size=dataset_cfg["train_size"],
        valid_size=dataset_cfg["valid_size"],
        test_size=dataset_cfg["test_size"],
        dataset_seed=dataset_cfg["dataset_seed"],
        confidence_threshold=dataset_cfg["confidence_threshold"],
        logit_scale=dataset_cfg["logit_scale"],
    )
    train_loader = DataLoader(
        dataset["train"], batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        dataset["valid"], batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=config["batch_size"], shuffle=False
    )

    model_cfg["output_length"] = 2
    model_cfg["imask"] = torch.load(
        options_cfg["imask"],
        map_location="cuda:{}".format(options_cfg["device"]) if options_cfg["cuda"] else "cpu",
    )
    model_cfg['dense_forward'] = False

    model = ExitLUTNeqModel(model_cfg)
    if options_cfg["cuda"]:
        model.cuda()

    checkpoint = torch.load(options_cfg["checkpoint"], map_location="cuda:{}".format(options_cfg["device"]) if options_cfg["cuda"] else "cpu")
    model.load_state_dict(checkpoint["model_dict"])

    print("Running inference on baseline model...")
    baseline_accuracy = test(model, val_loader, cuda=options_cfg["cuda"])
    print("Baseline accuracy: %f" % (baseline_accuracy))

    lut_model = ExitLUTLutModel(model_cfg)
    if options_cfg["cuda"]:
        lut_model.cuda()
    lut_model.load_state_dict(checkpoint['model_dict'])

    print("Converting to NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)

    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    print("Testing LUT-Based Model")
    lut_accuracy = test(lut_model, val_loader, cuda=options_cfg["cuda"])
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    modelSave = {
        "model_dict": lut_model.state_dict(),
        "test_accuracy": lut_accuracy,
        "model_cfg": {k: v for k, v in model_cfg.items() if k != "imask"},
    }
    torch.save(modelSave, options_cfg["log_dir"] + "/lut_based_model.pth")
    torch.save(model_cfg["imask"].cpu(), options_cfg["log_dir"] + "/imask.pth")

    first_layer = lut_model.module_list[0]
    _, q_out_bits = first_layer.input_quant.get_scale_factor_bits(options_cfg["cuda"])
    quant_layer = QuantLayer(
        num_features=model_cfg["input_length"],
        output_bitwidth=int(q_out_bits),
    )
    quant_layer.calculate_truth_tables(
        first_layer.input_quant,
        cuda=options_cfg["cuda"],
        logit_scale=dataset_cfg["logit_scale"],
    )
    augmented_module_list = nn.ModuleList([quant_layer] + list(lut_model.module_list))

    if options_cfg["add_max_output"]:
        last_layer = lut_model.module_list[-1]
        _, out_bw = last_layer.output_quant.get_scale_factor_bits(options_cfg["cuda"])
        max_layer = MaxOutputLayer(input_bitwidth=int(out_bw))
        max_layer.calculate_truth_tables()
        augmented_module_list = nn.ModuleList(list(augmented_module_list) + [max_layer])

    print("Generating verilog in %s..." % (options_cfg["log_dir"]))
    module_list_to_verilog_module(
        augmented_module_list,
        "neuralut",
        options_cfg["log_dir"],
        add_registers=options_cfg["add_registers"],
        add_input_register=options_cfg["add_input_register"],
        add_output_register=options_cfg["add_output_register"],
    )
    print("Top level entity stored at: %s/neuralut.v ..." % (options_cfg["log_dir"]))

    io_filename = None
    print("Running inference simulation of Verilog-based model...")
    lut_model.verilog_inference(options_cfg["log_dir"], "neuralut.v", logfile=io_filename, add_registers=options_cfg["add_registers"], add_input_register=options_cfg["add_input_register"], add_output_register=options_cfg["add_output_register"], with_quant_layer=True, with_max_output=options_cfg["add_max_output"])
    print("Testing Verilog-Based Model")
    verilog_accuracy = test(lut_model, val_loader, cuda=options_cfg["cuda"])
    print("Verilog-Based Model accuracy: %f" % (verilog_accuracy))

    print("Running out-of-context synthesis")
    ret = synthesize_and_get_resource_counts(options_cfg["log_dir"], "neuralut", fpga_part='xc7z045ffg900-2', clk_period_ns='8.0', post_synthesis=1)
    print("Max f: " + str(ret))