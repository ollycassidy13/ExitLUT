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
from functools import reduce
import random

import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = callable(getattr(wandb, "init", None))
except ImportError:
    WANDB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SoftmaxApproxDataset
from models import ExitLUTNeqModel

configs = {
    "b-lenet": {
        "hidden_layers": [64, 32, 16, 8, 4],
        "support_layers": [0, 1, 1, 1, 1, 1],
        "input_length": 10,
        "output_length": 2,
        "input_bitwidth": 4,
        "hidden_bitwidth": 3,
        "output_bitwidth": 3,
        "support_bitwidth": 3,
        "input_fanin": 2,
        "hidden_fanin": 2,
        "output_fanin": 2,
        "support_fanin": 2,
        "width_n": 64,
        "weight_decay": 5e-4,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate": 0.002,
        "seed": 8766,
        "checkpoint": None,
        "dense_epochs": 25,
        "wd": 0.0001,
        "wd1": 0.07,
        "train_size": 55000,
        "valid_size": 5000,
        "test_size": 0,
        "dataset_seed": 1234,
        "confidence_threshold": 0.996,
        "logit_scale": 128.0,
    },
}

model_config = {
    "hidden_layers": None,
    "support_layers": None,
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "support_bitwidth": None,
    "input_fanin": None,
    "hidden_fanin": None,
    "output_fanin": None,
    "support_fanin": None,
    "width_n": None,
    "input_length": None,
    "output_length": None,
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
    "dense_epochs": None,
    "wd": None,
    "wd1": None,
}

dataset_config = {
    "dataset_file": None,
    "dataset_config": None,
    "train_size": None,
    "valid_size": None,
    "test_size": None,
    "dataset_seed": None,
    "confidence_threshold": None,
    "logit_scale": None,
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "device": 1,
}


def train(model, datasets, train_cfg, options):
    train_loader = DataLoader(
        datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        datasets["valid"], batch_size=train_cfg["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=train_cfg["batch_size"], shuffle=False
    )

    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = [
        "bn",
        "bias",
        "learned_value",
    ]
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(
                lambda a, b: a or b, map(lambda x: x in pname, decay_exclusions)
            ):
                no_decay_params.append(params)
            else:
                decay_params.append(params)
    params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        betas=(0.5, 0.999),
        weight_decay=weight_decay,
    )

    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps * 100, T_mult=1
    )

    criterion = nn.CrossEntropyLoss()

    if options["cuda"]:
        model.cuda()

    maxAcc = 0.0
    if model_cfg['dense_forward'] == True:
        num_epochs = train_cfg["dense_epochs"]
    else:
        num_epochs = train_cfg["epochs"]
    for epoch in range(0, num_epochs):
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            target_labels = target.long().view(-1)
            loss = criterion(output, target_labels)
            if model_cfg['dense_forward'] == True:
                loss = loss + regularizer(model)
            pred = torch.argmax(output.detach(), dim=1)
            curCorrect = pred.eq(target_labels).long().sum()
            curAcc = 100.0 * curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach() * len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_size = len(train_loader.dataset)
        if train_size > 0:
            accLoss /= train_size
            accuracy = 100.0 * correct / train_size
        else:
            accLoss = 0.0
            accuracy = 0.0
        val_accuracy = test(model, val_loader, options["cuda"])
        test_accuracy = test(model, test_loader, options["cuda"])
        modelSave = {
            "model_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "epoch": epoch,
            "logit_scale": options.get("logit_scale"),
            "model_cfg": {k: v for k, v in model_cfg.items() if k != "imask"},
        }
        if model_cfg['dense_forward'] == False:
            torch.save(modelSave, "test_" + options["log_dir"] + "/checkpoint.pth")
            if maxAcc < val_accuracy:
                torch.save(modelSave, "test_" + options["log_dir"] + "/best_accuracy.pth")
                maxAcc = val_accuracy
            if WANDB_AVAILABLE:
                wandb.log(
                    {
                        "Train Acc (%)": accuracy.detach().cpu().numpy(),
                        "Train Loss(%)": accLoss.detach().cpu().numpy(),
                        "Test Acc (%)": test_accuracy,
                        "Valid Acc(%)": val_accuracy,
                    }
                )
        else:
            print(f"Epoch: {epoch}/{num_epochs}\tValid Acc: {val_accuracy:.2f}\tTest Acc: {test_accuracy:.2f}")
            if maxAcc < val_accuracy:
                torch.save(modelSave, "test_" + options["log_dir"] + "/best_dense.pth")
                maxAcc = val_accuracy

def GetInputMask(model, is_cuda):
    imask = torch.tensor([])
    if is_cuda:
        imask = imask.cuda()
    max_fan_in = max(model_cfg["hidden_fanin"], model_cfg["input_fanin"], model_cfg["output_fanin"], model_cfg["support_fanin"])
    for i in range(0,len(model_cfg["hidden_layers"])+1):
        module = model.module_list[i]
        if hasattr(module, 'dense_forward') and module.dense_forward:
            weight = module.res2_dense.weight
            fan_in = module.fan_in
            for row in range(weight.size(0)):
                weight_row = weight[row, :]
                idx = torch.topk(torch.abs(weight_row), fan_in).indices
                if is_cuda:
                    imask_row = torch.cat((idx, torch.zeros(max_fan_in-len(idx)).cuda())).unsqueeze(0)
                else:
                    imask_row = torch.cat((idx, torch.zeros(max_fan_in-len(idx)))).unsqueeze(0)
                imask = torch.cat((imask, imask_row), dim = 0)
        
    return imask.long()

def regularizer(model):
    R = 0
    for i in range(0,len(model_cfg["hidden_layers"])+1):
        module = model.module_list[i]
        if hasattr(module, 'dense_forward') and module.dense_forward:
            weight1 = module.res2_dense.weight
            weight2 = module.fc1_dense.weight
            res2_dense_sum = weight1.abs().sum(dim=-1)
            fc1_dense_sum = weight2.abs().sum(dim=-1)
            l1 = torch.cat((res2_dense_sum, fc1_dense_sum), dim=0)
            
            R += train_cfg['wd'] * (torch.pow(2, train_cfg['wd1'] * l1) - 1).sum()
    return R


def test(model, dataset_loader, cuda):
    model.eval()
    correct = 0
    accLoss = 0.0
    dataset_size = len(dataset_loader.dataset)
    if dataset_size == 0:
        return 0.0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = torch.argmax(output.detach(), dim=1)
        target_label = target.long().view(-1)
        curCorrect = pred.eq(target_label).long().sum()
        curAcc = 100.0 * curCorrect / len(data)
        correct += curCorrect
    accuracy = 100 * float(correct) / dataset_size
    return accuracy


if __name__ == "__main__":
    parser = ArgumentParser(description="NeuraLUT Example")
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="b-lenet",
        metavar="",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        metavar="",
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="",
        help="Batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        metavar="",
        help="Initial learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Train on a GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="",
        help="Seed to use for RNG (default: %(default)s)",
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
        "--input_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output_bitwidth",
        type=int,
        default=None,
        metavar="",
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
        "--input_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_fanin",
        type=int,
        default=None,
        metavar="",
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
        "--output_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--width_n",
        type=int,
        default=None,
        metavar="",
        help="Width of sub-network(N) (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=int,
        default=None,
        metavar="",
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        metavar="",
        help="Early-exit logits .npz saved by evaluate.py --save-logits on the training split",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="config/yaml_IP_OP_config.yml",
        metavar="",
        help="The file to use to configure the input dataset (default: %(default)s)",
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
        "--dataset_seed",
        type=int,
        default=None,
        metavar="",
        help="Seed used to generate synthetic dataset splits (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        metavar="",
        help="Decision threshold on max(softmax(logits)) in (0, 1) (default: %(default)s)",
    )
    parser.add_argument(
        "--logit_scale",
        type=float,
        default=None,
        metavar="",
        help="Scale for synthetic input logits (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="",
        help="Retrain the model from a previous checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=1, 
        metavar="", 
        help="Device_id for GPU",
    )
    parser.add_argument(
        "--dense_epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train the dense model (default: %(default)s)",
    )
    parser.add_argument(
        "--support_fanin",  
        type=int,
        default=None,
        metavar="",
        help="Fanin to use for the support layers (default: %(default)s)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=None,
        metavar="",
        help="Weight decay for group regularizer (default: %(default)s)",
    )
    parser.add_argument(
        "--wd1",
        type=float,
        default=None,
        metavar="",
        help="Weight decay for group regularizer (default: %(default)s)",
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

    if config["input_length"] <= 1:
        raise ValueError("input_length must be > 1")
    if config["output_length"] != 2:
        raise ValueError("output_length must be exactly 2 for binary decision output")
    if not 0.0 < config["confidence_threshold"] < 1.0:
        raise ValueError("confidence_threshold must be in (0, 1)")
    if config["logit_scale"] <= 0:
        raise ValueError("logit_scale must be > 0")

    if not os.path.exists("test_" + config["log_dir"]):
        os.makedirs("test_" + config["log_dir"])

    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]
    model_cfg["cuda"] = options_cfg["cuda"]
    options_cfg["logit_scale"] = dataset_cfg["logit_scale"]

    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    os.environ["PYTHONHASHSEED"] = str(train_cfg["seed"])
    if options["cuda"]:
        torch.cuda.manual_seed(train_cfg["seed"])
        torch.cuda.manual_seed_all(train_cfg["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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


    if WANDB_AVAILABLE:
        wandb.init(
            project="ATHEENA-ExitLUT",
            config={
                "hidden_layers": model_cfg["hidden_layers"],
                "support_layers": model_cfg["support_layers"],
                "input_bitwidth": model_cfg["input_bitwidth"],
                "hidden_bitwidth": model_cfg["hidden_bitwidth"],
                "output_bitwidth": model_cfg["output_bitwidth"],
                "support_bitwidth": model_cfg["support_bitwidth"],
                "input_fanin": model_cfg["input_fanin"],
                "hidden_fanin": model_cfg["hidden_fanin"],
                "output_fanin": model_cfg["output_fanin"],
                "support_fanin": model_cfg["support_fanin"],
                "width_n": model_cfg["width_n"],
                "weight_decay": train_cfg["weight_decay"],
                "batch_size": train_cfg["batch_size"],
                "epochs": train_cfg["epochs"],
                "learning_rate": train_cfg["learning_rate"],
                "seed": train_cfg["seed"],
                "dense_epochs": train_cfg["dense_epochs"],
                "wd": train_cfg["wd"],
                "wd1": train_cfg["wd1"],
                "confidence_threshold": dataset_cfg["confidence_threshold"],
                "logit_scale": dataset_cfg["logit_scale"],
                "dataset": "atheena-exit-logits",
            },
        )
        wandb.define_metric("Train Acc (%)", summary="max")
        wandb.define_metric("Test Acc (%)", summary="max")
        wandb.define_metric("Valid Acc(%)", summary="max")
        wandb.define_metric("Train Loss(%)", summary="min")
    

    x0, y0 = dataset["train"][0]
    model_cfg["input_length"] = len(x0)
    model_cfg["output_length"] = 2

    model_cfg['dense_forward'] = True
    model_cfg['imask'] = torch.tensor([]).cuda() if options_cfg["cuda"] else torch.tensor([])

    model_dense = ExitLUTNeqModel(model_cfg)
    train(model_dense, dataset, train_cfg, options_cfg)
    imask = GetInputMask(model_dense, options_cfg["cuda"])

    model_cfg['dense_forward'] = False
    model_cfg['imask'] = imask.cuda() if options_cfg["cuda"] else imask

    model = ExitLUTNeqModel(model_cfg)
    torch.save(imask, "test_" + options_cfg["log_dir"] + "/imask.pth")

    train(model, dataset, train_cfg, options_cfg)

    if WANDB_AVAILABLE:
        wandb.finish()

