#!/bin/bash
# Copyright (c) 2026 Oliver Cassidy, Marta Andronic and George A. Constantinides
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0 (see ../LICENSE).
# Usage: bash run.sh [DATA_DIR]   (DATA_DIR is where MNIST is stored/downloaded; default: ./data)

R=$(cd "$(dirname "$0")" && pwd)
DATA=${1:-$R/data}
ONNX=$R/optimiser/examples/models/atheena/branchy_lenet_20220902.onnx
NPZ=$R/softmaxes/mnist_train_logits_0.996.npz

# disable wandb logging
# export WANDB_MODE=disabled

cd $R
python evaluate_onnx.py --onnx $ONNX --dataset mnist --split train \
    --data-dir $DATA --save-logits $NPZ

cd $R/ExitLUT-Predictor/examples/b-lenet
python train.py --arch b-lenet --log_dir exitlut_0.996_rerun \
    --dataset_file $NPZ --cuda --device 0

python neq2lut.py --arch b-lenet \
    --checkpoint test_exitlut_0.996_rerun/best_accuracy.pth \
    --imask test_exitlut_0.996_rerun/imask.pth \
    --dataset-file $NPZ \
    --log-dir $R/softmaxes/verilog-0.996 \
    --add-max-output --add-input-register --add-output-register \
    --cuda --device 0

cd $R
python evaluate_onnx.py --onnx $ONNX --dataset mnist --split test \
    --data-dir $DATA --neuralut --output $R/softmaxes/exitlut_test_0.996.json
