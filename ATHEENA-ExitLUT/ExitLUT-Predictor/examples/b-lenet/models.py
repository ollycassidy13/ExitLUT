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

from functools import reduce
import shutil
from os.path import realpath

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ParameterScaling
from brevitas.nn import QuantHardTanh, QuantReLU, QuantIdentity

from pyverilator import PyVerilator

from neuralut.quant import QuantBrevitasActivation
from neuralut.nn import (
    SparseLinearNeq,
    ScalarBiasScale,
    SupportMask,
)


class ExitLUTNeqModel(nn.Module):
    def __init__(self, model_config):
        super(ExitLUTNeqModel, self).__init__()
        self.model_config = model_config
        self.is_cuda = model_config["cuda"]
        self.num_neurons = (
            [model_config["input_length"]]
            + model_config["hidden_layers"]
            + [model_config["output_length"]]
        )
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            initial_value = 1.33
            neurons_with_mask = (-1*(torch.tensor(model_config["support_layers"], dtype=int)-1))*torch.tensor(self.num_neurons[1:], dtype=int)
            if i == 1:
                print("Creating input layer")
                bn_in = nn.BatchNorm1d(in_features)
                input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(
                    QuantHardTanh(
                        bit_width = model_config["input_bitwidth"],
                        max_val=1.0,
                        min_val=-1.0,
                        act_scaling_impl=ParameterScaling(initial_value),
                        quant_type=QuantType.INT,
                        return_quant_tensor = False,
                        
                    ),
                    pre_transforms=[bn_in, input_bias],
                )
                if model_config["support_layers"][i] == 1:
                    output_quant = QuantBrevitasActivation(
                        QuantIdentity(
                            bit_width=model_config["support_bitwidth"],
                            quant_type=QuantType.INT,
                            scaling_impl=ParameterScaling(initial_value),
                            return_quant_tensor = False,
                        ),
                        pre_transforms=[bn],
                    )
                else:
                    output_quant = QuantBrevitasActivation(
                        QuantReLU(
                            bit_width=model_config["hidden_bitwidth"],
                            quant_type=QuantType.INT,
                            scaling_impl=ParameterScaling(initial_value),
                            return_quant_tensor = False,
                            
                        ),
                        pre_transforms=[bn],
                    )
                imask = model_config["imask"][:self.num_neurons[i]]
                imask = imask[:, (imask != 0).any(dim=0)]

                layer = SparseLinearNeq(
                    in_features,
                    out_features,
                    input_quant=input_quant,
                    output_quant=output_quant,
                    imask=imask,
                    support = False,
                    dense_forward=model_config["dense_forward"],
                    fan_in=model_config["input_fanin"],
                    width_n=model_config["width_n"],
                    cuda=self.is_cuda
                )
                layer_list.append(layer)
            elif i == len(self.num_neurons) - 1:
                print("Creating output layer")
                if model_config["support_layers"][i-1] == 1:
                    output_bias_scale = ScalarBiasScale(bias_init=0.33)
                    output_quant = QuantBrevitasActivation(
                        QuantIdentity(
                            bit_width=model_config["output_bitwidth"],
                            quant_type=QuantType.INT,
                            scaling_impl=ParameterScaling(initial_value),
                            return_quant_tensor = False,
                        ),
                        pre_transforms=[bn],
                        post_transforms=[output_bias_scale],
                    )
                    imask = SupportMask(in_features, model_config["support_fanin"])
                    layer = SparseLinearNeq(
                        in_features,
                        out_features,
                        input_quant=layer_list[-1].output_quant,
                        output_quant=output_quant,
                        imask=imask,
                        support = True,
                        dense_forward=False,
                        fan_in=model_config["support_fanin"],
                        width_n=model_config["width_n"],
                        apply_input_quant=False,
                        cuda=self.is_cuda
                    )
                    layer_list.append(layer)
                else:
                    output_bias_scale = ScalarBiasScale(bias_init=0.33)
                    output_quant = QuantBrevitasActivation(
                        QuantReLU(
                            bit_width=model_config["output_bitwidth"],
                            quant_type=QuantType.INT,
                            scaling_impl=ParameterScaling(initial_value),
                            return_quant_tensor = False,   
                        ),
                        pre_transforms=[bn],
                        post_transforms=[output_bias_scale],
                    )
                    if len(model_config["imask"]) != 0:
                        imask = model_config["imask"][sum(neurons_with_mask[:i-1]):sum(neurons_with_mask[:i])]
                    else:
                        imask = model_config["imask"]
                    imask = imask[:, (imask != 0).any(dim=0)]
                    layer = SparseLinearNeq(
                        in_features,
                        out_features,
                        input_quant=layer_list[-1].output_quant,
                        output_quant=output_quant,
                        imask=imask,
                        support = False,
                        dense_forward=model_config["dense_forward"],
                        fan_in=model_config["hidden_fanin"],
                        width_n=model_config["width_n"],
                        apply_input_quant=False,
                        cuda=self.is_cuda
                    )
                    layer_list.append(layer)
            else:
                if model_config["support_layers"][i-1] == 1:
                    print("Creating support layer")
                    if model_config["support_layers"][i] == 1:
                        output_quant = QuantBrevitasActivation(
                            QuantIdentity(
                                bit_width=model_config["support_bitwidth"],
                                quant_type=QuantType.INT,
                                scaling_impl=ParameterScaling(initial_value),
                                return_quant_tensor = False, 
                            ),
                            pre_transforms=[bn],
                        )
                    else:
                        output_quant = QuantBrevitasActivation(
                            QuantReLU(
                                bit_width=model_config["hidden_bitwidth"],
                                quant_type=QuantType.INT,
                                scaling_impl=ParameterScaling(initial_value),
                                return_quant_tensor = False,   
                            ),
                            pre_transforms=[bn],
                        )
                    imask = SupportMask(in_features, model_config["support_fanin"])
                    layer = SparseLinearNeq(
                        in_features,
                        out_features,
                        input_quant=layer_list[-1].output_quant,
                        output_quant=output_quant,
                        imask=imask,
                        support = True,
                        dense_forward=False,
                        fan_in=model_config["support_fanin"],
                        width_n=model_config["width_n"],
                        apply_input_quant=False,
                        cuda=self.is_cuda
                    )
                    layer_list.append(layer)
                else:
                    print("Creating hidden layer")
                    if model_config["support_layers"][i] == 1:
                        output_quant = QuantBrevitasActivation(
                            QuantIdentity(
                                bit_width=model_config["support_bitwidth"],
                                quant_type=QuantType.INT,
                                scaling_impl=ParameterScaling(initial_value),
                                return_quant_tensor = False,
                                
                            ),
                            pre_transforms=[bn],
                        )
                    else:
                        output_quant = QuantBrevitasActivation(
                            QuantReLU(
                                bit_width=model_config["hidden_bitwidth"],
                                quant_type=QuantType.INT,
                                scaling_impl=ParameterScaling(initial_value),
                                return_quant_tensor = False,
                                
                            ),
                            pre_transforms=[bn],
                        )
                    if len(model_config["imask"]) != 0:
                        imask = model_config["imask"][sum(neurons_with_mask[:i-1]):sum(neurons_with_mask[:i])]
                    else:
                        imask = model_config["imask"]
                    imask = imask[:, (imask != 0).any(dim=0)]
                    layer = SparseLinearNeq(
                        in_features,
                        out_features,
                        input_quant=layer_list[-1].output_quant,
                        output_quant=output_quant,
                        imask=imask,
                        support = False,
                        dense_forward=model_config["dense_forward"],
                        fan_in=model_config["hidden_fanin"],
                        width_n=model_config["width_n"],
                        apply_input_quant=False,
                        cuda=self.is_cuda
                    )
                    layer_list.append(layer)
        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False
        self.with_quant_layer = False
        self.with_max_output = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    def verilog_inference(
        self,
        verilog_dir,
        top_module_filename,
        logfile: bool = False,
        add_registers: bool = False,
        add_input_register: bool = False,
        add_output_register: bool = False,
        with_quant_layer: bool = False,
        with_max_output: bool = False,
    ):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        build_dir = f"{self.verilog_dir}/verilator"
        shutil.rmtree(build_dir, ignore_errors=True)
        self.dut = PyVerilator.build(
            f"{self.verilog_dir}/{self.top_module_filename}",
            verilog_path=[self.verilog_dir],
            build_dir=build_dir,
        )
        self.is_verilog_inference = True
        self.with_quant_layer = with_quant_layer
        self.with_max_output = with_max_output
        self.logfile = logfile
        num_verilog_layers = len(self.module_list) + int(with_quant_layer) + int(with_max_output)
        self.latency = (
            int(add_input_register)
            + (num_verilog_layers - 1) * int(add_registers)
            + int(add_output_register)
        )

    def pytorch_inference(self):
        self.is_verilog_inference = False

    def verilog_forward(self, x):
        output_quant = self.module_list[-1].output_quant
        _, output_bitwidth = output_quant.get_scale_factor_bits(self.is_cuda)
        output_bitwidth = int(output_bitwidth)
        total_output_bits = self.module_list[-1].out_features * output_bitwidth

        if self.with_quant_layer:
            input_quant = self.module_list[0].input_quant
            total_input_bits = self.module_list[0].in_features * 8
            num_layers = len(self.module_list) + 1 + int(self.with_max_output)
            input_quant.bin_output()
            self.module_list[0].apply_input_quant = False
            logit_scale = self.model_config.get("logit_scale", 128.0)
            x_int8_signed = torch.floor(x * 128.0 / logit_scale).long().clamp(-128, 127)
            x_int8u = x_int8_signed % 256
            x_quant = input_quant(x_int8_signed.float() * (logit_scale / 128.0))
            if not hasattr(self, '_vf_debug_printed'):
                self._vf_debug_printed = True
                s_vf, bits_vf = input_quant.get_scale_factor_bits(input_quant.cuda)
                print(f"[verilog_forward DEBUG] logit_scale={logit_scale}, is_bin_output={input_quant.is_bin_output}, "
                      f"input_quant.cuda={input_quant.cuda}, scale={s_vf.item():.6f}, bits={bits_vf.item()}")
                print(f"[verilog_forward DEBUG] x_int8_signed[0,:5]={x_int8_signed[0,:5].tolist()}")
                print(f"[verilog_forward DEBUG] x_quant[0,:5]={x_quant[0,:5].tolist()}")
            y = torch.zeros(x.shape[0], self.module_list[-1].out_features).to(x.device)
            self.dut.io.rst = 0
            self.dut.io.clk = 0
            for i in range(x.shape[0]):
                y_i = self.pytorch_forward(x_quant[i : i + 1, :])[0]
                ys_i = list(map(lambda z: output_quant.get_bin_str_from_int(z, self.is_cuda), y_i))
                xv_i = [f"{int(z):08b}" for z in x_int8u[i, :]]
                xvc_i = reduce(lambda a, b: a + b, xv_i[::-1])
                ysc_i = reduce(lambda a, b: a + b, ys_i[::-1])
                self.dut["M0"] = int(xvc_i, 2)
                for j in range(self.latency + 1):
                    res = self.dut[f"M{num_layers}"]
                    self.dut.io.clk = 1
                    self.dut.io.clk = 0
                if self.with_max_output:
                    hw_bit = res & 1
                    ysc_int = int(ysc_i, 2)
                    mask = (1 << output_bitwidth) - 1
                    sw_w0 = ysc_int & mask
                    sw_w1 = (ysc_int >> output_bitwidth) & mask
                    sw_bit = 1 if sw_w1 > sw_w0 else 0
                    if hw_bit != sw_bit:
                        print(f"\n=== MaxOutputLayer mismatch: hw={hw_bit}, sw={sw_bit}, w0={sw_w0}, w1={sw_w1} ===")
                        print(f"  x_int8u[{i},:] = {x_int8u[i,:].tolist()}")
                        print(f"  xvc_i (M0, hex) = 0x{int(xvc_i,2):020x}")
                        print(f"  SW x_quant = {x_quant[i,:].tolist()}")
                        _xt = x_quant[i:i+1, :].clone()
                        for _li, _layer in enumerate(self.module_list):
                            _xt = _layer(_xt)
                            print(f"  SW after module_list[{_li}]: {_xt[0].tolist()}")
                        raise AssertionError(
                            f"MaxOutputLayer mismatch: hw={hw_bit}, sw={sw_bit}, "
                            f"w0={sw_w0}, w1={sw_w1}"
                        )
                    assert hw_bit == sw_bit, (
                        f"MaxOutputLayer mismatch: hw={hw_bit}, sw={sw_bit}, "
                        f"w0={sw_w0}, w1={sw_w1}"
                    )
                    yv_i = torch.Tensor([1 - hw_bit, hw_bit]).to(y.device)
                else:
                    expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
                    result = f"{res:0{int(total_output_bits)}b}"
                    assert expected == result
                    res_split = [
                        result[j : j + output_bitwidth]
                        for j in range(0, len(result), output_bitwidth)
                    ][::-1]
                    yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split))).to(y.device)
                y[i, :] = yv_i
                if self.logfile is not None:
                    with open(self.logfile, "a") as f:
                        f.write(
                            f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n"
                        )
            return y

        input_quant = self.module_list[0].input_quant
        _, input_bitwidth = input_quant.get_scale_factor_bits(self.is_cuda)
        input_bitwidth = int(input_bitwidth)
        total_input_bits = self.module_list[0].in_features * input_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features).to(x.device)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i, :]
            y_i = self.pytorch_forward(x[i : i + 1, :])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str_from_int(z, self.is_cuda), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str_from_int(z, self.is_cuda), y_i))
            xvc_i = reduce(lambda a, b: a + b, xv_i[::-1])
            ysc_i = reduce(lambda a, b: a + b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert expected == result
            res_split = [
                result[i : i + output_bitwidth]
                for i in range(0, len(result), output_bitwidth)
            ][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split))).to(y.device)
            y[i, :] = yv_i
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(
                        f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n"
                    )
        return y

    def pytorch_forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)  


class ExitLUTLutModel(ExitLUTNeqModel):
    pass


class ExitLUTVerilogModel(ExitLUTNeqModel):
    pass

