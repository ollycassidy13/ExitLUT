# Copyright (c) 2026 Oliver Cassidy, Marta Andronic and George A. Constantinides
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0 (see ../LICENSE).

import argparse
import copy
import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.shape_inference
import onnxruntime
import torch
import torchvision.datasets as tvds
import torchvision.transforms as T

DEFAULT_LOGIT_SCALE = 128.0
DEFAULT_THRESHOLD = 0.999
DEFAULT_BATCH_SIZE = 256
DATASET_LOAD_BATCH_SIZE = 512
INT8_MIN, INT8_MAX = -128, 127


def merge_subgraphs(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, List[str]]:
    model = copy.deepcopy(model)
    subnodes: List[onnx.NodeProto] = []
    valinfs: List[onnx.ValueInfoProto] = []
    if_idxs: List[int] = []
    else_branch_outputs: List[str] = []

    for idx, node in enumerate(model.graph.node):
        if node.op_type != "If":
            continue
        for attr in node.attribute:
            sg = attr.g
            for sg_node in sg.node:
                new_node = onnx.NodeProto()
                new_node.CopyFrom(sg_node)
                subnodes.append(new_node)
            for vi in list(sg.value_info) + list(sg.output):
                vip = onnx.ValueInfoProto()
                vip.CopyFrom(vi)
                valinfs.append(vip)
            if attr.name == "else_branch":
                for out in sg.output:
                    else_branch_outputs.append(out.name)
        for if_out in node.output:
            for existing in list(model.graph.output):
                if existing.name == if_out:
                    model.graph.output.remove(existing)
        if_idxs.append(idx)

    model.graph.node.extend(subnodes)
    model.graph.value_info.extend(valinfs)
    for idx in reversed(if_idxs):
        del model.graph.node[idx]

    return model, else_branch_outputs


def find_exit_logit_info(model: onnx.ModelProto) -> List[Tuple[str, float]]:
    out_to_node: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for o in node.output:
            out_to_node[o] = node

    init_map: Dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        init_map[init.name] = onnx.numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                try:
                    init_map[node.output[0]] = onnx.numpy_helper.to_array(attr.t)
                except Exception:
                    pass

    exits = []
    for node in model.graph.node:
        if node.op_type != "Greater" or len(node.input) < 2:
            continue

        thr_name = node.input[1]
        if thr_name not in init_map:
            continue
        threshold = float(init_map[thr_name].flat[0])

        logit_name = node.input[0]
        if logit_name in out_to_node:
            pred = out_to_node[logit_name]
            if pred.op_type == "ReduceMax":
                logit_name = pred.input[0]
                if logit_name in out_to_node:
                    pred2 = out_to_node[logit_name]
                    if pred2.op_type == "Softmax":
                        logit_name = pred2.input[0]

        exits.append((logit_name, threshold))

    return exits


def _make_batch_dynamic(model: onnx.ModelProto) -> onnx.ModelProto:
    for tensor in list(model.graph.input) + list(model.graph.output):
        t = tensor.type.tensor_type
        if t.HasField("shape") and t.shape.dim:
            t.shape.dim[0].ClearField("dim_value")
            t.shape.dim[0].dim_param = "batch"
    return model


def _vi_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    m: Dict[str, onnx.ValueInfoProto] = {}
    for vi in model.graph.value_info:
        m[vi.name] = vi
    for o in model.graph.output:
        m[o.name] = o
    for i in model.graph.input:
        m[i.name] = i
    return m


def build_eval_session(
        onnx_path: str,
) -> Tuple[onnxruntime.InferenceSession, List[str], List[float], str, str]:
    model = onnx.load(onnx_path)

    original_output_name = model.graph.output[0].name

    model, else_outputs = merge_subgraphs(model)
    model = _make_batch_dynamic(model)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    exit_info = find_exit_logit_info(model)
    exit_logit_names = [name for name, _ in exit_info]
    thresholds = [thr for _, thr in exit_info]

    final_exit_name = else_outputs[-1] if else_outputs else original_output_name

    vi = _vi_map(model)

    if exit_logit_names and final_exit_name in vi:
        n_classes = None
        ei_vi = vi.get(exit_logit_names[0])
        if ei_vi is not None:
            s = ei_vi.type.tensor_type.shape
            if s and len(s.dim) >= 2:
                n_classes = s.dim[-1].dim_value or None
        if n_classes:
            fe_vi = vi.get(final_exit_name)
            fe_shape = fe_vi.type.tensor_type.shape if fe_vi else None
            fe_ndim = len(fe_shape.dim) if fe_shape else 0
            if fe_ndim != 2:
                early_set = set(exit_logit_names)
                for node in model.graph.node:
                    for out_name in node.output:
                        if out_name in early_set or out_name not in vi:
                            continue
                        s2 = vi[out_name].type.tensor_type.shape
                        if s2 and len(s2.dim) == 2 and s2.dim[-1].dim_value == n_classes:
                            final_exit_name = out_name
    added = {o.name for o in model.graph.output}

    for name in exit_logit_names + [final_exit_name]:
        if name in added:
            continue
        if name in vi:
            vip = onnx.ValueInfoProto()
            vip.CopyFrom(vi[name])
        else:
            vip = onnx.ValueInfoProto()
            vip.name = name
        model.graph.output.append(vip)
        added.add(name)

    model = _make_batch_dynamic(model)

    available_providers = onnxruntime.get_available_providers()
    providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers
                 else []) + ["CPUExecutionProvider"]
    sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)
    device_used = "CUDA" if "CUDAExecutionProvider" in available_providers else "CPU"
    print(f"  onnxruntime device : {device_used}")
    input_name = sess.get_inputs()[0].name

    return sess, exit_logit_names, thresholds, final_exit_name, input_name


def load_dataset(name: str, split: str, data_dir: str,
                 num_samples: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    transform = T.Compose([T.ToTensor()])
    is_train = split == "train"
    name_lower = name.lower()

    if name_lower == "cifar10":
        ds = tvds.CIFAR10(data_dir, train=is_train, download=True, transform=transform)
    elif name_lower == "mnist":
        ds = tvds.MNIST(data_dir, train=is_train, download=True, transform=transform)
    elif name_lower in ("fashionmnist", "fashion_mnist"):
        ds = tvds.FashionMNIST(data_dir, train=is_train, download=True, transform=transform)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: cifar10, mnist, fashionmnist.")

    if num_samples is not None:
        ds = torch.utils.data.Subset(ds, range(min(num_samples, len(ds))))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=DATASET_LOAD_BATCH_SIZE, shuffle=False, num_workers=0)

    data_chunks, label_chunks = [], []
    for imgs, lbls in loader:
        data_chunks.append(imgs.numpy())
        label_chunks.append(lbls.numpy())

    return (np.concatenate(data_chunks, axis=0),
            np.concatenate(label_chunks, axis=0).astype(np.int64))


def _softmaxes_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "softmaxes")


def _neuralut_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ExitLUT-Predictor")


def _find_neuralut_dir(threshold: float, softmaxes_dir: str) -> Tuple[str, float]:
    candidates = glob.glob(os.path.join(softmaxes_dir, "verilog-*"))
    if not candidates:
        raise FileNotFoundError(
            f"No neuralut directories found under {softmaxes_dir!r}.")
    best_dir, best_delta, best_thr = None, float("inf"), None
    for c in candidates:
        try:
            thr = float(os.path.basename(c).split("-", 1)[1])
            d = abs(thr - threshold)
            if d < best_delta:
                best_delta, best_dir, best_thr = d, c, thr
        except (ValueError, IndexError):
            continue
    if best_dir is None:
        raise FileNotFoundError(
            f"Could not parse any threshold from dirs in {softmaxes_dir!r}.")
    return best_dir, best_thr


def _infer_neuralut_cfg(model_dict: Dict[str, torch.Tensor]) -> dict:
    layer_idxs = sorted(set(
        int(k.split(".")[1]) for k in model_dict if k.startswith("module_list.")
    ))
    hidden_sizes = []
    fan_ins = []
    for i in layer_idxs:
        rd = model_dict.get(f"module_list.{i}.res2_dense.weight")
        fc = model_dict.get(f"module_list.{i}.fc1.weight")
        if rd is not None:
            hidden_sizes.append(rd.shape[0])
        if fc is not None:
            fan_ins.append(fc.shape[1])
    width_n = model_dict[f"module_list.{layer_idxs[0]}.fc4.weight"].shape[1]
    fan_in = fan_ins[0] if fan_ins else 2
    return {
        "hidden_layers":    hidden_sizes[:-1],
        "support_layers":   [0] + [1] * (len(layer_idxs) - 1),
        "input_length":     model_dict["module_list.0.input_quant.pre_transforms.0.weight"].shape[0],
        "output_length":    hidden_sizes[-1],
        "input_bitwidth":   4,
        "hidden_bitwidth":  3,
        "output_bitwidth":  3,
        "support_bitwidth": 3,
        "input_fanin":      fan_in,
        "hidden_fanin":     fan_in,
        "output_fanin":     fan_in,
        "support_fanin":    fan_in,
        "width_n":          width_n,
    }


def load_neuralut(threshold: float, softmaxes_dir: str, logit_scale: float = DEFAULT_LOGIT_SCALE,
                  neuralut_dir: Optional[str] = None) -> Tuple[torch.nn.Module, float]:
    if neuralut_dir is not None:
        best_dir, best_thr = neuralut_dir, threshold
    else:
        best_dir, best_thr = _find_neuralut_dir(threshold, softmaxes_dir)
    pth_path = os.path.join(best_dir, "lut_based_model.pth")
    imask_path = os.path.join(best_dir, "imask.pth")

    nlut_root = _neuralut_root()
    models_dir = os.path.join(nlut_root, "examples", "b-lenet")
    src_dir    = os.path.join(nlut_root, "src")
    for p in [src_dir, models_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    for mod in ("pyverilator", "tclwrapper"):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # these modules are in /ExitLUT-Predictor and so only
    # importable after the sys.path and pyverilator mock setup above.
    from models import ExitLUTNeqModel
    from neuralut.nn import generate_truth_tables, lut_inference
    from neuralut.quant import QuantBrevitasActivation

    ckpt = torch.load(pth_path, map_location="cpu")
    model_cfg = dict(ckpt.get("model_cfg") or _infer_neuralut_cfg(ckpt["model_dict"]))
    ckpt_scale = model_cfg.get("logit_scale")
    if ckpt_scale is not None and float(ckpt_scale) != float(logit_scale):
        raise ValueError(
            f"--logit-scale {logit_scale} does not match checkpoint logit_scale {ckpt_scale}")
    use_cuda = torch.cuda.is_available()
    model_cfg["cuda"] = use_cuda
    model_cfg["dense_forward"] = False
    model_cfg["imask"] = torch.load(imask_path, map_location="cuda" if use_cuda else "cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExitLUTNeqModel(model_cfg)
    model.load_state_dict(ckpt["model_dict"])
    for m in model.modules():
        if isinstance(m, QuantBrevitasActivation):
            m.cuda = use_cuda
    if use_cuda:
        model.cuda()
    model.eval()
    generate_truth_tables(model)
    lut_inference(model)

    test_acc = ckpt.get('test_accuracy', float('nan'))
    print(f"  NeuralUT: loaded {pth_path}  (threshold={best_thr}, "
          f"LUT test_acc={test_acc:.2f}%)")
    return model, best_thr


def exit_decision_batch(logits_batch: np.ndarray, *,
                        neuralut_model=None,
                        threshold: float,
                        logit_scale: float = DEFAULT_LOGIT_SCALE) -> np.ndarray:
    logits_batch = np.asarray(logits_batch, dtype=np.float32)
    if logits_batch.ndim == 1:
        logits_batch = logits_batch[np.newaxis, :]

    if neuralut_model is not None:
        x = logits_batch.astype(np.float64)
        x_int8 = np.clip(np.floor(x), INT8_MIN, INT8_MAX).astype(np.int32)
        x_float = (x_int8.astype(np.float32)) * (logit_scale / DEFAULT_LOGIT_SCALE)
        with torch.no_grad():
            t = torch.from_numpy(x_float)
            out = neuralut_model(t)
            return torch.argmax(out, dim=-1).cpu().numpy().astype(bool)

    x = logits_batch.astype(np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs.max(axis=1) >= threshold


def evaluate(sess: onnxruntime.InferenceSession,
             exit_logit_names: List[str],
             thresholds: List[float],
             final_exit_name: str,
             input_name: str,
             data: np.ndarray,
             labels: np.ndarray,
             use_neuralut: bool,
             neuralut_model,
             threshold: float,
             batch_size: int,
             logit_scale: float = DEFAULT_LOGIT_SCALE) -> dict:
    n = len(labels)
    n_exits = len(exit_logit_names) + 1

    exit_correct = [0] * n_exits
    exit_count   = [0] * n_exits

    n_early = len(exit_logit_names)
    agree_exit     = [0] * n_early
    agree_cont     = [0] * n_early
    nlut_only_exit = [0] * n_early
    sm_only_exit   = [0] * n_early

    req_outputs: List[str] = []
    seen: set = set()
    for name in exit_logit_names + [final_exit_name]:
        if name not in seen:
            req_outputs.append(name)
            seen.add(name)
    final_req_idx = req_outputs.index(final_exit_name)

    for start in range(0, n, batch_size):
        batch_data   = data[start : start + batch_size].astype(np.float32)
        batch_labels = labels[start : start + batch_size]
        B = len(batch_labels)

        try:
            results = sess.run(req_outputs, {input_name: batch_data})
        except Exception as exc:
            print(f"\nERROR: onnxruntime failed on batch {start}–{start+B}: {exc}",
                  file=sys.stderr)
            raise

        taken = np.full(B, -1, dtype=int)

        for ei, logit_name in enumerate(exit_logit_names):
            pending_mask = taken == -1
            if not pending_mask.any():
                break

            logits = np.asarray(results[req_outputs.index(logit_name)],
                                 dtype=np.float32)
            if logits.ndim == 1:
                logits = logits.reshape(B, -1)

            eff_thr = thresholds[ei] if ei < len(thresholds) else threshold
            pending_logits = logits[pending_mask]
            decisions = exit_decision_batch(
                pending_logits,
                neuralut_model=neuralut_model if use_neuralut else None,
                threshold=eff_thr,
                logit_scale=logit_scale)

            if use_neuralut:
                sm_decisions = exit_decision_batch(
                    pending_logits,
                    neuralut_model=None,
                    threshold=eff_thr,
                    logit_scale=logit_scale)
                agree_exit[ei]     += int((decisions &  sm_decisions).sum())
                agree_cont[ei]     += int((~decisions & ~sm_decisions).sum())
                nlut_only_exit[ei] += int((decisions & ~sm_decisions).sum())
                sm_only_exit[ei]   += int((~decisions &  sm_decisions).sum())

            pending_indices = np.where(pending_mask)[0]
            for pi, decided in zip(pending_indices, decisions):
                if decided:
                    taken[pi] = ei

        taken[taken == -1] = n_exits - 1

        final_logits = np.asarray(results[final_req_idx], dtype=np.float32)
        if final_logits.ndim == 1:
            final_logits = final_logits.reshape(B, -1)

        for ei in range(n_exits - 1):
            mask = taken == ei
            if not mask.any():
                continue
            ei_logits = np.asarray(
                results[req_outputs.index(exit_logit_names[ei])],
                dtype=np.float32)
            if ei_logits.ndim == 1:
                ei_logits = ei_logits.reshape(B, -1)
            preds = ei_logits[mask].argmax(axis=1)
            exit_correct[ei] += int((preds == batch_labels[mask]).sum())
            exit_count[ei]   += int(mask.sum())

        mask = taken == (n_exits - 1)
        if mask.any():
            preds = final_logits[mask].argmax(axis=1)
            exit_correct[-1] += int((preds == batch_labels[mask]).sum())
            exit_count[-1]   += int(mask.sum())

        print(f"  {min(start + batch_size, n)}/{n} samples ...", end="\r")

    print()

    total = sum(exit_count)
    total_correct = sum(exit_correct)
    accuracy = total_correct / total if total else 0.0

    exits_out = []
    for i in range(n_exits):
        cnt = exit_count[i]
        e = {
            "exit":     i,
            "count":    cnt,
            "fraction": cnt / total if total else 0.0,
            "correct":  exit_correct[i],
            "accuracy": exit_correct[i] / cnt if cnt else 0.0,
        }
        if use_neuralut and i < n_early:
            evaluated = agree_exit[i] + agree_cont[i] + nlut_only_exit[i] + sm_only_exit[i]
            e["agreement"] = {
                "evaluated":     evaluated,
                "agree_exit":    agree_exit[i],
                "agree_cont":    agree_cont[i],
                "nlut_only_exit": nlut_only_exit[i],
                "sm_only_exit":  sm_only_exit[i],
                "agree_rate":    (agree_exit[i] + agree_cont[i]) / evaluated if evaluated else 0.0,
            }
        exits_out.append(e)

    return {
        "accuracy": accuracy,
        "correct":  total_correct,
        "total":    total,
        "exits":    exits_out,
    }


def print_results(results: dict, config: dict) -> None:
    print()
    print(f"Overall accuracy : {results['accuracy']:.2%}"
          f"  ({results['correct']}/{results['total']})")
    print()
    print(f"{'Exit':>5} | {'Samples':>8} | {'Fraction':>9} | {'Accuracy':>9}")
    print("------+----------+-----------+-----------")
    for e in results["exits"]:
        print(f"{e['exit']:>5} | {e['count']:>8} | {e['fraction']:>8.1%}"
              f"   | {e['accuracy']:>8.2%}")

    early_with_agreement = [e for e in results["exits"] if "agreement" in e]
    if early_with_agreement:
        print()
        print("NeuralUT vs softmax agreement at each early exit:")
        print(f"{'Exit':>5} | {'Evaluated':>10} | {'Agree%':>7} | "
              f"{'Both exit':>10} | {'Both cont':>10} | "
              f"{'NLut only':>10} | {'SM only':>9}")
        print("------+------------+---------+------------+------------+------------+-----------")
        for e in early_with_agreement:
            ag = e["agreement"]
            print(f"{e['exit']:>5} | {ag['evaluated']:>10} | {ag['agree_rate']:>6.1%} | "
                  f"{ag['agree_exit']:>10} | {ag['agree_cont']:>10} | "
                  f"{ag['nlut_only_exit']:>10} | {ag['sm_only_exit']:>9}")

    print()
    print(f"Config: dataset={config['dataset']}  split={config['split']}"
          f"  neuralut={config['use_neuralut']}  threshold={config['threshold']}")


def collect_and_save_logits(
        sess: onnxruntime.InferenceSession,
        exit_logit_names: List[str],
        final_exit_name: str,
        input_name: str,
        data: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        batch_size: int,
        output_path: str) -> None:
    if not exit_logit_names:
        print("  No early exits in this model — nothing to collect.", file=sys.stderr)
        return

    target_name = exit_logit_names[0]
    req_outputs = list(dict.fromkeys([target_name, final_exit_name]))
    target_idx  = req_outputs.index(target_name)

    n = len(labels)
    all_logits: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch = data[start : start + batch_size].astype(np.float32)
        results = sess.run(req_outputs, {input_name: batch})
        logits = np.asarray(results[target_idx], dtype=np.float32)
        B = len(batch)
        if logits.ndim == 1:
            logits = logits.reshape(B, -1)
        all_logits.append(logits)
        print(f"  {min(start + batch_size, n)}/{n} samples ...", end="\r")
    print()

    all_logits_arr = np.concatenate(all_logits, axis=0)

    x = all_logits_arr.astype(np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    exit_labels = (probs.max(axis=1) >= threshold).astype(np.int32)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path,
             logits=all_logits_arr,
             exit_labels=exit_labels,
             gt_labels=labels.astype(np.int64),
             threshold=np.float32(threshold))

    frac = exit_labels.mean()
    print(f"  Saved {n} logit vectors → {output_path!r}")
    print(f"  Exit-label balance: {frac:.1%} exit=1,  {1-frac:.1%} exit=0"
          f"  (threshold={threshold})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate an ATHEENA ONNX model: accuracy + exit distribution.")
    ap.add_argument("--onnx",        required=True,
                    help="Path to the ONNX model.")
    ap.add_argument("--dataset",     required=True,
                    help="Dataset name: cifar10 | mnist | fashionmnist.")
    ap.add_argument("--split",       default="test", choices=["test", "train"],
                    help="Dataset split (default: test).")
    ap.add_argument("--data-dir",    default="./data",
                    help="Torchvision data directory (default: ./data).")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Maximum number of samples to evaluate.")
    ap.add_argument("--neuralut",    action="store_true",
                    help="Use the NeuralUT LUT checkpoint for exit decisions.")
    ap.add_argument("--threshold",   type=float, default=None,
                    help="Confidence threshold override (auto-detected if omitted).")
    ap.add_argument("--neuralut-dir", default=None,
                    help="neq2lut.py --log-dir to load (default: closest "
                         "softmaxes/verilog-<threshold> next to this script).")
    ap.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--logit-scale", type=float, default=DEFAULT_LOGIT_SCALE,
                    help="Logit scale used when training the NeuralUT checkpoint "
                         "(default: 128.0 for ap_fixed<16,8>). Must match train.py --logit_scale.")
    ap.add_argument("--save-logits", default=None,
                    help="Save early-exit logits to this .npz file for NeuralUT training. "
                         "Use --split train to collect training data.")
    ap.add_argument("--output",      default=None,
                    help="Write results to this JSON file.")
    args = ap.parse_args()

    print(f"Loading ONNX model: {args.onnx}")
    (sess, exit_logit_names, auto_thresholds,
     final_exit_name, input_name) = build_eval_session(args.onnx)

    if exit_logit_names:
        print(f"  {len(exit_logit_names)} early exit(s) detected, "
              f"auto-detected threshold(s): {auto_thresholds}")
    else:
        print("  No early exits detected – evaluating as baseline model")
    print(f"  Final exit tensor : {final_exit_name!r}")

    threshold = args.threshold
    if threshold is None:
        threshold = auto_thresholds[0] if auto_thresholds else DEFAULT_THRESHOLD
    print(f"  Effective threshold : {threshold}")

    neuralut_model = None
    if args.neuralut:
        sdir = _softmaxes_dir()
        neuralut_model, _ = load_neuralut(threshold, sdir, logit_scale=args.logit_scale,
                                          neuralut_dir=args.neuralut_dir)

    print(f"\nLoading dataset: {args.dataset} ({args.split})")
    data, labels = load_dataset(
        args.dataset, args.split, args.data_dir, args.num_samples)
    print(f"  {len(labels)} samples, shape: {data.shape}")

    if args.save_logits:
        print("\nCollecting early-exit logits...")
        collect_and_save_logits(
            sess, exit_logit_names, final_exit_name, input_name,
            data, labels, threshold, args.batch_size, args.save_logits)
        return

    print("\nEvaluating...")
    thresholds = ([args.threshold] * len(exit_logit_names)
                  if args.threshold is not None else auto_thresholds)
    results = evaluate(
        sess, exit_logit_names, thresholds,
        final_exit_name, input_name,
        data, labels,
        use_neuralut=args.neuralut,
        neuralut_model=neuralut_model,
        threshold=threshold,
        batch_size=args.batch_size,
        logit_scale=args.logit_scale)

    config = {
        "onnx":         args.onnx,
        "dataset":      args.dataset,
        "split":        args.split,
        "num_samples":  int(results["total"]),
        "use_neuralut": args.neuralut,
        "threshold":    threshold,
    }
    print_results(results, config)

    if args.output:
        out_data = {**results, "config": config}
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
