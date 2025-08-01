# extract approximating LoRA by svd from two FLUX models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import json
import os
import time
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
from .library import flux_utils, sai_model_spec
from .library.utils import MemoryEfficientSafeOpen
from .library.utils import setup_logging
from .networks import lora_flux

# Imports needed for the new quant_svd function
from typing import Dict, Tuple

setup_logging()
import logging

logger = logging.getLogger(__name__)

from comfy.utils import ProgressBar

# ===============================================================================
# START: NEW CODE FOR QUANTIZATION (added without modifying existing code)
# ===============================================================================

# --- Configuration for FP8 Quantization ---
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32  # Use float32 for stable delta calculations
SCALE_DTYPE = torch.float32

# --- Helper functions for FP8 quantization with stochastic rounding ---

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_min_pos = 2**-9  # Smallest subnormal for E4M3FN
    elif fp8_dtype == torch.float8_e5m2:
        fp8_min_pos = 2**-16  # Smallest subnormal for E5M2
    else:
        fp8_min_pos = finfo.tiny
    return float(finfo.min), float(finfo.max), float(fp8_min_pos)

# Global FP8 constants for our new function
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )
    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    exponent = torch.clamp(torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS, 0, 2**EXPONENT_BITS - 1)
    normal_mask = ~(exponent == 0)
    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )
    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign

def stochastic_rounding(value, dtype=TARGET_FP8_DTYPE, seed=0):
    if dtype in [torch.float32, torch.float16, torch.bfloat16]:
        return value.to(dtype=dtype)
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        # Slicing for large tensors to avoid potential issues
        num_slices = max(1, (value.numel() // (1536 * 1536)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i + slice_size].copy_(manual_stochastic_round_to_float8(value[i:i + slice_size], dtype, generator=generator))
        return output
    return value.to(dtype=dtype)


def quant_svd(
    model_org=None,
    save_to=None,
    dim=4,
    device=None,
    store_device='cpu',
    save_precision=None,
    outlier_quantile=0.99,
    no_metadata=False,
    mem_eff_safe_open=False,
):
    """
    Creates a LoRA by extracting the outlier quantization errors of a model.
    This function quantizes a model to FP8 and then calculates the difference
    between the original and the de-quantized version. It then isolates the
    largest errors (outliers) and performs SVD on them to create a
    'quantization error correction' LoRA.
    """
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    save_dtype = str_to_dtype(save_precision)

    if not mem_eff_safe_open:
        open_fn = lambda fn: safe_open(fn, framework="pt")
    else:
        logger.info("Using memory efficient safe_open")
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)

    lora_weights = {}
    with open_fn(model_org) as fo:
        keys = []
        for key in fo.keys():
            if not ("single_block" in key or "double_block" in key or "final_layer" in key or "txt_in" in key or "img_in" in key):
                continue
            if ".bias" in key or "norm" in key:
                continue
            if not key.endswith(".weight"):
                continue
            keys.append(key)

        comfy_pbar = ProgressBar(len(keys))
        for key in tqdm(keys, desc="Processing Quantization Deltas"):
            # 1. Get original tensor
            original_tensor = fo.get_tensor(key).to(COMPUTE_DTYPE)
            if original_tensor.numel() == 0:
                continue

            # 2. Quantize the tensor to FP8
            abs_max = torch.max(torch.abs(original_tensor))
            if abs_max < 1e-12:
                scale_factor = torch.tensor(1.0, dtype=COMPUTE_DTYPE)
            else:
                scale_factor = (FP8_MAX - FP8_MIN_POS) / abs_max.clamp(min=FP8_MIN_POS)

            scaled_tensor = original_tensor.mul(scale_factor)
            clamped_tensor = torch.clamp(scaled_tensor, FP8_MIN, FP8_MAX)
            quantized_fp8_tensor = stochastic_rounding(clamped_tensor, dtype=TARGET_FP8_DTYPE)

            # 3. De-quantize back to high precision
            dequant_scale = scale_factor.reciprocal()
            dequantized_tensor = quantized_fp8_tensor.to(COMPUTE_DTYPE) * dequant_scale

            # 4. Calculate the quantization error delta
            delta = original_tensor - dequantized_tensor

            # 5. Isolate outliers (the opposite of clamping)
            if outlier_quantile < 1.0:
                abs_delta = delta.abs()

                # --- START OF THE FIX ---
                # Use torch.kthvalue for memory efficiency on huge tensors
                flat_delta = abs_delta.flatten()
                num_elements = flat_delta.numel()

                # Calculate the index 'k' corresponding to the quantile
                # Ensure k is a valid index (between 1 and num_elements)
                k = min(max(1, int(outlier_quantile * num_elements)), num_elements)

                # Find the k-th smallest value, which is our quantile threshold
                threshold = torch.kthvalue(flat_delta, k).values
                # --- END OF THE FIX ---

                mask = abs_delta >= threshold
                outlier_delta = delta * mask
            else:
                outlier_delta = delta # No outlier isolation if quantile is 1.0

            # 6. Perform SVD on the outlier delta matrix
            if device:
                outlier_delta = outlier_delta.to(device)

            out_dim, in_dim = outlier_delta.size()[0:2]
            rank = min(dim, in_dim, out_dim)

            mat = outlier_delta.squeeze()
            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)
            Vh = Vh[:rank, :]

            # No clamping here, as we isolated outliers *before* SVD
            U = U.to(store_device, dtype=save_dtype).contiguous()
            Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

            logger.info(f"key: {key}, U: {U.size()}, Vh: {Vh.size()}")
            comfy_pbar.update(1)
            lora_weights[key] = (U, Vh)
            del mat, U, S, Vh, delta, outlier_delta, original_tensor

    # 7. Make state dict for LoRA (same logic as original svd function)
    lora_sd = {}
    for key, (up_weight, down_weight) in lora_weights.items():
        lora_name = key.replace(".weight", "").replace(".", "_")
        lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + lora_name
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])

    net_kwargs = {}
    metadata = {
        "ss_v2": str(False),
        "ss_base_model_version": flux_utils.MODEL_VERSION_FLUX_V1,
        "ss_network_module": "networks.lora_flux",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
        "ss_comment": f"LoRA extracted from quantization errors with outlier_quantile={outlier_quantile}"
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(lora_sd, False, False, False, True, False, time.time(), title, flux="dev")
        metadata.update(sai_metadata)

    save_to_file(save_to, lora_sd, metadata, save_dtype)
    logger.info(f"Quantization Error LoRA weights saved to {save_to}")
    return save_to



# ===============================================================================
# END: NEW CODE
# ===============================================================================


def save_to_file(file_name, state_dict, metadata, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    save_file(state_dict, file_name, metadata=metadata)


def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    device=None,
    store_device='cpu',
    save_precision=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
    mem_eff_safe_open=False,
):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    calc_dtype = torch.float
    save_dtype = str_to_dtype(save_precision)

    # open models
    lora_weights = {}
    if not mem_eff_safe_open:
        # use original safetensors.safe_open
        open_fn = lambda fn: safe_open(fn, framework="pt")
    else:
        logger.info("Using memory efficient safe_open")
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)

    with open_fn(model_org) as fo:
        # filter keys or "img_in" in key
        keys = []
        for key in fo.keys():
            if not ("single_block" in key or "double_block" in key or "final_layer" in key or "txt_in" in key or "img_in" in key):
                continue
            if ".bias" in key:
                continue
            if "norm" in key:
                continue
            keys.append(key)
        comfy_pbar = ProgressBar(len(keys))
        with open_fn(model_tuned) as ft:
            for key in tqdm(keys):
                # get tensors and calculate difference
                value_o = fo.get_tensor(key)
                value_t = ft.get_tensor(key)
                mat = value_t.to(calc_dtype) - value_o.to(calc_dtype)
                del value_o, value_t

                # extract LoRA weights
                if device:
                    mat = mat.to(device)
                out_dim, in_dim = mat.size()[0:2]
                rank = min(dim, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

                mat = mat.squeeze()

                U, S, Vh = torch.linalg.svd(mat)

                U = U[:, :rank]
                S = S[:rank]
                U = U @ torch.diag(S)

                Vh = Vh[:rank, :]

                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist, clamp_quantile)
                low_val = -hi_val

                U = U.clamp(low_val, hi_val)
                Vh = Vh.clamp(low_val, hi_val)

                U = U.to(store_device, dtype=save_dtype).contiguous()
                Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

                print(f"key: {key}, U: {U.size()}, Vh: {Vh.size()}")
                comfy_pbar.update(1)
                lora_weights[key] = (U, Vh)
                del mat, U, S, Vh

    # make state dict for LoRA
    lora_sd = {}
    for key, (up_weight, down_weight) in lora_weights.items():
        lora_name = key.replace(".weight", "").replace(".", "_")
        lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + lora_name
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])  # same as rank

    # minimum metadata
    net_kwargs = {}
    metadata = {
        "ss_v2": str(False),
        "ss_base_model_version": flux_utils.MODEL_VERSION_FLUX_V1,
        "ss_network_module": "networks.lora_flux",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(lora_sd, False, False, False, True, False, time.time(), title, flux="dev")
        metadata.update(sai_metadata)

    save_to_file(save_to, lora_sd, metadata, save_dtype)

    logger.info(f"LoRA weights saved to {save_to}")
    return save_to


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # NEW ARGUMENT TO CHOOSE MODE
    parser.add_argument(
        "--mode",
        type=str,
        default="svd",
        choices=["svd", "quant_svd"],
        help="Extraction mode. 'svd' for normal diff, 'quant_svd' for quantization error extraction.",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はfloat",
    )
    parser.add_argument(
        "--model_org",
        type=str,
        default=None,
        required=True,
        help="Original model: safetensors file. For quant_svd, this is the high-precision source model.",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        default=None,
        help="Tuned model for 'svd' mode. Ignored in 'quant_svd' mode.",
    )
    parser.add_argument(
        "--mem_eff_safe_open",
        action="store_true",
        help="use memory efficient safe_open. This is an experimental feature, use only when memory is not enough.",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: safetensors file",
    )
    parser.add_argument(
        "--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU"
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="For 'svd' mode: Quantile clamping value, float, (0-1). Default = 0.99",
    )
    # NEW ARGUMENT FOR QUANT_SVD MODE
    parser.add_argument(
        "--outlier_quantile",
        type=float,
        default=0.99,
        help="For 'quant_svd' mode: keep only the error values above this quantile (0-1). E.g., 0.99 keeps the top 1%% of errors.",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.mode == 'svd':
        if not args.model_tuned:
            parser.error("--model_tuned is required for 'svd' mode.")
        # Call original function with its specific arguments
        svd(
            model_org=args.model_org,
            model_tuned=args.model_tuned,
            save_to=args.save_to,
            dim=args.dim,
            device=args.device,
            save_precision=args.save_precision,
            clamp_quantile=args.clamp_quantile,
            no_metadata=args.no_metadata,
            mem_eff_safe_open=args.mem_eff_safe_open,
        )
    elif args.mode == 'quant_svd':
        # Call new function with its specific arguments
        quant_svd(
            model_org=args.model_org,
            save_to=args.save_to,
            dim=args.dim,
            device=args.device,
            save_precision=args.save_precision,
            outlier_quantile=args.outlier_quantile,
            no_metadata=args.no_metadata,
            mem_eff_safe_open=args.mem_eff_safe_open,
        )