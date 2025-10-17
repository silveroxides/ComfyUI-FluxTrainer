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


#  or "final_layer" in key or "txt_in" in key or "img_in" in key or "distilled_guidance_layer" in key or "cross_attn" in key or "ffn" in key or "self_attn" in key or "norm3" in key or "head.head" in key or "text_embedding" in key or "time_embedding" in key or "time_projection" in key
# ===============================================================================
# START: REFACTORED AND IMPROVED CODE FOR QUANTIZATION
# ===============================================================================

# --- Configuration ---
COMPUTE_DTYPE = torch.float32  # Use float32 for stable delta calculations
SCALE_DTYPE = torch.float32

# --- Helper functions for FP8 quantization ---

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    if fp8_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")

    finfo = torch.finfo(fp8_dtype)
    # E4M3FN has a different tiny value in practice due to subnormals
    fp8_min_pos = 2**-9 if fp8_dtype == torch.float8_e4m3fn else finfo.tiny
    return float(finfo.min), float(finfo.max), float(fp8_min_pos)

def manual_stochastic_round_to_float8(x: torch.Tensor, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """Performs stochastic rounding for a given FP8 dtype."""
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype for manual stochastic rounding")

    # Use bfloat16 for intermediate calculations to prevent overflow/underflow issues with fp16
    x_bf16 = x.to(torch.bfloat16)
    sign = torch.sign(x_bf16)
    abs_x = x_bf16.abs()
    sign = torch.where(abs_x == 0, torch.tensor(0.0, device=x.device, dtype=x.dtype), sign)

    exponent = torch.clamp(torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS, 0, 2**EXPONENT_BITS - 1)
    normal_mask = ~(exponent == 0)

    # Calculate scaled mantissa
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    # Stochastic rounding step
    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    rounded_mantissa = mantissa_scaled.floor() / (2**MANTISSA_BITS)

    # Reconstruct the number
    reconstructed_abs = torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + rounded_mantissa),
        (2.0 ** (-EXPONENT_BIAS + 1)) * rounded_mantissa
    )

    reconstructed_val = sign * reconstructed_abs
    finfo = torch.finfo(dtype)
    return torch.clamp(reconstructed_val, min=finfo.min, max=finfo.max).to(dtype)


def quantize_dequantize_tensor(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    scaling_mode: str,
    seed: int
) -> torch.Tensor:
    """
    Quantizes a tensor to a specified FP8 format and then de-quantizes it back.

    Args:
        tensor (torch.Tensor): The input tensor to process.
        fp8_dtype (torch.dtype): The target FP8 dtype (e.g., torch.float8_e4m3fn).
        scaling_mode (str): 'tensor' for per-tensor or 'vector' for per-row scaling.
        seed (int): Random seed for stochastic rounding.

    Returns:
        torch.Tensor: The de-quantized tensor in the original compute dtype.
    """
    FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(fp8_dtype)
    original_device = tensor.device
    compute_tensor = tensor.to(COMPUTE_DTYPE)

    # 1. Calculate scale factor
    if scaling_mode == 'vector':
        # Per-row (vector) scaling
        abs_max = compute_tensor.abs().max(dim=1, keepdim=True)[0]
    else: # 'tensor'
        # Per-tensor scaling
        abs_max = compute_tensor.abs().max()

    # Avoid division by zero for empty/zero tensors
    abs_max.clamp_min_(FP8_MIN_POS)
    scale_factor = FP8_MAX / abs_max

    # 2. Scale, clamp, and quantize with stochastic rounding
    scaled_tensor = compute_tensor.mul(scale_factor)
    clamped_tensor = torch.clamp(scaled_tensor, FP8_MIN, FP8_MAX)

    # Stochastic rounding
    generator = torch.Generator(device=original_device).manual_seed(seed)
    # The manual function expects a high-precision tensor and returns an FP8 one
    quantized_fp8_tensor = manual_stochastic_round_to_float8(clamped_tensor, fp8_dtype, generator)

    # 3. De-quantize back to high precision
    dequant_scale = scale_factor.reciprocal()
    dequantized_tensor = quantized_fp8_tensor.to(COMPUTE_DTYPE) * dequant_scale

    return dequantized_tensor.to(original_device)


def quant_svd(
    model_org=None,
    save_to=None,
    dim=4,
    device=None,
    store_device='cpu',
    save_precision=None,
    outlier_quantile=0.99,
    scaling_mode='tensor',
    fp8_dtype_str='e4m3fn',
    seed=0,
    no_metadata=False,
    mem_eff_safe_open=False,
):
    """
    Creates a LoRA by extracting the outlier quantization errors of a model.
    This function quantizes a model to FP8, calculates the difference (error)
    between the original and the de-quantized version, isolates the largest
    errors (outliers), and performs SVD on them to create a LoRA that
    compensates for the quantization precision loss.
    """
    def str_to_dtype(p):
        if p == "float": return torch.float
        if p == "fp16": return torch.float16
        if p == "bf16": return torch.bfloat16
        return None

    save_dtype = str_to_dtype(save_precision)
    fp8_dtype = torch.float8_e4m3fn if fp8_dtype_str == 'e4m3fn' else torch.float8_e5m2
    logger.info(f"Starting quant_svd with rank={dim}, scaling_mode='{scaling_mode}', fp8_dtype='{fp8_dtype_str}', outlier_quantile={outlier_quantile}")


    if not mem_eff_safe_open:
        open_fn = lambda fn: safe_open(fn, framework="pt")
    else:
        logger.info("Using memory efficient safe_open")
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)

    lora_weights = {}
    with open_fn(model_org) as fo:
        keys = []
        for key in fo.keys():
            if not ("single_block" in key or "double_block" in key or "blocks" in key):
                continue
            if ".bias" in key or "norm" in key:
                continue
            if not key.endswith(".weight"):
                continue
            keys.append(key)

        comfy_pbar = ProgressBar(len(keys))
        for key in tqdm(keys, desc="Processing Quantization Deltas"):
            # 1. Get original tensor
            original_tensor = fo.get_tensor(key).to(device=device, dtype=COMPUTE_DTYPE)
            if original_tensor.numel() == 0 or original_tensor.ndim != 2:
                continue

            # 2. Quantize and de-quantize the tensor using the new modular function
            dequantized_tensor = quantize_dequantize_tensor(
                original_tensor,
                fp8_dtype=fp8_dtype,
                scaling_mode=scaling_mode,
                seed=seed
            )

            # 3. Calculate the quantization error delta
            delta = original_tensor - dequantized_tensor

            # 4. Isolate outliers (the opposite of clamping)
            if 0.0 < outlier_quantile < 1.0:
                abs_delta = delta.abs()
                flat_delta = abs_delta.flatten()
                num_elements = flat_delta.numel()

                # Use torch.kthvalue for memory efficiency, which is much better than quantile.
                # Calculate the index 'k' corresponding to the quantile.
                # Ensure k is a valid index (between 1 and num_elements).
                k = min(max(1, int(outlier_quantile * num_elements)), num_elements)

                # Find the k-th smallest value, which is our quantile threshold.
                threshold = torch.kthvalue(flat_delta, k).values

                mask = abs_delta >= threshold
                outlier_delta = delta * mask
            else:
                outlier_delta = delta # No outlier isolation if quantile is 1.0 or 0.0

            # 5. Perform SVD on the outlier delta matrix
            out_dim, in_dim = outlier_delta.shape
            rank = min(dim, in_dim, out_dim)

            # SVD works on 2D matrices
            mat = outlier_delta.squeeze()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

            # Reconstruct U with the singular values
            U = U @ torch.diag(S)

            U = U.to(store_device, dtype=save_dtype).contiguous()
            Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

            logger.info(f"Processed key: {key}, U: {U.size()}, Vh: {Vh.size()}")
            comfy_pbar.update(1)
            lora_weights[key] = (U, Vh)
            del mat, U, S, Vh, delta, outlier_delta, original_tensor, dequantized_tensor

    # 6. Make state dict for LoRA (same logic as original svd function)
    lora_sd = {}
    for key, (up_weight, down_weight) in lora_weights.items():
        lora_name = key.replace(".weight", "").replace(".", "_")
        lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + lora_name
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(float(dim)) # Use float(dim) for alpha

    net_kwargs = {"scaling_mode": scaling_mode, "fp8_dtype": fp8_dtype_str, "seed": seed}
    metadata = {
        "ss_v2": str(False),
        "ss_base_model_version": flux_utils.MODEL_VERSION_FLUX_V1,
        "ss_network_module": "networks.lora_flux",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
        "ss_comment": f"LoRA from quantization errors. Quantile={outlier_quantile}, Rank={dim}, Scaling={scaling_mode}"
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(lora_sd, False, False, False, True, False, time.time(), title, flux="dev")
        metadata.update(sai_metadata)

    save_to_file(save_to, lora_sd, metadata, save_dtype)
    logger.info(f"Quantization Error LoRA weights saved to {save_to}")
    return save_to


# ===============================================================================
# END: REFACTORED CODE
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
            if not ("single_block" in key or "double_block" in key or "blocks" in key):
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
    # SVD mode specific arg
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="For 'svd' mode: Quantile clamping value, float, (0-1). Default = 0.99",
    )
    # QUANT_SVD mode specific args
    parser.add_argument(
        "--outlier_quantile",
        type=float,
        default=0.99,
        help="For 'quant_svd': keep only error values above this quantile (0-1). E.g., 0.99 keeps top 1%% of errors.",
    )
    parser.add_argument(
        "--scaling_mode",
        type=str,
        default="tensor",
        choices=["vector", "tensor"],
        help="For 'quant_svd': Quantization scaling mode. 'vector' for row-wise, 'tensor' for per-tensor.",
    )
    parser.add_argument(
        "--fp8_dtype",
        type=str,
        default="e4m3fn",
        choices=["e4m3fn", "e5m2"],
        help="For 'quant_svd': The FP8 data type to use for the quantization simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="For 'quant_svd': Seed for stochastic rounding for reproducible results.",
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
            scaling_mode=args.scaling_mode,
            fp8_dtype_str=args.fp8_dtype,
            seed=args.seed,
            no_metadata=args.no_metadata,
            mem_eff_safe_open=args.mem_eff_safe_open,
        )