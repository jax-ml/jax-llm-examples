#!/usr/bin/env python
# coding: utf-8

"""
DeepSeek R1 Core Logic

This file contains the core MLIR generation logic for DeepSeek R1 model.
This file should be placed on the implementation machine and will be
downloaded and imported by the generator.
"""

import dataclasses
import json
import math
import os
import sys
import yaml
import glob
import shutil
from typing import Dict

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
try:
    from jax.sharding import use_mesh as set_mesh
except ImportError:
    pass
import numpy as np

# Import the deepseek_r1_jax model
try:
    from deepseek_r1_jax import model as dsjax
except ImportError:
    # Fallback if the module is not available
    dsjax = None

is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)

def load_config(config_file: str) -> Dict:
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        SystemExit: If configuration file is invalid or missing required fields
    """
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' does not exist.")
        sys.exit(1)

    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in '{config_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read configuration file '{config_file}': {e}")
        sys.exit(1)
    return config


def find_xla_dump_file(hlo_dir_path: str) -> str:
    """
    Find the XLA dump file that contains both 'forward' and
    'spmd-partitioner.after_spmd-partitioning.before_pipeline-end.txt' in its name.

    Args:
        hlo_dir_path: Path to the HLO dump directory

    Returns:
        Path to the found file

    Raises:
        FileNotFoundError: If the directory doesn't exist or no matching file is found
    """
    if not os.path.exists(hlo_dir_path):
        raise FileNotFoundError(f"HLO dump directory does not exist: {hlo_dir_path}")

    # Look for .txt files that contain both "forward" and "spmd-partitioner.after_spmd-partitioning.before_pipeline-end.txt"
    pattern = os.path.join(hlo_dir_path, "*.txt")
    txt_files = glob.glob(pattern)

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {hlo_dir_path}")

    print(f"Found {len(txt_files)} .txt files in {os.path.abspath(hlo_dir_path)}")

    # First, try to find exact match with both keywords
    matching_files = []
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        if "forward" in filename and "spmd-partitioner.after_spmd-partitioning.before_pipeline-end.txt" in filename:
            matching_files.append(file_path)
            print(f"Found exact match: {filename}")

    if not matching_files:
        # If no exact match, try to find files with all required components
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            if ("forward" in filename and
                "spmd-partitioner" in filename and
                "after_spmd-partitioning" in filename and
                "before_pipeline-end" in filename):
                matching_files.append(file_path)
                print(f"Found partial match: {filename}")

    if not matching_files:
        # Last resort: look for any file with "forward" and "spmd-partitioner"
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            if "forward" in filename and "spmd-partitioner" in filename:
                matching_files.append(file_path)
                print(f"Found basic match: {filename}")

    if not matching_files:
        available_files = [os.path.basename(f) for f in txt_files]
        raise FileNotFoundError(f"No matching XLA dump file found in {os.path.abspath(hlo_dir_path)}. Available files: {available_files}")

    # Return the most recent matching file
    selected_file = sorted(matching_files)[-1]
    print(f"Selected file: {os.path.basename(selected_file)}")
    return selected_file


def validate_dtype_for_inference(dtype_str: str) -> tuple[bool, str]:
    """
    Validate if a dtype is suitable for inference and provide warnings.

    Args:
        dtype_str: The dtype string to validate

    Returns:
        Tuple of (is_valid, warning_message)
    """
    warnings = []

    # FP4 warnings
    if dtype_str in ["fp4", "f4e2m1", "f4e2m1fn"]:
        warnings.append("FP4 is experimental and may cause numerical instability")
        warnings.append("Consider using FP8 (f8e4m3fn) for production inference")

    # FP8 warnings
    if dtype_str in ["f8e4m3fnuz", "f8e5m2fnuz", "f8e4m3b11fnuz"]:
        warnings.append("NVIDIA-specific FP8 formats may not be supported on all hardware")

    # Very low precision warnings
    if dtype_str in ["fp4", "f4e2m1", "f4e2m1fn"]:
        warnings.append("Very low precision may require careful model tuning")

    is_valid = True
    warning_msg = "; ".join(warnings) if warnings else ""

    return is_valid, warning_msg

def save_mlir(config: Dict, hlo_dir_path: str):
    print("Running save mlir step...")
    impl_config = config['impl_machine_config']
    remote_mlir_output_path = impl_config.get("remote_mlir_output_path")
    remote_mlir_file_name = impl_config.get("remote_mlir_file_name")
    full_remote_mlir_path = os.path.join(remote_mlir_output_path, hlo_dir_path, remote_mlir_file_name)
    os.makedirs(os.path.dirname(full_remote_mlir_path), exist_ok=True)
    xla_dump_file = find_xla_dump_file(hlo_dir_path)
    shutil.copy(xla_dump_file, full_remote_mlir_path)
    print(f"MLIR content saved to: {full_remote_mlir_path}")


def generate_mlir(config: Dict) -> str:
    """
    Generate MLIR content in SDY dialect for DeepSeek R1 model.

    Args:
        config: Dictionary containing configuration parameters for MLIR generation.
               Must include the following fields:
               - dtype: Data type (e.g., "f32", "f16", "bf16", "fp8", "f8e4m3fn", "f8e5m2", etc.)
               - seq_len: Sequence length
               - num_devices: Number of devices for distributed execution
               - batch_size: Batch size
               - num_layers: Number of layers in the model
               - num_experts: Number of experts in the model
               - strategy: String indicating the generation strategy ("prefill" or "decode")
               - model_name: Name of the model
               - quant: Boolean for quantization
               - mesh_shape: Tuple for mesh shape
               - mesh_axes: Tuple for mesh axes
               - input_text: Text input for the model
               - remote_mlir_output_path: Path where to save the MLIR content (optional)

    Returns:
        String containing the generated MLIR content in SDY dialect

    Note:
        FP8 Data Type Recommendations for Inference:
        - "fp8" or "f8e4m3fn": Recommended for general inference (IEEE 754-2008 E4M3)
          - Good balance between precision and range
          - Widely supported on modern hardware
          - Suitable for most transformer operations

        - "f8e5m2": Use when you need larger dynamic range
          - Better for operations with wide value ranges
          - May have slightly lower precision than E4M3

        - "f8e4m3fnuz": NVIDIA format with no underflow
          - Optimized for NVIDIA hardware
          - Good for gradient computations

        - "fp4" or "f4e2m1fn": Experimental, use with caution
          - Maximum compression but limited precision
          - May require careful tuning for stability
    """
    # Extract configuration parameters
    dtype_str = config.get("dtype", "bf16")
    seq_len = config.get("seq_len", 8192)
    num_devices = config.get("num_devices", None)
    batch_size = config.get("batch_size", 16)
    num_layers = config.get("num_layers", 61)
    num_experts = config.get("num_experts", 256)
    num_heads = config.get("num_heads", None)
    per_head_dim = config.get("per_head_dim", None)
    num_experts_per_tok = config.get("num_experts_per_tok", 8)
    per_head_dim = config.get("per_head_dim", None)
    num_groups = config.get("num_groups", 8)
    topk_group = config.get("topk_group", 4)
    routed_scaling_factor = config.get("routed_scaling_factor", 2.5)
    n_shared_experts = config.get("n_shared_experts", 1)
    psum_before_expert_reduce = config.get("psum_before_expert_reduce", False)
    strategy = config.get("strategy", "decode")
    use_decode_ragged_dot_kernel = config.get("use_decode_ragged_dot_kernel", True)
    model_name = config.get("model_name", "deepseek_r1")
    quant = config.get("quant", False)
    strategy = config.get("strategy", "prefill")  # Options: "prefill" or "decode"
    mesh_shape = config.get("mesh_shape", (2, 8, 1))
    mesh_axes = config.get("mesh_axes", ("x", "y", "z"))
    input_text = config.get("input_text", "What is the weather like expressed in long prose in Old English")


    # Calculate device count from mesh_shape
    mesh_device_count = 1
    for dim in mesh_shape:
        mesh_device_count *= dim
    
    # Validate num_devices matches mesh_shape device count
    if num_devices is not None:
        if num_devices != mesh_device_count:
            error_msg = f"# Error: num_devices ({num_devices}) does not match device count calculated from mesh_shape {mesh_shape} ({mesh_device_count})"
            print(error_msg)
            return error_msg
    else:
        # If num_devices is not provided, set it from mesh_shape
        num_devices = mesh_device_count
        print(f"# Info: num_devices not provided, using device count from mesh_shape: {num_devices}")
        # Validate strategy parameter
    valid_strategies = ["prefill", "decode"]
    if strategy not in valid_strategies:
        error_msg = f"# Error: Invalid strategy '{strategy}'. Strategy must be one of: {valid_strategies}"
        print(error_msg)
        return error_msg
    # Check if deepseek_r1_jax is available
    if dsjax is None:
        return f"# Error: deepseek_r1_jax module not available\n# Config: dtype={dtype_str}, seq_len={seq_len}, num_devices={num_devices}, batch_size={batch_size}, num_layers={num_layers}, num_experts={num_experts}"
    try:
        # Convert dtype string to JAX dtype
        dtype_map = {
            # Standard precision formats
            "f32": jnp.float32,
            "f16": jnp.float16,
            "bf16": jnp.bfloat16,

            # FP8 variants (IEEE 754-2008 and NVIDIA formats)
            "fp8": jnp.float8_e4m3fn,  # Default FP8 for inference
            "f8e4m3fn": jnp.float8_e4m3fn,  # IEEE 754-2008 FP8 E4M3
            "f8e5m2": jnp.float8_e5m2,      # IEEE 754-2008 FP8 E5M2
            "f8e4m3b11fnuz": jnp.float8_e4m3b11fnuz,  # NVIDIA FP8 E4M3 with bias 11
            "f8e5m2fnuz": jnp.float8_e5m2fnuz,        # NVIDIA FP8 E5M2 with no underflow
            "f8e4m3fnuz": jnp.float8_e4m3fnuz,        # NVIDIA FP8 E4M3 with no underflow

            # FP4 variants (experimental)
            "fp4": jnp.float4_e2m1fn,  # Default FP4 for inference
            "f4e2m1": jnp.float4_e2m1fn,  # FP4 E2M1 format
            "f4e2m1fn": jnp.float4_e2m1fn,  # FP4 E2M1 format (explicit)
        }
        
        # Validate dtype
        if dtype_str not in dtype_map:
            error_msg = f"# Error: Unsupported dtype '{dtype_str}'. Supported dtypes: {list(dtype_map.keys())}"
            print(error_msg)
            return error_msg
        print(f"# Debug: dtype_map: {dtype_str}")
        # Validate dtype for inference and provide warnings
        is_valid, warning_msg = validate_dtype_for_inference(dtype_str)

        if not is_valid:
            error_msg = f"# Error: Invalid dtype '{dtype_str}' for inference: {warning_msg}"
            print(error_msg)
            return error_msg
        if warning_msg:
            print(f"# Warning: {warning_msg}")
        dtype = dtype_map[dtype_str]
        
        # Set up environment variables based on strategy
        if strategy == "decode":
            inf_stage = "dec"
        else:  # strategy == "prefill" (default)
            inf_stage = "pre"

        # Extract hlo_dir_path from config if available, otherwise construct it
        hlo_dir_path = config.get('hlo_dir_path')
        if not hlo_dir_path:
            raise ValueError("hlo_dir_path is required in config")
        os.makedirs(hlo_dir_path, exist_ok=True)
        os.environ['JAX_DUMP_IR_TO'] = hlo_dir_path
        # 
        os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices} --xla_dump_hlo_pass_re=spmd* --xla_dump_hlo_as_text --xla_dump_to={hlo_dir_path}'
        jax.config.update("jax_use_shardy_partitioner", True)

        # Set JAX configuration to handle FP8 more gracefully
        if dtype_str in ["fp8", "f8e4m3fn", "f8e5m2", "f8e4m3fnuz", "f8e5m2fnuz", "f8e4m3b11fnuz"]:
            print(f"# Debug: Configuring JAX for FP8 operations with {dtype_str}")
            # Enable FP8 support in JAX
            jax.config.update("jax_enable_x64", False)  # Disable x64 for better FP8 compatibility
        # REsponsible for adding source line number
        #jax.config.update("jax_cache_compilation_metadata", True)
        # Load tokenizer
        tokenizer = dsjax.load_tokenizer()

        # Create mesh
        mesh = jax.make_mesh(
            mesh_shape, mesh_axes, devices=jax.devices(), axis_types=(AxisType.Auto,) * len(mesh_shape)
        )
  
        cfg = dataclasses.replace(dsjax.Config(), mesh=mesh)
        # Update configuration
        cfg = dataclasses.replace(cfg, num_layers=num_layers)
        cfg = dataclasses.replace(cfg, n_routed_experts=num_experts)
        cfg = dataclasses.replace(cfg, num_experts_per_tok=num_experts_per_tok)
        if num_heads is None or per_head_dim is None:
            raise ValueError("num_heads and per_head_dim must be provided in config")
        cfg = dataclasses.replace(cfg, num_heads=num_heads)
        embed = per_head_dim * num_heads
        cfg = dataclasses.replace(cfg, embed=embed)
        # Set quantization flags and scale dtype
        # quant_scale_dtype should be a dtype (e.g., bfloat16), not a boolean
        quant_scale_dtype = dtype  # Use the same dtype as the model for quantization scales
        cfg = dataclasses.replace(cfg, quantize_attn=quant, quantize_cache=quant, quantize_moe=quant, quant_scale_dtype=quant_scale_dtype)
        cfg = dataclasses.replace(cfg, strategy=strategy)
        cfg = dataclasses.replace(cfg, max_seq_len=seq_len)
        # Setting the below flag to True triggers error:interpret mode is only supported on CPU backend.
        cfg = dataclasses.replace(cfg, use_decode_ragged_dot_kernel=False)
        cfg = dataclasses.replace(cfg, moe_gate_dtype=dtype)

        # Set dtype in config (Config class has dtype field, but not compute_dtype or param_dtype)
        cfg = dataclasses.replace(cfg, dtype=dtype)
        # Create weights and cache
        wts_abs = dsjax.Weights.abstract(cfg)
        wts_shd = dsjax.Weights.shardings(cfg)
        kv_shrd = dsjax.KVCache.shardings(cfg, batch_size, cfg.max_seq_len)
        kv_abst = dsjax.KVCache.abstract(cfg, batch_size, cfg.max_seq_len)

        # Simplified conversion using JAX tree utilities
        # Since ArrayInfo is registered as a pytree, we can use jax.tree.map to automatically traverse
        def convert_arrayinfo_to_shapedtypestruct(ai, shrd):
            """Convert ArrayInfo to ShapeDtypeStruct with the target dtype."""
            if isinstance(ai, dsjax.ArrayInfo):
                return jax.ShapeDtypeStruct(ai.shape, dtype, sharding=shrd)
            return ai  # Already converted or not an ArrayInfo
        
        # Use JAX tree.map to automatically traverse and convert all ArrayInfo instances
        # This is much simpler and more efficient than manual recursion
        is_param = lambda x: isinstance(x, dsjax.ArrayInfo)
        wts_abs = jax.tree.map(convert_arrayinfo_to_shapedtypestruct, wts_abs, wts_shd, is_leaf=is_param)
        kv_abst = jax.tree.map(convert_arrayinfo_to_shapedtypestruct, kv_abst, kv_shrd, is_leaf=is_param)

        # Debug: Check if weights and cache are using the correct dtype
        print(f"# Debug: Weights and cache configured with dtype: {dtype}")
        # Note: No need for update_dtypes_recursive - dtype is already set correctly
        # during the ArrayInfo -> ShapeDtypeStruct conversion above

        # Encode input
        def encode_input(tokenizer, texts, pad_id: int = 0):
            assert isinstance(texts, list)
            inputs = [
                tokenizer.apply_chat_template([{"role": "user", "content": text}]) + tokenizer.encode("<|Assistant|><think>")
                for text in texts
            ]
            max_len = max([len(x) for x in inputs])
            inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
            return np.array(inputs)

        input_tokens = encode_input(tokenizer, [input_text])

        # Print debug information about dtype configuration
        print(f"# Debug: Using dtype {dtype} ({dtype_str}) for model configuration")
        
        # Prepare input
        cur_seq_len = input_tokens.shape[1]
        seq_tile_factor = cfg.max_seq_len // cur_seq_len
        input_tokens = jnp.tile(input_tokens, [batch_size, seq_tile_factor])

        # Initialize cache
        def init_zero_cache(cfg):
            # Use KVCache.abstract() to get the correct structure (handles quantization)
            cache_abstract = dsjax.KVCache.abstract(cfg, batch_size, cfg.max_seq_len, dtype)
            
            # Convert ArrayInfo/QuantArray to ShapeDtypeStruct
            def convert_cache_leaf(obj, shrd):
                """Convert ArrayInfo or QuantArray to ShapeDtypeStruct structure."""
                # Handle QuantArray (when quantize_cache=True)
                if is_type(obj, dsjax.QuantArray):
                    # Extract shardings for quant and scale
                    if is_type(shrd, dsjax.QuantArray):
                        quant_shrd = shrd.quant
                        scale_shrd = shrd.scale
                    else:
                        # Fallback: use same sharding for both
                        quant_shrd = shrd
                        scale_shrd = shrd
                    
                    # Convert quant field: should be int8
                    if isinstance(obj.quant, dsjax.ArrayInfo):
                        quant_sds = jax.ShapeDtypeStruct(obj.quant.shape, jnp.int8, sharding=quant_shrd)
                    else:
                        quant_sds = obj.quant  # Already converted
                    
                    # Convert scale field: should use quant_scale_dtype
                    scale_dtype = cfg.quant_scale_dtype if cfg.quantize_cache else dtype
                    if isinstance(obj.scale, dsjax.ArrayInfo):
                        scale_sds = jax.ShapeDtypeStruct(obj.scale.shape, scale_dtype, sharding=scale_shrd)
                    else:
                        scale_sds = obj.scale  # Already converted
                    
                    # Return QuantArray with ShapeDtypeStruct fields, preserving metadata
                    return dsjax.QuantArray(
                        quant=quant_sds,
                        scale=scale_sds,
                        out_scaling=obj.out_scaling,
                        scale_expand_dims=obj.scale_expand_dims
                    )
                
                # Handle ArrayInfo (regular case)
                if isinstance(obj, dsjax.ArrayInfo):
                    # Preserve the original dtype from ArrayInfo (important for iter, starts which are int32)
                    return jax.ShapeDtypeStruct(obj.shape, obj.dtype, sharding=shrd)
                
                return obj  # Already converted or not a leaf we handle
            
            # Get shardings for the cache
            cache_shardings = dsjax.KVCache.shardings(cfg, batch_size, cfg.max_seq_len)
            
            # Convert using tree.map with proper leaf detection
            is_leaf = lambda x: isinstance(x, dsjax.ArrayInfo) or is_type(x, dsjax.QuantArray)
            cache = jax.tree.map(convert_cache_leaf, cache_abstract, cache_shardings, is_leaf=is_leaf)
            
            return cache

        # Generate MLIR
        tokens = input_tokens
        pad_id = 0

        with set_mesh(cfg.mesh):
            assert tokens.shape[-1] <= cfg.max_seq_len
            pad_to = 2 ** math.ceil(math.log2((tokens.shape[-1])))

            #prompt, prompt_segment_ids = dsjax.prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)
            prompt_shape = (batch_size, cfg.max_seq_len)
            prompt = jax.ShapeDtypeStruct(prompt_shape, jnp.int32, sharding=None)
            prompt_segment_ids = jax.ShapeDtypeStruct(prompt_shape, jnp.int32, sharding=None)
            assert prompt.ndim == 2

            zero_cache = init_zero_cache(cfg)
            cache = zero_cache

            cache_shardings = dsjax.KVCache.shardings(cfg, prompt.shape[0], cfg.max_seq_len)
            if is_type(cache, dsjax.KVCache):
                # Create uninitialized_iter with explicit int32 dtype (not using ones_like which might use wrong dtype)
                uninitialized_iter = jax.ShapeDtypeStruct((), jnp.int32, sharding=None)
                cache = dataclasses.replace(cache, starts=jax.ShapeDtypeStruct((batch_size,), jnp.int32, sharding=None), iter=uninitialized_iter)
            else:
                cache_shardings = tuple([z[idx] for idx in range(cfg.num_layers)] for z in cache_shardings)
            # cache_shardings should only contain Sharding objects
            # when quant=True, cache_shardings contains QuantArray(quant=Sharding, scale=Sharding)
            if cfg.quantize_cache:
                cache_shardings = None
            logits_shardings = dsjax.logical_to_sharding(("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules)
            fwd_jit = jax.jit(dsjax.forward, donate_argnums=(4,), out_shardings=(logits_shardings, cache_shardings))

            if strategy != "decode":
                try:
                    print(f"Prompt Shape: {type(prompt)}, {prompt.shape}, {prompt_segment_ids.shape}")
                    print(f"num_experts: {num_experts}, mesh: {mesh_shape}")
                    prefill_fwd_lowered = fwd_jit.lower(
                        prompt, prompt_segment_ids, wts_abs, cfg, cache
                    )
                    # Allow compilation to finish and generate XLA dumps
                    prefill_fwd = prefill_fwd_lowered.compile().as_text()
                    print(f"Prefill MLIR content length: {len(prefill_fwd)}")
                    save_mlir(config, hlo_dir_path)
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error in prefill mode. Unable to lower the jit function: {error_msg}")
            else:
                avg_seq_len = tokens.shape[1]
                cache.length = avg_seq_len
                #last_tokens = prompt[:, avg_seq_len - 1 : avg_seq_len]
                last_tokens = jax.ShapeDtypeStruct((batch_size, 1),jnp.int32,sharding=None)
                print("Decode input", last_tokens.shape)
                segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
                try:
                    print(f"Decode:{prompt.shape}\t{last_tokens.shape}\t{cache.starts}")
                    print(f"num_experts: {num_experts}, mesh: {mesh_shape}")
                    decode_fwd_lowered = fwd_jit.lower(last_tokens, segment_ids, wts_abs, cfg, cache)
                    decode_fwd = decode_fwd_lowered.compile().as_text()
                    print(f"Decode MLIR content length: {len(decode_fwd)}")
                    save_mlir(config, hlo_dir_path)
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error in decode mode. Unable to lower the jit function: {error_msg}")
    except Exception as e:
        error_result = f"# Error generating DeepSeek R1 MLIR: {str(e)}\n# Config: dtype={dtype_str}, seq_len={seq_len}, num_devices={num_devices}, batch_size={batch_size}, num_layers={num_layers}, num_experts={num_experts}, strategy={strategy}"

        return error_result


# Example usage when run directly
if __name__ == "__main__":
    # Check if config file is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python deepseek_core_logic.py <config_file>")
        print("Example: python deepseek_core_logic.py deepseek_config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]

    # Load configuration from YAML file
    config = load_config(config_file)

    # Generate MLIR content
    result = generate_mlir(config)
    #print("Generated MLIR content in otter:", len(result))
    #print(result)
