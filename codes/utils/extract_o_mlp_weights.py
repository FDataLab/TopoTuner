#!/usr/bin/env python3
"""
Extract O projection and MLP weights from safetensors checkpoints and save as numpy arrays.
This allows us to analyze weight changes for O and MLP matrices similar to K/Q/V.

Usage:
    python codes/utils/extract_o_mlp_weights.py \
        --checkpoint-dir /home/kadir/topo/numpy_weights/imdb/llama32_3b/full/epoch_weights/checkpoint-epoch-6 \
        --output-dir /home/kadir/topo/numpy_weights/imdb/llama32_3b/full/epoch_weights/checkpoint-epoch-6/numpy_weights
"""

import os
import argparse
import numpy as np
from safetensors import safe_open
from pathlib import Path
import re


def extract_o_mlp_weights(checkpoint_dir: str, output_dir: str):
    """
    Extract O projection and MLP weights from safetensors files.
    
    Saves:
        - layer{N}_o.npy: O projection weights
        - layer{N}_mlp_down.npy: MLP down projection
        - layer{N}_mlp_gate.npy: MLP gate projection
        - layer{N}_mlp_up.npy: MLP up projection
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all safetensors files
    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    safetensor_files = [f for f in safetensor_files if not f.name.endswith('.index.json')]
    
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {checkpoint_dir}")
    
    print(f"Found {len(safetensor_files)} safetensors files")
    
    extracted_count = 0
    
    # Process each safetensors file
    for safetensor_file in safetensor_files:
        print(f"\nProcessing {safetensor_file.name}...")
        
        with safe_open(str(safetensor_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                # Match O projection: model.layers.{N}.self_attn.o_proj.weight
                o_match = re.match(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight", key)
                if o_match:
                    layer_idx = int(o_match.group(1))
                    tensor = f.get_tensor(key)
                    output_path = output_dir / f"layer{layer_idx}_o.npy"
                    np.save(output_path, tensor.numpy().astype(np.float16))
                    extracted_count += 1
                    if layer_idx < 3:  # Only print first few
                        print(f"  ✓ Saved layer{layer_idx}_o.npy (shape: {tensor.shape})")
                
                # Match MLP projections: model.layers.{N}.mlp.{type}_proj.weight
                mlp_match = re.match(r"model\.layers\.(\d+)\.mlp\.(down|gate|up)_proj\.weight", key)
                if mlp_match:
                    layer_idx = int(mlp_match.group(1))
                    proj_type = mlp_match.group(2)
                    tensor = f.get_tensor(key)
                    output_path = output_dir / f"layer{layer_idx}_mlp_{proj_type}.npy"
                    np.save(output_path, tensor.numpy().astype(np.float16))
                    extracted_count += 1
                    if layer_idx < 3:  # Only print first few
                        print(f"  ✓ Saved layer{layer_idx}_mlp_{proj_type}.npy (shape: {tensor.shape})")
    
    print(f"\n✅ Total extracted: {extracted_count} weight matrices")
    return extracted_count


def extract_lora_o_mlp_weights(checkpoint_dir: str, output_dir: str):
    """
    Extract O projection and MLP LoRA adapters from safetensors files.
    
    Saves:
        - layer{N}_o_A.npy, layer{N}_o_B.npy: O projection LoRA adapters
        - layer{N}_mlp_{type}_A.npy, layer{N}_mlp_{type}_B.npy: MLP LoRA adapters
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for adapter_model.safetensors (LoRA)
    adapter_file = checkpoint_dir / "adapter_model.safetensors"
    
    if not adapter_file.exists():
        print(f"No adapter_model.safetensors found in {checkpoint_dir}")
        return 0
    
    print(f"Processing LoRA adapters from {adapter_file.name}...")
    
    extracted_count = 0
    
    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        for key in keys:
            # Match O projection LoRA: base_model.model.model.layers.{N}.self_attn.o_proj.lora_{A|B}.default.weight
            o_match = re.match(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.o_proj\.lora_(A|B)\.default\.weight", key)
            if o_match:
                layer_idx = int(o_match.group(1))
                lora_type = o_match.group(2)
                tensor = f.get_tensor(key)
                output_path = output_dir / f"layer{layer_idx}_o_{lora_type}.npy"
                np.save(output_path, tensor.numpy().astype(np.float16))
                extracted_count += 1
                if layer_idx < 3:
                    print(f"  ✓ Saved layer{layer_idx}_o_{lora_type}.npy (shape: {tensor.shape})")
            
            # Match MLP LoRA: base_model.model.model.layers.{N}.mlp.{type}_proj.lora_{A|B}.default.weight
            mlp_match = re.match(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(down|gate|up)_proj\.lora_(A|B)\.default\.weight", key)
            if mlp_match:
                layer_idx = int(mlp_match.group(1))
                proj_type = mlp_match.group(2)
                lora_type = mlp_match.group(3)
                tensor = f.get_tensor(key)
                output_path = output_dir / f"layer{layer_idx}_mlp_{proj_type}_{lora_type}.npy"
                np.save(output_path, tensor.numpy().astype(np.float16))
                extracted_count += 1
                if layer_idx < 3:
                    print(f"  ✓ Saved layer{layer_idx}_mlp_{proj_type}_{lora_type}.npy (shape: {tensor.shape})")
    
    print(f"\n✅ Total extracted: {extracted_count} LoRA adapters")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract O and MLP weights from safetensors checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing safetensors checkpoint files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for numpy arrays"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Extract LoRA adapters instead of full weights"
    )
    
    args = parser.parse_args()
    
    if args.lora:
        extract_lora_o_mlp_weights(args.checkpoint_dir, args.output_dir)
    else:
        extract_o_mlp_weights(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
