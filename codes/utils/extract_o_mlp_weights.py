#!/usr/bin/env python3
"""
Extract attention projection and MLP weights from safetensors checkpoints as numpy arrays.

Usage:
    # Extract all projections (Q/K/V/O + MLP):
    python codes/utils/extract_o_mlp_weights.py \
        --checkpoint-dir <path-to-safetensors> \
        --output-dir <output-numpy-dir> \
        --all

    # Extract only O + MLP (original behavior):
    python codes/utils/extract_o_mlp_weights.py \
        --checkpoint-dir <path-to-safetensors> \
        --output-dir <output-numpy-dir>
"""

import os
import argparse
import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path
import re


def _save_tensor(tensor, output_path, label=None):
    """Save a safetensors tensor to .npy, handling bfloat16 safely."""
    arr = tensor.to(torch.float16).numpy()
    np.save(output_path, arr)
    if label:
        print(f"  ✓ Saved {os.path.basename(output_path)} (shape: {tuple(tensor.shape)})")


def extract_weights(checkpoint_dir: str, output_dir: str, include_qkv=False, include_mlp=True):
    """
    Extract projection weights from safetensors files.

    When include_qkv=True, also extracts Q/K/V projections (layer{N}_q/k/v.npy).
    Always extracts O projection (layer{N}_o.npy).
    When include_mlp=True, also extracts MLP projections.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    safetensor_files = [f for f in safetensor_files if not f.name.endswith('.index.json')]

    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {checkpoint_dir}")

    print(f"Found {len(safetensor_files)} safetensors files")

    extracted_count = 0

    for safetensor_file in safetensor_files:
        print(f"\nProcessing {safetensor_file.name}...")

        with safe_open(str(safetensor_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())

            for key in keys:
                # Q/K/V projections
                if include_qkv:
                    qkv_match = re.match(r"model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight", key)
                    if qkv_match:
                        layer_idx = int(qkv_match.group(1))
                        proj = qkv_match.group(2)
                        tensor = f.get_tensor(key)
                        label = f"layer{layer_idx}_{proj}.npy"
                        _save_tensor(tensor, output_dir / label, label if layer_idx < 3 else None)
                        extracted_count += 1

                # O projection (always)
                o_match = re.match(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight", key)
                if o_match:
                    layer_idx = int(o_match.group(1))
                    tensor = f.get_tensor(key)
                    label = f"layer{layer_idx}_o.npy"
                    _save_tensor(tensor, output_dir / label, label if layer_idx < 3 else None)
                    extracted_count += 1

                # MLP projections
                if include_mlp:
                    mlp_match = re.match(r"model\.layers\.(\d+)\.mlp\.(down|gate|up)_proj\.weight", key)
                    if mlp_match:
                        layer_idx = int(mlp_match.group(1))
                        proj_type = mlp_match.group(2)
                        tensor = f.get_tensor(key)
                        label = f"layer{layer_idx}_mlp_{proj_type}.npy"
                        _save_tensor(tensor, output_dir / label, label if layer_idx < 3 else None)
                        extracted_count += 1

    print(f"\n✅ Total extracted: {extracted_count} weight matrices")
    return extracted_count


def extract_lora_o_mlp_weights(checkpoint_dir: str, output_dir: str):
    """Extract O projection and MLP LoRA adapters from safetensors files."""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_file = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"No adapter_model.safetensors found in {checkpoint_dir}")
        return 0

    print(f"Processing LoRA adapters from {adapter_file.name}...")
    extracted_count = 0

    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            o_match = re.match(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.o_proj\.lora_(A|B)\.default\.weight", key)
            if o_match:
                layer_idx = int(o_match.group(1))
                lora_type = o_match.group(2)
                tensor = f.get_tensor(key)
                _save_tensor(tensor, output_dir / f"layer{layer_idx}_o_{lora_type}.npy",
                             f"layer{layer_idx}_o_{lora_type}.npy" if layer_idx < 3 else None)
                extracted_count += 1

            mlp_match = re.match(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(down|gate|up)_proj\.lora_(A|B)\.default\.weight", key)
            if mlp_match:
                layer_idx = int(mlp_match.group(1))
                proj_type = mlp_match.group(2)
                lora_type = mlp_match.group(3)
                tensor = f.get_tensor(key)
                _save_tensor(tensor, output_dir / f"layer{layer_idx}_mlp_{proj_type}_{lora_type}.npy",
                             f"layer{layer_idx}_mlp_{proj_type}_{lora_type}.npy" if layer_idx < 3 else None)
                extracted_count += 1

    print(f"\n✅ Total extracted: {extracted_count} LoRA adapters")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention/MLP weights from safetensors checkpoints"
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory containing safetensors checkpoint files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for numpy arrays")
    parser.add_argument("--lora", action="store_true",
                        help="Extract LoRA adapters instead of full weights")
    parser.add_argument("--all", action="store_true",
                        help="Extract all projections: Q/K/V/O + MLP (default: O + MLP only)")
    parser.add_argument("--no-mlp", action="store_true",
                        help="Skip MLP projections (useful when only attention weights are needed)")

    args = parser.parse_args()

    if args.lora:
        extract_lora_o_mlp_weights(args.checkpoint_dir, args.output_dir)
    else:
        extract_weights(args.checkpoint_dir, args.output_dir,
                        include_qkv=args.all, include_mlp=not args.no_mlp)


if __name__ == "__main__":
    main()
