#!/usr/bin/env python3
"""
Batch extract O and MLP weights for multiple experiments.
Processes epoch 0 and epoch 6 for all specified dataset/model/type combinations.
"""

import os
import subprocess
from pathlib import Path

# Target combinations
DATASETS = ["mmlu", "imdb", "sst2"]
MODELS = ["llama32_3b", "llama31_8b", "qwen_8b_base"]
TYPES = ["full", "lora"]
EPOCHS = [0, 6]  # We need epoch 0 (baseline) and epoch 6 (final)

BASE_DIR = Path("/home/kadir/topo/numpy_weights")
SCRIPT_PATH = Path("/home/kadir/topo/codes/utils/extract_o_mlp_weights.py")


def extract_for_experiment(dataset, model, exp_type, epoch):
    """Extract O and MLP weights for a specific experiment and epoch."""
    
    checkpoint_dir = BASE_DIR / dataset / model / exp_type / "epoch_weights" / f"checkpoint-epoch-{epoch}"
    output_dir = checkpoint_dir / "numpy_weights"
    
    if not checkpoint_dir.exists():
        print(f"  ⚠️  Checkpoint not found: {checkpoint_dir}")
        return False
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted (check for layer0_o.npy or layer0_o_A.npy)
    if exp_type == "lora":
        check_file = output_dir / "layer0_o_A.npy"
    else:
        check_file = output_dir / "layer0_o.npy"
    
    if check_file.exists():
        print(f"  ✓ Already extracted: {dataset}/{model}/{exp_type}/epoch-{epoch}")
        return True
    
    # Run extraction
    cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--checkpoint-dir", str(checkpoint_dir),
        "--output-dir", str(output_dir)
    ]
    
    if exp_type == "lora":
        cmd.append("--lora")
    
    print(f"  🔄 Extracting: {dataset}/{model}/{exp_type}/epoch-{epoch}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  ✅ Success: {dataset}/{model}/{exp_type}/epoch-{epoch}")
            return True
        else:
            print(f"  ❌ Failed: {dataset}/{model}/{exp_type}/epoch-{epoch}")
            print(f"     Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout: {dataset}/{model}/{exp_type}/epoch-{epoch}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {dataset}/{model}/{exp_type}/epoch-{epoch} - {e}")
        return False


def main():
    print("=" * 80)
    print("BATCH EXTRACTION OF O AND MLP WEIGHTS")
    print("=" * 80)
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Types: {TYPES}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 80)
    
    total = len(DATASETS) * len(MODELS) * len(TYPES) * len(EPOCHS)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset}")
        print(f"{'='*80}")
        
        for model in MODELS:
            print(f"\n  MODEL: {model}")
            print(f"  {'-'*76}")
            
            for exp_type in TYPES:
                print(f"\n    TYPE: {exp_type}")
                
                for epoch in EPOCHS:
                    result = extract_for_experiment(dataset, model, exp_type, epoch)
                    if result:
                        success_count += 1
                    elif result is False:
                        failed_count += 1
                    else:
                        skipped_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {total}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
