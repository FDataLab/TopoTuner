#!/usr/bin/env python3
"""
Extract training information from saved checkpoints and logs.
"""

import os
import torch
import pandas as pd
from pathlib import Path
import re

def extract_training_args(checkpoint_dir):
    """Extract training arguments from saved checkpoint."""
    args_file = os.path.join(checkpoint_dir, "training_args.bin")
    if not os.path.exists(args_file):
        return None
    
    try:
        args = torch.load(args_file, map_location="cpu", weights_only=False)
        return args
    except Exception as e:
        print(f"Error loading {args_file}: {e}")
        return None

def extract_from_log(log_file):
    """Extract training info from log file."""
    if not os.path.exists(log_file):
        return {}
    
    info = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract command line args
            if '[DEBUG] sys.argv:' in content:
                match = re.search(r"\[DEBUG\] sys.argv: \[(.*?)\]", content)
                if match:
                    argv_str = match.group(1)
                    # Parse key arguments
                    if '--batch-size' in argv_str:
                        match = re.search(r"--batch-size['\"]?\s+(\d+)", argv_str)
                        if match:
                            info['batch_size'] = int(match.group(1))
                    if '--epochs' in argv_str:
                        match = re.search(r"--epochs['\"]?\s+(\d+)", argv_str)
                        if match:
                            info['epochs'] = int(match.group(1))
                    if '--learning-rate' in argv_str:
                        match = re.search(r"--learning-rate['\"]?\s+([\d.e-]+)", argv_str)
                        if match:
                            info['learning_rate'] = float(match.group(1))
                    if '--gradient_accumulation_steps' in argv_str:
                        match = re.search(r"--gradient_accumulation_steps['\"]?\s+(\d+)", argv_str)
                        if match:
                            info['gradient_accumulation_steps'] = int(match.group(1))
                    if '--subset-train-size' in argv_str:
                        match = re.search(r"--subset-train-size['\"]?\s+(\d+)", argv_str)
                        if match:
                            info['subset_train_size'] = int(match.group(1))
                    if '--hotpot-evidence' in argv_str:
                        match = re.search(r"--hotpot-evidence['\"]?\s+(\w+)", argv_str)
                        if match:
                            info['hotpot_evidence'] = match.group(1)
    except Exception as e:
        print(f"Error reading log {log_file}: {e}")
    
    return info

def get_max_len_from_code():
    """Get max_len from preprocessing code."""
    # HotpotQA uses max_len=2048 based on finetuning code
    return 2048

def main():
    base_dir = Path("/home/kadir/topo")
    hotpotqa_dir = base_dir / "numpy_weights" / "hotpotqa"
    
    experiments = []
    
    # Check each model directory
    for model_dir in hotpotqa_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Check each experiment type (full, lora, etc.)
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_type = exp_dir.name
            epoch_weights_dir = exp_dir / "epoch_weights"
            
            if not epoch_weights_dir.exists():
                continue
            
            # Get first checkpoint to extract training args
            checkpoints = sorted([d for d in epoch_weights_dir.iterdir() if d.is_dir() and "epoch" in d.name])
            if not checkpoints:
                continue
            
            first_checkpoint = checkpoints[0]
            args = extract_training_args(str(first_checkpoint))
            
            # Try to find log file
            log_patterns = [
                f"finetune_HotpotQA_*{model_name}*{exp_type}*.log",
                f"finetune_HotpotQA_*.log"
            ]
            
            log_info = {}
            for pattern in log_patterns:
                log_files = list((base_dir / "logs").glob(pattern))
                if log_files:
                    log_info = extract_from_log(str(log_files[0]))
                    break
            
            # Get epoch count
            epoch_count = len(checkpoints)
            
            # Compile info
            exp_info = {
                'model': model_name,
                'experiment': exp_type,
                'epochs': epoch_count,
                'max_len': get_max_len_from_code(),
            }
            
            # Add from training args if available
            if args:
                if hasattr(args, 'per_device_train_batch_size'):
                    exp_info['batch_size'] = args.per_device_train_batch_size
                if hasattr(args, 'gradient_accumulation_steps'):
                    exp_info['gradient_accumulation_steps'] = args.gradient_accumulation_steps
                if hasattr(args, 'learning_rate'):
                    exp_info['learning_rate'] = args.learning_rate
                if hasattr(args, 'num_train_epochs'):
                    exp_info['epochs'] = int(args.num_train_epochs)
            
            # Add from log if available
            exp_info.update(log_info)
            
            # Check if LoRA
            exp_info['use_lora'] = exp_type == 'lora'
            
            experiments.append(exp_info)
    
    # Print summary
    print("=" * 100)
    print("HotpotQA Finetuning Experiments Summary")
    print("=" * 100)
    print()
    
    for exp in experiments:
        print(f"Model: {exp['model']}")
        print(f"  Experiment Type: {exp['experiment']}")
        print(f"  LoRA: {exp.get('use_lora', 'Unknown')}")
        print(f"  Epochs: {exp.get('epochs', 'Unknown')}")
        print(f"  Batch Size: {exp.get('batch_size', 'Unknown')}")
        print(f"  Gradient Accumulation: {exp.get('gradient_accumulation_steps', 'Unknown')}")
        print(f"  Learning Rate: {exp.get('learning_rate', 'Unknown')}")
        print(f"  Max Length: {exp.get('max_len', 'Unknown')}")
        print(f"  Training Samples: {exp.get('subset_train_size', 'Unknown')}")
        print(f"  Evidence Mode: {exp.get('hotpot_evidence', 'Unknown')}")
        print()
    
    # Create CSV summary
    df = pd.DataFrame(experiments)
    output_csv = base_dir / "analysis" / "hotpotqa_training_info.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved summary to: {output_csv}")

if __name__ == "__main__":
    main()
