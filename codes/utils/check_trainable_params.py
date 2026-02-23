#!/usr/bin/env python3
"""
Check trainable vs frozen parameters for different freezing configurations.
This helps understand why freezing 15 layers still gives similar accuracy.
"""

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import os

# Set token
HF_TOKEN = os.getenv("HF_TOKEN", "hf_YourTokenHere")

def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def analyze_freezing(model_id, freeze_q_layers, freeze_k_layers, freeze_v_layers):
    """Analyze parameter counts with specific freezing configuration."""
    print(f"\n{'='*80}")
    print(f"Analyzing: Q={freeze_q_layers}, K={freeze_k_layers}, V={freeze_v_layers}")
    print(f"{'='*80}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    
    # Freeze Q/K/V projections
    qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
    frozen_count = {"q": 0, "k": 0, "v": 0, "other": 0}
    
    for name, p in model.named_parameters():
        hit_layer = None
        for i in (qset | kset | vset):
            if f".layers.{i}." in name:
                hit_layer = i
                break
        
        if hit_layer is not None:
            if hit_layer in qset and ".q_proj." in name:
                p.requires_grad = False
                frozen_count["q"] += 1
            if hit_layer in kset and ".k_proj." in name:
                p.requires_grad = False
                frozen_count["k"] += 1
            if hit_layer in vset and ".v_proj." in name:
                p.requires_grad = False
                frozen_count["v"] += 1
    
    print(f"Frozen base Q/K/V projections: Q={frozen_count['q']}, K={frozen_count['k']}, V={frozen_count['v']}")
    
    # Apply LoRA
    lcfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lcfg)
    
    # Freeze LoRA in frozen layers
    lora_frozen = {"q": 0, "k": 0, "v": 0}
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        
        layer_hit = None
        for i in (qset | kset | vset):
            if f".layers.{i}." in name:
                layer_hit = i
                break
        if layer_hit is None:
            continue
        
        if layer_hit in qset and "q_proj" in name:
            p.requires_grad = False
            lora_frozen["q"] += 1
        if layer_hit in kset and "k_proj" in name:
            p.requires_grad = False
            lora_frozen["k"] += 1
        if layer_hit in vset and "v_proj" in name:
            p.requires_grad = False
            lora_frozen["v"] += 1
    
    print(f"Frozen LoRA adapters: Q={lora_frozen['q']}, K={lora_frozen['k']}, V={lora_frozen['v']}")
    
    # Count parameters by type
    trainable_base = 0
    trainable_lora = 0
    trainable_other = 0
    frozen_total = 0
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_total += p.numel()
        else:
            if "lora_" in name:
                trainable_lora += p.numel()
            elif any(proj in name for proj in [".q_proj.", ".k_proj.", ".v_proj."]):
                trainable_base += p.numel()
            else:
                trainable_other += p.numel()
    
    trainable_total = trainable_base + trainable_lora + trainable_other
    total_params = trainable_total + frozen_total
    
    print(f"\n📊 Parameter Breakdown:")
    print(f"  Trainable base Q/K/V: {trainable_base:,} ({100*trainable_base/total_params:.2f}%)")
    print(f"  Trainable LoRA:       {trainable_lora:,} ({100*trainable_lora/total_params:.2f}%)")
    print(f"  Trainable other:      {trainable_other:,} ({100*trainable_other/total_params:.2f}%)")
    print(f"  Frozen total:         {frozen_total:,} ({100*frozen_total/total_params:.2f}%)")
    print(f"  ──────────────────────────────────────────")
    print(f"  Trainable TOTAL:      {trainable_total:,} ({100*trainable_total/total_params:.2f}%)")
    print(f"  Total parameters:     {total_params:,} (100.00%)")
    
    model.print_trainable_parameters()
    
    return trainable_total, total_params

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Get layer orderings
    Q_ORDERED = [0, 16, 23, 18, 20, 17, 5, 10, 24, 12, 15, 8, 13, 19, 3, 6, 22, 2, 9, 21, 25, 11, 4, 14, 1, 7, 26, 27]
    K_ORDERED = [15, 8, 0, 12, 16, 18, 14, 17, 21, 9, 13, 10, 19, 11, 20, 24, 4, 6, 23, 7, 5, 25, 26, 22, 3, 27, 1, 2]
    V_ORDERED = [25, 23, 27, 24, 26, 20, 22, 21, 18, 15, 19, 3, 14, 13, 16, 1, 17, 11, 9, 6, 4, 8, 12, 7, 5, 10, 2, 0]
    
    # Test configurations
    configs = [
        ("No freezing", [], [], []),
        ("Lowest 3", Q_ORDERED[:3], K_ORDERED[:3], V_ORDERED[:3]),
        ("Lowest 15", Q_ORDERED[:15], K_ORDERED[:15], V_ORDERED[:15]),
        ("Highest 15", Q_ORDERED[-15:], K_ORDERED[-15:], V_ORDERED[-15:]),
    ]
    
    for name, q_layers, k_layers, v_layers in configs:
        print(f"\n\n🔍 Configuration: {name}")
        try:
            trainable, total = analyze_freezing(model_id, q_layers, k_layers, v_layers)
        except Exception as e:
            print(f"Error: {e}")
            continue
