import os
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM

def save_weight_matrix(param, path):
    if hasattr(param, "detach"):
        param = param.detach().cpu().numpy()
    np.save(path, param)

def concise_proj_filename(param_name: str) -> str:
    """
    Convert full model weight name to concise filename like 'layer0_q.npy'
    """
    parts = param_name.split(".")
    try:
        layer_idx = parts[2]
        proj_type = parts[4].split("_")[0]  # 'q_proj.weight' -> 'q'
        return f"layer{layer_idx}_{proj_type}.npy"
    except IndexError:
        return None

def save_proj_weights(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for name, param in model.named_parameters():
        if name.endswith("q_proj.weight") or name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
            shortname = concise_proj_filename(name)
            if shortname:
                save_path = os.path.join(output_dir, shortname)
                save_weight_matrix(param, save_path)
                count += 1
    print(f"✅ Saved {count} baseline projection weights to: {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--dataset", default="FinEntity", help="Dataset name for folder structure")
    parser.add_argument("--model_tag", default="DeepSeek-Qwen-7B", help="Folder name tag for the model")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    output_dir = f"/staging/users/aerol1/tda/Topo-Tuner/numpy_weights/{args.dataset}/{args.model_tag}/baseline"
    save_proj_weights(model, output_dir)

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0 nohup python /staging/users/aerol1/tda/Topo-Tuner/code/finetuning/generate_baseline.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset FinEntity \
  --model_tag DeepSeek-Qwen-7B \
  > /staging/users/aerol1/tda/Topo-Tuner/logs/baseline_FinEntity_DeepSeek-Qwen-7B.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python /staging/users/aerol1/tda/Topo-Tuner/code/utils/generate_baseline.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset GSM8K \
  --model_tag DeepSeek-Qwen-7B \
  > /staging/users/aerol1/tda/Topo-Tuner/logs/baseline_GSM8K_DeepSeek-Qwen-7B.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python /staging/users/aerol1/tda/Topo-Tuner/code/utils/generate_baseline.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset IFEval \
  --model_tag DeepSeek-Qwen-7B \
  > /staging/users/aerol1/tda/Topo-Tuner/logs/baseline_IFEval_DeepSeek-Qwen-7B.log 2>&1 &
"""
