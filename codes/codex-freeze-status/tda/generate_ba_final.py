import os
import numpy as np
from tqdm import tqdm
import argparse

def generate_ba_from_lora_weights(dataset_name, model_name):
    base_path = os.path.join(
        "./numpy_weights",
        dataset_name,
        model_name
    )
    lora_path = os.path.join(base_path, "lora")  # Use standard LoRA path
    lora_ba_path = os.path.join(base_path, "loraBA")
    lora_final_path = os.path.join(base_path, "lora-final")
    # Use full model epoch 0 as baseline (W₀) - these are the original pre-trained weights
    w0_path = os.path.join(base_path, "full", "epoch_weights", "checkpoint-epoch-0", "numpy_weights")

    if not os.path.exists(lora_path):
        print(f"❌ LoRA path does not exist: {lora_path}")
        return

    # Look for epoch_weights directory structure
    epoch_weights_path = os.path.join(lora_path, "epoch_weights")
    if not os.path.exists(epoch_weights_path):
        print(f"❌ Epoch weights path does not exist: {epoch_weights_path}")
        return

    for epoch in range(7):  # epochs 0-6
        epoch_dir = f"checkpoint-epoch-{epoch}"
        epoch_path = os.path.join(epoch_weights_path, epoch_dir, "numpy_weights")
        
        if not os.path.exists(epoch_path):
            print(f"⚠️ Skipping missing epoch: {epoch_dir}")
            continue

        output_ba_epoch_path = os.path.join(lora_ba_path, f"epoch_{epoch}")
        output_final_epoch_path = os.path.join(lora_final_path, f"epoch_{epoch}")
        os.makedirs(output_ba_epoch_path, exist_ok=True)
        os.makedirs(output_final_epoch_path, exist_ok=True)

        print(f"🔄 Processing: {dataset_name}/{model_name}/{epoch_dir}")

        for file in tqdm(os.listdir(epoch_path), desc=f"{epoch_dir}", ncols=100):
            if not file.endswith("_A.npy"):
                continue

            a_path = os.path.join(epoch_path, file)
            b_path = a_path.replace("_A.npy", "_B.npy")

            if not os.path.exists(b_path):
                print(f"⚠️ Missing B for {file}, skipping...")
                continue

            # Derive baseline file path: layer0_q_A.npy → layer0_q.npy (full model weights)
            # The baseline is the original pre-trained weight matrix (not LoRA A/B matrices)
            baseline_file = file.replace("_A.npy", ".npy")  # Remove _A suffix for full model weights
            w0_file_path = os.path.join(w0_path, baseline_file)

            if not os.path.exists(w0_file_path):
                print(f"⚠️ Missing W₀ for {file}, skipping...")
                continue

            # Load matrices
            A = np.load(a_path)
            B = np.load(b_path)
            W0 = np.load(w0_file_path)

            # Compute BA and BAfinal
            BA = B @ A
            alpha_over_r = 2
            BA_final = W0 + alpha_over_r * BA

            # Save BA to loraBA
            ba_filename = file.replace("_A.npy", "_BA.npy")
            ba_path = os.path.join(output_ba_epoch_path, ba_filename)
            np.save(ba_path, BA)

            # Save BAfinal to lora-final
            bafinal_filename = file.replace("_A.npy", "_BAfinal.npy")
            bafinal_path = os.path.join(output_final_epoch_path, bafinal_filename)
            np.save(bafinal_path, BA_final)

    print(f"✅ Finished BA + BAfinal generation for {dataset_name}/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., FinEntity)")
    parser.add_argument("--model", required=True, help="Model name (e.g., DeepSeek-Qwen-7B)")
    args = parser.parse_args()

    generate_ba_from_lora_weights(args.dataset, args.model)

"""
# MMLU BA and BAfinal Generation Commands
# Run these after LoRA training is complete

python codes/tda/generate_ba_final.py --dataset mmlu --model llama32_3b
python codes/tda/generate_ba_final.py --dataset mmlu --model llama31_8b
python codes/tda/generate_ba_final.py --dataset mmlu --model mistral7b
python codes/tda/generate_ba_final.py --dataset mmlu --model qwen_8b
"""
