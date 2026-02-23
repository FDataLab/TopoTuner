import os
import numpy as np
from tqdm import tqdm
import argparse

def generate_ba_from_lora_weights(dataset_name, model_name):
    base_path = os.path.join(
        "/staging/users/aerol1/tda/Topo-Tuner/numpy_weights",
        dataset_name,
        model_name
    )
    lora_path = os.path.join(base_path, "lora")
    lora_ba_path = os.path.join(base_path, "loraBA")
    lora_final_path = os.path.join(base_path, "lora-final")
    w0_path = os.path.join(base_path, "baseline")

    if not os.path.exists(lora_path):
        print(f"❌ LoRA path does not exist: {lora_path}")
        return

    for epoch_dir in os.listdir(lora_path):
        epoch_path = os.path.join(lora_path, epoch_dir)
        if not os.path.isdir(epoch_path):
            continue

        output_ba_epoch_path = os.path.join(lora_ba_path, epoch_dir)
        output_final_epoch_path = os.path.join(lora_final_path, epoch_dir)
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

            # Derive baseline file path: layer0_q_A.npy → layer0_q.npy
            baseline_file = file.replace("_A.npy", ".npy")
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
python code/finetuning/generate_ba_final.py --dataset FinEntity --model DeepSeek-Qwen-7B
"""
