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

    if not os.path.exists(lora_path):
        print(f"❌ LoRA path does not exist: {lora_path}")
        return

    for epoch_dir in os.listdir(lora_path):
        epoch_path = os.path.join(lora_path, epoch_dir)
        if not os.path.isdir(epoch_path):
            continue

        output_epoch_path = os.path.join(lora_ba_path, epoch_dir)
        os.makedirs(output_epoch_path, exist_ok=True)

        print(f"🔄 Processing: {dataset_name}/{model_name}/{epoch_dir}")

        for file in tqdm(os.listdir(epoch_path), desc=f"{epoch_dir}", ncols=100):
            if not file.endswith("_A.npy"):
                continue

            a_path = os.path.join(epoch_path, file)
            b_path = a_path.replace("_A.npy", "_B.npy")
            w0_path = ""

            if not os.path.exists(b_path):
                print(f"⚠️ Missing B for {file}, skipping...")
                continue

            # Load matrices
            A = np.load(a_path)
            B = np.load(b_path)
            W0 = np.load(w0_path)  # assuming you have this as your baseline

            # Compute BA
            BA = B @ A

            # Set scalar
            alpha_over_r = 2

            # Final result
            BA_final = W0 + alpha_over_r * BA

            # Optional: save it
            np.save("ba_final.npy", BA_final)


            ba_filename = file.replace("_A.npy", "_BA.npy")
            ba_path = os.path.join(output_epoch_path, ba_filename)
            np.save(ba_path, BA)

    print(f"✅ Finished BA generation for {dataset_name}/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., FinEntity)")
    parser.add_argument("--model", required=True, help="Model name (e.g., DeepSeek-Qwen-7B)")
    args = parser.parse_args()

    generate_ba_from_lora_weights(args.dataset, args.model)


'''
python code/tda/generate_ba.py --dataset FinEntity --model DeepSeek-Qwen-7B
'''