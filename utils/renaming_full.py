import os
import re

def rename_full_finetune_weights(base_dir):
    if not os.path.exists(base_dir):
        print(f"❌ Base directory does not exist: {base_dir}")
        return

    for epoch_folder in os.listdir(base_dir):
        epoch_path = os.path.join(base_dir, epoch_folder)
        if not os.path.isdir(epoch_path):
            continue

        print(f"🔄 Processing {epoch_folder}")
        for filename in os.listdir(epoch_path):
            if not filename.endswith(".npy"):
                continue

            # Match pattern for self-attention projection weights
            match = re.match(r"model\.layers\.(\d+)\.self_attn\.(k|q|v)_proj\.weight\.npy", filename)
            if match:
                layer, proj = match.groups()
                new_name = f"layer{layer}_{proj}.npy"
                old_path = os.path.join(epoch_path, filename)
                new_path = os.path.join(epoch_path, new_name)
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {filename} → {new_name}")

if __name__ == "__main__":
    # Change this to your actual full weight directory
    full_weights_dir = "/staging/users/aerol1/tda/Topo-Tuner/numpy_weights/FinEntity/DeepSeek-Qwen-7B/full"
    rename_full_finetune_weights(full_weights_dir)