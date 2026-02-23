import os
import re
import argparse
import pandas as pd
import pickle
from tqdm import tqdm

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def determine_type_label(path):
    if "baseline" in path:
        return "Baseline"
    elif "full" in path:
        return "Full Finetuned"
    elif "lora-final" in path:
        return "LoRA Final"
    else:
        return "Unknown"

def method_sort(row):
    if row["Type"] == "Baseline":
        return 0
    elif row["Type"] == "Full Finetuned":
        return row["Epoch"]
    elif row["Type"] == "LoRA Final":
        return 100 + row["Epoch"]
    return 999

def main(dataset, model):
    base_dir = "/staging/users/aerol1/tda/Topo-Tuner/persistence_diagrams"
    out_dir = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_results"
    all_data = []

    for variant in ["baseline", "full", "lora-final"]:
        for epoch in range(0, 21):
            if variant == "baseline":
                diag_dir = f"{base_dir}/{dataset}/{model}/baseline/SavedDiagrams"
            else:
                diag_dir = f"{base_dir}/{dataset}/{model}/{variant}/epoch_{epoch}/SavedDiagrams"
            if not os.path.exists(diag_dir):
                continue

            for fname in os.listdir(diag_dir):
                if not fname.endswith(".pkl"):
                    continue
                fpath = os.path.join(diag_dir, fname)
                try:
                    dgm = load_pkl(fpath)
                    num_h1 = len(dgm[1])
                except Exception as e:
                    print(f"❌ Error loading {fpath}: {e}")
                    continue

                # Extract head and layer
                head_type = fname.split("_")[-1].replace(".pkl", "").lower()
                layer_match = re.search(r"layer(\d+)", fname)
                layer = int(layer_match.group(1)) if layer_match else -1

                # Record result
                all_data.append({
                    "HeadType": head_type,
                    "Layer": layer,
                    "Type": determine_type_label(diag_dir),
                    "Epoch": epoch,
                    "File": fname,
                    "Num H1s": num_h1
                })

    # === DataFrame + Sorting ===
    df = pd.DataFrame(all_data)
    df["MethodOrder"] = df.apply(method_sort, axis=1)
    head_priority = {"k": 0, "q": 1, "v": 2}
    df["HeadOrder"] = df["HeadType"].map(head_priority)
    df_sorted = df.sort_values(by=["HeadOrder", "Layer", "MethodOrder"])
    df_sorted = df_sorted[["HeadType", "Layer", "Type", "Epoch", "File", "Num H1s"]]

    # === Save Output ===
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/num_h1s_{dataset}_{model}.csv"
    df_sorted.to_csv(output_path, index=False)
    print(f"✅ Final sorted H1 summary saved to:\n{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.dataset, args.model)

    """
    python /staging/users/aerol1/tda/Topo-Tuner/code/tda/compute_h1s.py --dataset FinEntity --model DeepSeek-Qwen-7B
    """