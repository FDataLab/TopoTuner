import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from gudhi.wasserstein import wasserstein_distance

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_direct_comparisons(pairs, label, h0_only: bool = False):
    results = []
    for file1, file2 in tqdm(pairs, desc=f"{label}"):
        try:
            dgm1 = load_pkl(file1)
            dgm2 = load_pkl(file2)

            h0 = wasserstein_distance(dgm1[0], dgm2[0])
            row = {
                "Type": label,
                "File": os.path.basename(file1),
                "Wasserstein H0": h0,
            }
            if not h0_only:
                h1 = (
                    wasserstein_distance(dgm1[1], dgm2[1])
                    if len(dgm1) > 1 and len(dgm2) > 1
                    else None
                )
                if h1 is not None:
                    row["Wasserstein H1"] = h1
            results.append(row)
        except Exception as e:
            print(f"❌ Skipping {file1} vs {file2}: {e}")
    return results

def compute_baseline_vs_epochs(
    baseline_dir,
    epoch_dirs,
    output_csv,
    projections=None,
    h0_only: bool = False,
):
    """Compare baseline persistence diagrams against each epoch's diagrams.

    Args:
        baseline_dir: path to SavedDiagrams/ for baseline (epoch 0 / pretrained)
        epoch_dirs: list of (epoch_label, path_to_SavedDiagrams)
        output_csv: where to write results
        projections: list of projection suffixes to compare, e.g. ["q", "k", "v", "o"]
        h0_only: if True, only Wasserstein H0 is computed and stored (no H1 column).
    """
    if projections is None:
        projections = ["q", "k", "v", "o"]

    all_results = []
    for epoch_label, epoch_dir in epoch_dirs:
        for proj in projections:
            suffix = f"_{proj}.pkl"
            pairs = []
            for fname in sorted(os.listdir(baseline_dir)):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_dir, fname)
                    f2 = os.path.join(epoch_dir, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            if not pairs:
                print(f"  ⚠️ No matching {proj} files for {epoch_label}")
                continue
            results = compute_direct_comparisons(
                pairs, f"Baseline vs {epoch_label} ({proj})", h0_only=h0_only
            )
            for r in results:
                r["Epoch"] = epoch_label
                r["Projection"] = proj
            all_results += results

    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved Wasserstein results to: {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--baseline-dir", default=None,
                        help="Direct path to baseline SavedDiagrams/ (bypasses dataset/model path construction)")
    parser.add_argument("--epoch-dirs", default=None,
                        help="Comma-separated epoch_label:path pairs, e.g. 'epoch_1:/path/to/e1/SavedDiagrams,epoch_2:/path/to/e2/SavedDiagrams'")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--projections", default="q,k,v,o",
                        help="Comma-separated projection types (default: q,k,v,o)")
    parser.add_argument(
        "--h0-only",
        action="store_true",
        help="Only compute Wasserstein H0 (omit H1 from CSV).",
    )
    args = parser.parse_args()

    # Direct path mode
    if args.baseline_dir:
        if not args.epoch_dirs or not args.output:
            parser.error("--epoch-dirs and --output are required with --baseline-dir")
        epoch_dirs = []
        for entry in args.epoch_dirs.split(","):
            label, path = entry.split(":", 1)
            epoch_dirs.append((label, path))
        projections = [p.strip() for p in args.projections.split(",")]
        compute_baseline_vs_epochs(
            args.baseline_dir,
            epoch_dirs,
            args.output,
            projections,
            h0_only=args.h0_only,
        )
        exit(0)

    # Legacy mode (original hardcoded paths)
    if not args.dataset or not args.model:
        parser.error("--dataset and --model are required in legacy mode")

    base = "/staging/users/aerol1/tda/Topo-Tuner"
    dset, model = args.dataset, args.model

    paths = {
        "baseline": f"{base}/persistence_diagrams/{dset}/{model}/baseline/SavedDiagrams",
        "lora_A_1": f"{base}/persistence_diagrams/{dset}/{model}/lora_A/epoch_1/SavedDiagrams",
        "lora_A_6": f"{base}/persistence_diagrams/{dset}/{model}/lora_A/epoch_6/SavedDiagrams",
        "lora_B_1": f"{base}/persistence_diagrams/{dset}/{model}/lora_B/epoch_1/SavedDiagrams",
        "lora_B_6": f"{base}/persistence_diagrams/{dset}/{model}/lora_B/epoch_6/SavedDiagrams",
        "lora_BA": f"{base}/persistence_diagrams/{dset}/{model}/lora_BA/epoch_6/SavedDiagrams",
        "full": f"{base}/persistence_diagrams/{dset}/{model}/full/epoch_6/SavedDiagrams"
    }

    all_results = []

    for suffix in ["_q.pkl", "_k.pkl", "_v.pkl"]:
        pairs = []
        for fname in os.listdir(paths["baseline"]):
            if fname.endswith(suffix):
                f1 = os.path.join(paths["baseline"], fname)
                f2 = os.path.join(paths["lora_BA"], fname)
                if os.path.exists(f2):
                    pairs.append((f1, f2))
        all_results += compute_direct_comparisons(pairs, f"Baseline vs LoRA BA ({suffix[1]})")

        pairs = []
        for fname in os.listdir(paths["baseline"]):
            if fname.endswith(suffix):
                f1 = os.path.join(paths["baseline"], fname)
                f2 = os.path.join(paths["full"], fname)
                if os.path.exists(f2):
                    pairs.append((f1, f2))
        all_results += compute_direct_comparisons(pairs, f"Baseline vs Full Finetune ({suffix[1]})")

        pairs = []
        for fname in os.listdir(paths["lora_BA"]):
            if fname.endswith(suffix):
                f1 = os.path.join(paths["lora_BA"], fname)
                f2 = os.path.join(paths["full"], fname)
                if os.path.exists(f2):
                    pairs.append((f1, f2))
        all_results += compute_direct_comparisons(pairs, f"LoRA BA vs Full Finetune ({suffix[1]})")

        pairs = []
        for fname in os.listdir(paths["lora_A_1"]):
            if fname.endswith(suffix):
                f1 = os.path.join(paths["lora_A_1"], fname)
                f2 = os.path.join(paths["lora_A_6"], fname)
                if os.path.exists(f2):
                    pairs.append((f1, f2))
        all_results += compute_direct_comparisons(pairs, f"LoRA A: Epoch 1 vs 6 ({suffix[1]})")

        pairs = []
        for fname in os.listdir(paths["lora_B_1"]):
            if fname.endswith(suffix):
                f1 = os.path.join(paths["lora_B_1"], fname)
                f2 = os.path.join(paths["lora_B_6"], fname)
                if os.path.exists(f2):
                    pairs.append((f1, f2))
        all_results += compute_direct_comparisons(pairs, f"LoRA B: Epoch 1 vs 6 ({suffix[1]})")

    df = pd.DataFrame(all_results)
    os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
    save_path = f"{base}/wasserstein_results/wasserstein_{dset}_{model}.csv"
    df.to_csv(save_path, index=False)
    print(f"\n✅ Saved Wasserstein results to: {save_path}")


# import os
# import argparse
# import pickle
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from gudhi.wasserstein import wasserstein_distance

# def load_pkl(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def compute_direct_comparisons(pairs, label):
#     results = []
#     for file1, file2 in tqdm(pairs, desc=f"{label}"):
#         try:
#             dgm1 = load_pkl(file1)
#             dgm2 = load_pkl(file2)

#             h0 = wasserstein_distance(dgm1[0], dgm2[0])
#             h1 = wasserstein_distance(dgm1[1], dgm2[1])
#             results.append({
#                 "Type": label,
#                 "File": os.path.basename(file1),
#                 "Wasserstein H0": h0,
#                 "Wasserstein H1": h1
#             })
#         except Exception as e:
#             print(f"❌ Skipping {file1} vs {file2}: {e}")
#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", required=True)
#     parser.add_argument("--model", required=True)
#     args = parser.parse_args()

#     base = "/staging/users/aerol1/tda/Topo-Tuner"
#     dset, model = args.dataset, args.model

#     paths = {
#         "baseline": f"{base}/persistence_diagrams/{dset}/{model}/baseline/SavedDiagrams",
#         "lora_A": f"{base}/persistence_diagrams/{dset}/{model}/lora_A/epoch_1/SavedDiagrams",
#         "lora_B": f"{base}/persistence_diagrams/{dset}/{model}/lora_B/epoch_1/SavedDiagrams",
#         "lora_BA": f"{base}/persistence_diagrams/{dset}/{model}/lora_BA/epoch_1/SavedDiagrams",
#         "full": f"{base}/persistence_diagrams/{dset}/{model}/full/epoch_1/SavedDiagrams"
#     }

#     all_results = []

#     # Comparison: baseline/layerX_k.pkl vs lora_BA/layerX_k.pkl
#     pairs = []
#     for fname in os.listdir(paths["baseline"]):
#         if fname.endswith("_k.pkl") and "_A" not in fname and "_B" not in fname:
#             f1 = os.path.join(paths["baseline"], fname)
#             f2 = os.path.join(paths["lora_BA"], fname)
#             if os.path.exists(f2):
#                 pairs.append((f1, f2))
#     all_results += compute_direct_comparisons(pairs, "Baseline vs LoRA BA")

#     # Comparison: baseline/layerX_k.pkl vs full/layerX_k.pkl
#     pairs = []
#     for fname in os.listdir(paths["baseline"]):
#         if fname.endswith("_k.pkl") and "_A" not in fname and "_B" not in fname:
#             f1 = os.path.join(paths["baseline"], fname)
#             f2 = os.path.join(paths["full"], fname)
#             if os.path.exists(f2):
#                 pairs.append((f1, f2))
#     all_results += compute_direct_comparisons(pairs, "Baseline vs Full Finetune")

#     # Comparison: lora_BA/layerX_k.pkl vs full/layerX_k.pkl
#     pairs = []
#     for fname in os.listdir(paths["lora_BA"]):
#         if fname.endswith("_k.pkl"):
#             f1 = os.path.join(paths["lora_BA"], fname)
#             f2 = os.path.join(paths["full"], fname)
#             if os.path.exists(f2):
#                 pairs.append((f1, f2))
#     all_results += compute_direct_comparisons(pairs, "LoRA BA vs Full Finetune")

#     # Comparison: baseline/layerX_k_A.pkl vs lora_A/layerX_k.pkl
#     pairs = []
#     for fname in os.listdir(paths["baseline"]):
#         if fname.endswith("_k_A.pkl"):
#             f1 = os.path.join(paths["baseline"], fname)
#             base_name = fname.replace("_A", "")
#             f2 = os.path.join(paths["lora_A"], base_name)
#             if os.path.exists(f2):
#                 pairs.append((f1, f2))
#     all_results += compute_direct_comparisons(pairs, "Baseline A vs LoRA A")

#     # Comparison: baseline/layerX_k_B.pkl vs lora_B/layerX_k.pkl
#     pairs = []
#     for fname in os.listdir(paths["baseline"]):
#         if fname.endswith("_k_B.pkl"):
#             f1 = os.path.join(paths["baseline"], fname)
#             base_name = fname.replace("_B", "")
#             f2 = os.path.join(paths["lora_B"], base_name)
#             if os.path.exists(f2):
#                 pairs.append((f1, f2))
#     all_results += compute_direct_comparisons(pairs, "Baseline B vs LoRA B")

#     # Save results
#     df = pd.DataFrame(all_results)
#     os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
#     save_path = f"{base}/wasserstein_results/wasserstein_{dset}_{model}.csv"
#     df.to_csv(save_path, index=False)
#     print(f"\n✅ Saved Wasserstein results to: {save_path}")


"""
nohup python code/tda/compute_wasserstein.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  > logs/wasserstein_FinEntity_DeepSeek-Qwen-7B.log 2>&1 &
"""