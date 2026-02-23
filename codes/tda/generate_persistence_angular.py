import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import ripser
from persim import plot_diagrams

def concise_filename(fname):
    parts = fname.replace(".npy", "").split("_")
    if "layer" in parts[0]:
        layer = parts[0].replace("layer", "")
        proj = parts[1]
        return f"layer{layer}_{proj}"
    return fname.replace(".npy", "")

def plot_persistence(diagrams, shortname, output_folder):
    plt.figure(figsize=(6, 4))
    plot_diagrams(diagrams)
    plt.title(f"Persistence Diagram - {shortname}")
    output_path = os.path.join(output_folder, f"{shortname}_H0_H1.png")
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()

def _row_normalize(X, eps=1e-12):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)

def compute_sparse_cosine_knn_distance_matrix(X, k=64):
    """
    Builds a scipy sparse CSR distance matrix for cosine distances using kNN.
    Missing edges are treated as infinity by ripser, so sparsity helps a lot.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        import scipy.sparse as sp
    except Exception as e:
        raise RuntimeError("Need scikit-learn and scipy for sparse kNN distances") from e

    Xn = _row_normalize(X)
    n = Xn.shape[0]
    k = min(k, n)

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="auto", n_jobs=-1)
    nn.fit(Xn)
    dist, idx = nn.kneighbors(Xn, return_distance=True)

    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = idx.reshape(-1).astype(np.int32, copy=False)
    data = dist.reshape(-1).astype(np.float32, copy=False)

    D = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    D = D.minimum(D.T)
    D.setdiag(0.0)
    D.eliminate_zeros()
    return D

def compute_dense_cosine_distance_matrix_fast(X):
    """
    Fast dense cosine distance via BLAS: D = 1 - Xn Xn^T
    Still O(n^2) memory, so use only if you must.
    """
    Xn = _row_normalize(X)
    S = Xn @ Xn.T
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    np.clip(D, 0.0, 2.0, out=D)
    return D.astype(np.float32, copy=False)

def compute_persistence(diagram_input, thresh=None, maxdim=1):
    """
    diagram_input can be dense ndarray or scipy sparse CSR.
    thresh limits the filtration radius and can speed things up further.
    """
    kwargs = dict(distance_matrix=True, maxdim=maxdim)
    if thresh is not None:
        kwargs["thresh"] = float(thresh)
    return ripser(diagram_input, **kwargs)["dgms"]

def process_npy_files_to_persistence(
    input_folder,
    output_folder,
    file_filter,
    plot=False,
    use_sparse_knn=True,
    knn_k=64,
    ripser_thresh=None,
    overwrite=False
):
    os.makedirs(output_folder, exist_ok=True)
    diagrams_folder = os.path.join(output_folder, "SavedDiagrams")
    os.makedirs(diagrams_folder, exist_ok=True)

    npy_files = [f for f in os.listdir(input_folder) if f.endswith(".npy") and file_filter(f)]
    print(f"Found {len(npy_files)} .npy files in {input_folder}")

    for file in tqdm(npy_files, desc=f"[{os.path.basename(input_folder)}] persistence", ncols=100, mininterval=0.5):
        shortname = concise_filename(file)
        diagram_file = os.path.join(diagrams_folder, f"{shortname}.pkl")
        fig_file = os.path.join(diagrams_folder, f"{shortname}_H0_H1.png")

        if (not overwrite) and os.path.exists(diagram_file) and ((not plot) or os.path.exists(fig_file)):
            continue

        file_path = os.path.join(input_folder, file)
        weights = np.load(file_path, mmap_mode="r")

        if use_sparse_knn:
            dist_matrix = compute_sparse_cosine_knn_distance_matrix(weights, k=knn_k)
        else:
            dist_matrix = compute_dense_cosine_distance_matrix_fast(weights)

        diagrams = compute_persistence(dist_matrix, thresh=ripser_thresh, maxdim=1)

        with open(diagram_file, "wb") as f:
            pickle.dump(diagrams, f, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            plot_persistence(diagrams, shortname, diagrams_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=["lora_A", "lora_B", "lora_BA", "full", "baseline", "lora-final"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--knn_k", type=int, default=64)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--thresh", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    folder_map = {
        "lora_A": "lora",
        "lora_B": "lora",
        "lora_BA": "loraBA",
        "full": "full",
        "baseline": "baseline",
        "lora-final": "lora-final"
    }
    weight_folder = folder_map[args.mode]

    base_path = f"/staging/users/aerol1/tda/Topo-Tuner/numpy_weights/{args.dataset}/{args.model}/{weight_folder}"
    output_base = f"/staging/users/aerol1/tda/Topo-Tuner/persistence_diagrams/{args.dataset}/{args.model}/{args.mode}"

    def lora_A_filter(f): return f.endswith("_A.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_B_filter(f): return f.endswith("_B.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_BA_filter(f): return f.endswith("_BA.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_final_filter(f): return f.endswith("_BAfinal.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def full_filter(f): return f.endswith(".npy") and any(f"_{k}.npy" in f for k in ["k", "q", "v"])
    def baseline_filter(f): return f.endswith(".npy") and any(f"_{k}" in f for k in ["k", "q", "v"])

    mode_filters = {
        "lora_A": lora_A_filter,
        "lora_B": lora_B_filter,
        "lora_BA": lora_BA_filter,
        "full": full_filter,
        "baseline": baseline_filter,
        "lora-final": lora_final_filter
    }
    file_filter = mode_filters[args.mode]

    use_sparse_knn = (not args.dense)

    if args.mode == "baseline":
        if not os.path.exists(base_path):
            print(f"Skipping missing folder: {base_path}")
        else:
            process_npy_files_to_persistence(
                base_path, output_base, file_filter,
                plot=args.plot,
                use_sparse_knn=use_sparse_knn,
                knn_k=args.knn_k,
                ripser_thresh=args.thresh,
                overwrite=args.overwrite
            )
    else:
        for epoch in range(1, 101):
            input_folder = os.path.join(base_path, f"epoch_{epoch}")
            output_folder = os.path.join(output_base, f"epoch_{epoch}")
            if not os.path.exists(input_folder):
                continue
            process_npy_files_to_persistence(
                input_folder, output_folder, file_filter,
                plot=args.plot,
                use_sparse_knn=use_sparse_knn,
                knn_k=args.knn_k,
                ripser_thresh=args.thresh,
                overwrite=args.overwrite
            )
