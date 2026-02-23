import os
import numpy as np
import re

# Paths to the two baseline folders
dir_a = "/staging/users/aerol1/tda/Topo-Tuner/numpy_weights/FinEntity/DeepSeek-Qwen-7B/baseline"
dir_b = "/staging/users/aerol1/tda/Topo-Tuner/numpy_weights/FinEntity/DeepSeek-Qwen-7B/baseline"

# Normalization function to get comparable keys
def normalize_name(filename):
    if filename.startswith("layer"):
        return filename.replace(".npy", "")
    match = re.match(r"model_layers_(\d+)_self_attn_(q|k|v)_proj_weight.npy", filename)
    if match:
        layer, proj = match.groups()
        return f"layer{layer}_{proj}"
    return None

# Load and map files from both directories
files_a = {normalize_name(f): os.path.join(dir_a, f) for f in os.listdir(dir_a) if f.endswith(".npy")}
files_b = {normalize_name(f): os.path.join(dir_b, f) for f in os.listdir(dir_b) if f.endswith(".npy")}

# Compare files
for key in sorted(set(files_a) & set(files_b)):
    arr1 = np.load(files_a[key])
    arr2 = np.load(files_b[key])

    exact = np.array_equal(arr1, arr2)
    close = np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8)
    l2_diff = np.linalg.norm(arr1 - arr2)

    print(f"\n🔍 Comparing: {key}")
    print(f" - Exact match     : {'✅' if exact else '❌'}")
    print(f" - Close match     : {'✅' if close else '❌'}")
    print(f" - L2 Norm of diff : {l2_diff:.6f}")
