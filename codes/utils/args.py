import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-R1-Distill-Qwen-7B with or without LoRA")
    parser.add_argument("--dataset-name", type=str, default="IFEval", help="Dataset name")
    parser.add_argument("--model-name", type=str, default="DeepSeek-Qwen-7B", help="Model name")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA for fine-tuning")
    parser.add_argument("--output-dir", type=str, default="./FinEntity-new/finetuned_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save model weights at the end of each epoch")
    parser.add_argument("--save-npy", action="store_true", help="Save model weights in NumPy format (.npy)")
    parser.add_argument("--save-baseline", action="store_true", help="Save baseline (epoch 0) weights")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    # Subset options (for large datasets like HotpotQA)
    parser.add_argument("--subset-train-size", type=int, default=20000, help="Number of training examples to sample (HotpotQA)")
    parser.add_argument("--subset-seed", type=int, default=42, help="Random seed for subset sampling")
    parser.add_argument("--subset-save-dir", type=str, default="", help="If set, save/load subset DatasetDict at this path")
    parser.add_argument("--train-csv", type=str, default="", help="Optional CSV file to override training split (e.g., perturbed dataset)")
    parser.add_argument(
        "--freeze-layers",
        nargs="*",
        type=int,
        default=[],
        help="List of transformer layer indices to freeze (e.g., --freeze-layers 7 11)"
    )
    parser.add_argument("--freeze-q-layers", nargs="*", type=int, default=[], help="Freeze ONLY q_proj in these layer idxs")
    parser.add_argument("--freeze-k-layers", nargs="*", type=int, default=[], help="Freeze ONLY k_proj in these layer idxs")
    parser.add_argument("--freeze-v-layers", nargs="*", type=int, default=[], help="Freeze ONLY v_proj in these layer idxs")

    parser.add_argument(
        "--hotpot-evidence",
        type=str,
        default="supporting",
        choices=["supporting", "full"],
        help="Evidence mode for HotpotQA prompts"
    )
    parser.add_argument(
        "--debug-hotpot",
        action="store_true",
        help="HotpotQA only: print one decoded tokenized sample ..."
    )
    return parser.parse_args()
