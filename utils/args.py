import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-R1-Distill-Qwen-7B with or without LoRA")
    parser.add_argument("--dataset-name", type=str, default="GSM8K", help="Dataset name")
    parser.add_argument("--model-name", type=str, default="DeepSeek-Qwen-7B", help="Model name")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA for fine-tuning")
    parser.add_argument("--output-dir", type=str, default="./GSM8K/finetuned_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save model weights at the end of each epoch")
    parser.add_argument("--save-npy", action="store_true", help="Save model weights in NumPy format (.npy)")
    parser.add_argument("--save-baseline", action="store_true", help="Save baseline (epoch 0) weights")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()
