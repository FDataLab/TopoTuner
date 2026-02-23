from datasets import load_dataset

# Load the FinEntity dataset
dataset = load_dataset("yixuantt/FinEntity")
train_data = dataset["train"]

# Compute the max length of input text
max_length = max(len(sample["content"]) for sample in train_data)
print("Maximum character length of a sample:", max_length)
max_length = max(len(sample["annotations"]) for sample in train_data)
print("Maximum character length of annotations:", max_length)

