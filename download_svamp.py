from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

# Load SVAMP dataset from Hugging Face
dataset = load_dataset("deepmind/aqua_rat")  # Correct HuggingFace repo

data = dataset['validation']
examples = data.to_list()

with open("datasets/aqua/val.jsonl", "w") as f:
    for example in examples:
        json.dump(example, f, ensure_ascii=False)
        f.write("\n")


#     aqua_val_dataset = load_dataset("your_username/AquaVAL") 


# # Convert to list of dicts
# data = dataset["train"]  # Entire dataset is in 'train' split
# examples = data.to_list()

# # Split into 80% train, 20% test
# train_data, test_data = train_test_split(examples, test_size=0.2, random_state=42)

# # Save to JSON files
# with open("datasets/SVAMP/train.json", "w") as f:
#     json.dump(train_data, f, indent=2)

# with open("datasets/SVAMP/test.json", "w") as f:
#     json.dump(test_data, f, indent=2)

# print(f"Saved {len(train_data)} training examples and {len(test_data)} testing examples.")