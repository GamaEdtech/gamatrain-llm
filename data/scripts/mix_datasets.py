import json
import random

gamatrain_file = "gamatrain_finetune_data.jsonl"
general_file = "general_knowledge.jsonl"
output_file = "gamatrain_finetune_data_mixed.jsonl"

def load_jsonl(filename):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
    return data

print("Loading datasets...")
gamatrain_data = load_jsonl(gamatrain_file)
general_data = load_jsonl(general_file)

print(f"Loaded {len(gamatrain_data)} Gamatrain samples.")
print(f"Loaded {len(general_data)} General Knowledge samples.")

# Weighting: Duplicate general data to ensure it's not drowned out
# If gamatrain has 1400 samples and general has 50, 50 is too small (~3%).
# We want general knowledge to be visible. Let's make it ~10-15% of the dataset.
# 1400 * 0.15 = 210. So we need ~200 general samples.
# 50 * 4 = 200.
weight_factor = 4
general_data_weighted = general_data * weight_factor

combined_data = gamatrain_data + general_data_weighted
random.shuffle(combined_data)

print(f"Combined dataset size: {len(combined_data)} samples.")

with open(output_file, 'w', encoding='utf-8') as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Saved mixed dataset to {output_file}")
