import json

# Files
main_file = "gamatrain_finetune_data.jsonl"
general_knowledge_file = "general_knowledge.jsonl"
output_file = "gamatrain_final_dataset.jsonl"

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
main_data = load_jsonl(main_file)
general_data = load_jsonl(general_knowledge_file)

print(f"Main dataset: {len(main_data)} samples")
print(f"General Knowledge: {len(general_data)} samples")

# Weight the general knowledge (4x) to avoid catastrophic forgetting
weight_factor = 4
general_data_weighted = general_data * weight_factor
print(f"General Knowledge (weighted {weight_factor}x): {len(general_data_weighted)} samples")

# Combine
import random
combined_data = main_data + general_data_weighted
random.shuffle(combined_data)

print(f"\nFinal dataset: {len(combined_data)} samples")

# Save
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Saved to {output_file}")
