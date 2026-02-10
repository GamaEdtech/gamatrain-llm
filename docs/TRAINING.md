# Training Guide 🎓

## Prerequisites

- Google Account (for Colab)
- GPU runtime (T4 or better)

## Step 1: Prepare Dataset

### Option A: Use Pre-built Dataset
The `data/gamatrain_final_dataset.jsonl` file is ready to use.

### Option B: Create Your Own
```bash
cd data/scripts/

# 1. Extract API data
pip install -r requirements.txt
python extract_and_format_data.py

# 2. Extract blog data
python extract_blog_data.py

# 3. Generate general knowledge
python generate_general_data.py

# 4. Merge datasets
python create_final_dataset.py
```

## Step 2: Upload to Colab

1. Open `notebooks/fine-tuning-demo.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Upload `gamatrain_final_dataset.jsonl`

## Step 3: Run Training

Execute all cells in the notebook. Key parameters:

```python
# Training config
max_seq_length = 2048
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
num_train_epochs = 3
learning_rate = 2e-4
```

Training takes ~30-60 minutes on T4.

## Step 4: Export Model

The notebook will:
1. Merge LoRA adapters with base model
2. Convert to GGUF format (4-bit quantized)
3. Download `qwen2-gamatrain.gguf`

## Step 5: Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md) for Ollama setup.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM Error | Reduce batch size or sequence length |
| Slow training | Ensure GPU is enabled |
| Poor results | Check dataset quality, increase epochs |
