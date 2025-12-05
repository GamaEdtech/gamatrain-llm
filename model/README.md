# Model Download

The trained model file `qwen2-gamatrain.gguf` is too large for GitHub (1-2GB).

## 📥 Download Options

### Option 1: Hugging Face (Recommended)
```bash
# Coming soon - upload your model to Hugging Face
# huggingface-cli download gamatrain/qwen2-gamatrain --local-dir .
```

### Option 2: Train Your Own
Follow the instructions in [../docs/TRAINING.md](../docs/TRAINING.md) to train your own model using Google Colab.

## 🚀 Usage with Ollama

Once you have the `qwen2-gamatrain.gguf` file:

1. Place it in this directory
2. Run:
```bash
ollama create gamatrain-qwen -f Modelfile
```

3. Test:
```bash
ollama run gamatrain-qwen "Hello, who are you?"
```

## 📋 Model Specifications

| Property | Value |
|----------|-------|
| Base Model | Qwen2-1.5B-Instruct |
| Fine-tuning Method | QLoRA |
| Quantization | 4-bit (Q4_K_M) |
| File Size | ~1-2 GB |
| Context Length | 4096 tokens |
