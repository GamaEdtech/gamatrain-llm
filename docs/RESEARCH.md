# Research Findings 📊

## Problem Statement

Gamatrain is an educational platform with rich content (courses, tests, blogs). We wanted to create an AI assistant that could answer questions about this content while maintaining general intelligence.

## Approach

### 1. Data Collection
- **API Extraction**: Used Gamatrain's API to fetch courses, tests, quizzes, and school data
- **Blog Scraping**: Extracted ~1000 blog posts from sitemaps (gamatrain.com/sitemap/blog-*.xml)
- **Manual Curation**: Created 48 general knowledge samples for math, logic, and chat

### 2. Data Formatting
Converted all data to Qwen's ChatML format:
```json
{
  "messages": [
    {"role": "system", "content": "You are Gamatrain AI..."},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

### 3. Fine-tuning
- **Method**: QLoRA (4-bit quantization with LoRA adapters)
- **Platform**: Google Colab (T4 GPU)
- **Base Model**: Qwen2-1.5B-Instruct

## Key Finding: Catastrophic Forgetting ⚠️

### The Problem
When we first fine-tuned with only Gamatrain data, the model:
- ❌ Failed basic math: `2 + 2 = 0`
- ❌ Lost general knowledge
- ❌ Hallucinated "EduBridge" platform (never in training data)

### Root Cause
The dataset was 100% domain-specific with repetitive patterns:
- "Does Gamatrain have...?" (repeated 1200+ times)
- "Tell me about..." (physics/chemistry terms)

This caused the model to **overwrite** its pre-trained general knowledge.

### The Solution
Mixed domain data with weighted general knowledge:

| Dataset | Samples | Weight | Final Count |
|---------|---------|--------|-------------|
| Gamatrain | 2,422 | 1x | 2,422 |
| General Knowledge | 48 | 4x | 192 |
| **Total** | | | **2,614** |

### Result
After retraining with mixed data:
- ✅ Math works: `2 + 2 = 4`
- ✅ General knowledge retained
- ✅ Gamatrain content learned

## Recommendations

1. **Always mix domain data with general data** (at least 5-10% general)
2. **Diversify question patterns** - don't use the same template 1000x
3. **Test basic abilities** after fine-tuning (math, logic, identity)
4. **Use weighted sampling** for smaller datasets to increase their impact
