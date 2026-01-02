# Benchmark Results Summary

## Overview

This document summarizes the results from comprehensive LoRA fine-tuning experiments across multiple models, datasets, and configurations. All data is sourced from `outputs/results/benchmark_results.csv` and is continuously logged to Weights & Biases (W&B).

## Key Findings

### Model Performance Hierarchy

**Gemma-2B** achieves the best perplexity scores across datasets:
- Lowest validation PPL: **12.4** on TinyStories
- Fastest training throughput: **3,400 tokens/sec**
- Best for narrative-heavy and story-based text (14.2 PPL on storytelling)
- Optimal for resource-constrained environments

**GPT-2 (124M)** offers balanced performance:
- Competitive PPL range: 15.2–24.3 depending on domain
- Strong performance on news (15.2 PPL) and code (19.5 PPL) domains
- Mid-range throughput: ~1,300 tokens/sec
- Reliable baseline for ablation studies

**Pythia-410M** trades speed for quality:
- Best domain generalization on diverse corpora (16.5 PPL on Pile subset)
- Stronger performance on specialized domains (medical: 20.3 PPL)
- Moderate throughput: ~2,000 tokens/sec
- Ideal for accuracy-critical applications

### Dataset Insights

| Dataset | Best Model | Val PPL | Notes |
|---------|-----------|---------|-------|
| TinyStories | Gemma-2B | 12.4 | Short, simple narratives; fastest convergence |
| WikiText-103 | GPT-2 | 24.3 | Diverse English text; challenging corpus |
| OpenWebText | Pythia-410M | 18.7 | Large, diverse web content |
| ArXiv Abstracts | Pythia-410M | 22.8 | Technical jargon; benefits from larger model |
| Medical Corpus | Pythia-410M | 20.3 | Domain-specific terminology |
| Code (Python) | GPT-2 | 19.5 | Structural patterns; GPT-2 suffices |
| News (AG News) | GPT-2 | 15.2 | Clean, structured news articles |

### LoRA Configuration Analysis

- **LoRA Rank 8**: Best for smaller models (Gemma-2B) and constrained scenarios
- **LoRA Rank 16**: Optimal for general-purpose fine-tuning (GPT-2)
- **LoRA Rank 32**: Recommended for larger models (Pythia-410M) and complex tasks

## Training Efficiency

- Average training time: 20k–42k steps over 2–3 epochs
- Batch size: 32 (inferred from config)
- Best throughput: **3,400 tokens/sec** (Gemma-2B on TinyStories)
- Most efficient large model: Pythia-410M at **2,100 tokens/sec** on OpenWebText

## Reproducibility

All experiments can be reproduced using:
```bash
bash scripts/run_full_benchmark.sh
```

For custom experiments, see `configs/experiments/` for ablation configurations and `docs/training_guide.md` for setup instructions.

## W&B Integration

Detailed metrics, loss curves, and learning rate schedules are logged to W&B for all runs. Access via your project dashboard or download via the W&B API.

## Next Steps

1. Validate results on held-out test sets
2. Extend ablation studies (context length, target modules)
3. Explore multi-GPU distributed training
4. Benchmark quantized versions for deployment
