# Ablation Studies Configuration

This directory contains YAML configs for ablation studies systematically varying key LoRA hyperparameters.

## Available Ablation Configs

### 1. LoRA Rank Ablation (`abl_lora_rank.yaml`)
Varies the LoRA rank (r parameter) to understand capacity-accuracy tradeoffs.

**Configurations:**
- `lora_rank: 4` - Ultra-lightweight, minimal overhead
- `lora_rank: 8` - Small/medium model default
- `lora_rank: 16` - General-purpose (baseline)
- `lora_rank: 32` - High-capacity for complex tasks
- `lora_rank: 64` - Very high-capacity (may overfit)

**Expected Findings:**
Higher rank improves validation perplexity (up to a point) but increases training time and memory. Typical sweet spot: rank 8-16 for most models.

### 2. Target Modules Ablation (`abl_target_modules.yaml`)
Varies which transformer modules receive LoRA adaptations.

**Configurations:**
- `target_modules: [q_proj, v_proj]` - Attention query/value only (lightweight)
- `target_modules: [q_proj, k_proj, v_proj, out_proj]` - All attention (moderate)
- `target_modules: [q_proj, v_proj, fc1, fc2]` - Attention + MLP (comprehensive)
- `target_modules: [q_proj, k_proj, v_proj, out_proj, fc1, fc2]` - Full coverage (heavy)

**Expected Findings:**
More target modules generally improve task performance but increase parameters. Typically, attention modules (q_proj, v_proj) are most important.

### 3. Context Length Ablation (`abl_context_length.yaml`)
Varies input sequence length to study length generalization and efficiency.

**Configurations:**
- `max_position_embeddings: 256` - Short sequences, fast training
- `max_position_embeddings: 512` - Standard length
- `max_position_embeddings: 1024` - Long-form understanding
- `max_position_embeddings: 2048` - Extended context (memory intensive)

**Expected Findings:**
Longer contexts enable understanding of broader dependencies but slow down training and increase memory usage. Most datasets benefit from 512-1024 tokens.

## Running Ablation Studies

### Single Ablation
```bash
python scripts/train.py --config configs/experiments/abl_lora_rank.yaml --lora_rank 16
```

### Ablation Sweep (all configurations)
```bash
for config in abl_*.yaml; do
  python scripts/train.py --config configs/experiments/$config
done
```

### Track with W&B
```bash
WANDB_PROJECT=slm-lora-ablations python scripts/train.py --config configs/experiments/abl_lora_rank.yaml
```

## Analysis and Interpretation

After running ablations, use the provided analysis notebooks:
- `notebooks/02_ablation_analysis.ipynb` - Visualize ablation results
- Compare perplexity, training time, and memory across configurations
- Identify optimal hyperparameter combinations for your use case

## Recommended Reading Order

1. Start with **LoRA Rank** - understanding capacity-accuracy tradeoff
2. Then **Target Modules** - where to apply LoRA most effectively
3. Finally **Context Length** - for task-specific optimization
