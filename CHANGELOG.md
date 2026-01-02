# Changelog

All notable changes to the SLM LoRA Benchmark project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive GitHub Actions CI/CD workflows (tests, linting, type checking)
- Contributing guidelines (CONTRIBUTING.md)
- Changelog tracking (CHANGELOG.md)
- Docker setup files for containerized development
- Enhanced API documentation with docstrings
- Experiment results with real benchmark data
- Multi-GPU support infrastructure
- 4-bit and 8-bit quantization support
- Issue templates for bug reports and feature requests
- Pull request template
- GitHub discussions setup
- Comprehensive test suite expansion
- ONNX export support for inference optimization

### Changed
- Updated README with complete feature list and real results
- Improved error handling and logging across modules
- Enhanced configuration validation
- Optimized data loading pipeline

### Fixed
- Reproducibility improvements with seed management
- Memory leak fixes in training loop
- Data loading edge cases

## [1.0.0] - 2025-01-02

### Added
- Initial stable release of SLM LoRA Benchmark framework
- Complete training pipeline for small language models
- Support for 10-15 curated datasets from Hugging Face
- LoRA (Low-Rank Adaptation) fine-tuning implementation
- Comprehensive metrics tracking (20+ metrics)
- Weights & Biases integration for experiment tracking
- Hyperparameter sweep support
- GPU-optimized training (T4/V100/A100 compatible)
- Inference benchmarking tools
- LaTeX research report generation
- Jupyter notebooks for analysis and visualization
- Unit tests for core components

### Features
- **Training Metrics**: Token loss, sequence loss, perplexity (PPL), bits-per-token (BPT), learning rate, gradient norm, throughput
- **Evaluation Metrics**: Validation loss, PPL, BPT, PPL by sequence length buckets, ROC/Delta-PPL
- **Inference**: Latency measurement, tokens per second throughput, memory usage tracking, batch size optimization
- **Models Supported**: Gemma-2B, GPT-2, Pythia-410M, and other HuggingFace models
- **Datasets**: WikiText, OpenWebText, BookCorpus, TinyStories, AG News, Yelp Reviews, Amazon Reviews, CNN/DailyMail, DailyDialog, ELI5, SQuAD, AI2 ARC

### Documentation
- Comprehensive README with quick start guide
- Dataset details documentation
- Model details documentation
- Training guide
- W&B integration guide
- Hyperparameter tuning guide
- Troubleshooting guide

### Project Structure
- `src/`: Modular source code (data, models, training, evaluation, inference, utils)
- `scripts/`: Executable scripts for training, evaluation, inference, sweeps
- `configs/`: YAML configuration files for models, datasets, training, LoRA, W&B
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `tests/`: Unit tests for core functionality
- `docs/`: Detailed documentation
- `report/`: LaTeX research report template

[Unreleased]: https://github.com/aryanpawar1234/slm-lora-benchmark/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aryanpawar1234/slm-lora-benchmark/releases/tag/v1.0.0
