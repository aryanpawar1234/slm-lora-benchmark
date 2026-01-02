# Contributing to SLM LoRA Benchmark

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the SLM LoRA Benchmark project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in your interactions.

## How to Contribute

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/slm-lora-benchmark.git
cd slm-lora-benchmark
git remote add upstream https://github.com/aryanpawar1234/slm-lora-benchmark.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Development Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 4. Code Style

We follow PEP 8 and use:
- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### 5. Testing

Write tests for new features and ensure all tests pass:

```bash
pytest tests/ -v --cov=src
```

### 6. Commit Guidelines

- Use clear, descriptive commit messages
- Use conventional commits format: `type(scope): description`
  - `feat(training)`: New feature
  - `fix(evaluation)`: Bug fix
  - `docs(readme)`: Documentation updates
  - `test(metrics)`: Test additions
  - `refactor(utils)`: Code refactoring

### 7. Pull Request Process

1. Update `docs/CHANGELOG.md` with your changes
2. Update relevant documentation
3. Ensure all tests pass: `pytest tests/`
4. Ensure code style compliance: `black`, `isort`, `mypy`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a Pull Request with a clear description

## Areas for Contribution

### Code Improvements
- Add docstrings and type hints
- Optimize existing code
- Add new evaluation metrics
- Improve error handling

### Features
- Support for new model architectures
- Integration with additional datasets
- Distributed training support
- Quantization and pruning methods

### Documentation
- Add tutorials and examples
- Improve API documentation
- Create troubleshooting guides
- Add architecture diagrams

### Research
- Run comprehensive experiments
- Compare with other PEFT methods
- Analyze LoRA adapter behavior
- Create benchmark reports

## Development Roadmap

- [ ] Multi-GPU and distributed training
- [ ] 4-bit and 8-bit quantization support
- [ ] Additional datasets integration
- [ ] Instruction fine-tuning support
- [ ] Web interface for inference
- [ ] Improved attention analysis tools

## Getting Help

- Check existing issues: https://github.com/aryanpawar1234/slm-lora-benchmark/issues
- Open a new issue for bugs or feature requests
- Reach out via GitHub discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to make this project better!
