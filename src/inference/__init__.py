"""Inference modules"""

from .generator import TextGenerator, generate_text
from .benchmark import InferenceBenchmark, benchmark_inference

__all__ = ["TextGenerator", "generate_text", "InferenceBenchmark", "benchmark_inference"]
