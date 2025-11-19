"""Inference benchmarking"""

import time
import torch
import numpy as np

class InferenceBenchmark:
    """Benchmark inference speed"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def benchmark(self, prompts, num_runs=10):
        """Run benchmark"""
        self.model.eval()
        
        latencies = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Warmup
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_length=50)
            
            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=50)
                
                latency = time.time() - start_time
                latencies.append(latency)
        
        return {
            'mean_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }

def benchmark_inference(model, tokenizer, prompts, num_runs=10, device="cuda"):
    """Convenience function"""
    benchmark = InferenceBenchmark(model, tokenizer, device)
    return benchmark.benchmark(prompts, num_runs)