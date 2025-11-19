"""Evaluation metrics"""

import math
import torch
from typing import Dict

def compute_ppl(loss):
    """Compute perplexity"""
    return math.exp(loss) if loss < 100 else float('inf')

def compute_bpt(loss):
    """Compute bits per token"""
    return loss / math.log(2)

def compute_metrics(outputs, labels):
    """Compute all metrics"""
    loss = outputs.loss.item()
    
    return {
        'loss': loss,
        'ppl': compute_ppl(loss),
        'bpt': compute_bpt(loss)
    }