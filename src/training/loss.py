"""Loss computation functions"""

import torch
import math


def compute_token_loss(outputs, labels):
    """Compute token-level loss"""
    return outputs.loss.item()


def compute_sequence_loss(outputs, labels):
    """Compute sequence-level loss"""
    return outputs.loss.item()


def compute_perplexity(loss):
    """Compute perplexity from loss"""
    return math.exp(loss) if loss < 100 else float('inf')


def compute_bits_per_token(loss):
    """Compute bits per token"""
    return loss / math.log(2)