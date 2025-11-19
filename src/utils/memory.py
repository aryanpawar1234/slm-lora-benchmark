"""Memory optimization utilities"""

import gc
import torch
from typing import Optional


def print_memory_stats(device: Optional[str] = None):
    """
    Print GPU memory statistics
    
    Args:
        device: CUDA device (None for current device)
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    
    print("=" * 60)
    print("GPU MEMORY STATS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Max Allocated: {max_allocated:.2f} GB")
    print("=" * 60)


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cache cleared")


def get_memory_usage(device: Optional[str] = None) -> dict:
    """
    Get GPU memory usage statistics
    
    Args:
        device: CUDA device
        
    Returns:
        Dictionary with memory stats
    """
    if not torch.cuda.is_available():
        return {}
    
    if device is None:
        device = torch.cuda.current_device()
    
    return {
        'allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved(device) / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024**3,
        'total_gb': torch.cuda.get_device_properties(device).total_memory / 1024**3
    }


def optimize_memory():
    """Apply memory optimization techniques"""
    # Enable memory efficient attention
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for memory efficiency")
    except:
        pass
    
    # Clear cache
    clear_memory()


class MemoryTracker:
    """Track memory usage during training"""
    
    def __init__(self):
        self.snapshots = []
    
    def snapshot(self, tag: str = ""):
        """Take memory snapshot"""
        if torch.cuda.is_available():
            stats = get_memory_usage()
            stats['tag'] = tag
            self.snapshots.append(stats)
    
    def print_summary(self):
        """Print memory usage summary"""
        if not self.snapshots:
            print("No memory snapshots")
            return
        
        print("=" * 60)
        print("MEMORY USAGE SUMMARY")
        print("=" * 60)
        
        for snapshot in self.snapshots:
            tag = snapshot.get('tag', 'Unknown')
            allocated = snapshot.get('allocated_gb', 0)
            print(f"{tag}: {allocated:.2f} GB")
        
        print("=" * 60)
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage"""
        if not self.snapshots:
            return 0
        return max(s.get('allocated_gb', 0) for s in self.snapshots)