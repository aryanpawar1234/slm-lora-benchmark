"""Prepare and cache datasets"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from datasets import load_dataset

def main():
    print("Preparing datasets...")
    
    config = load_config("configs/datasets.yaml")
    datasets = config["datasets"]
    
    for name, dataset_config in datasets.items():
        print(f"\nCaching {name}...")
        try:
            if dataset_config.get("subset"):
                load_dataset(
                    dataset_config["name"],
                    dataset_config["subset"],
                    split=dataset_config["split"]
                )
            else:
                load_dataset(
                    dataset_config["name"],
                    split=dataset_config["split"]
                )
            print(f"✓ {name} cached")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\nAll datasets prepared!")

if __name__ == "__main__":
    main()