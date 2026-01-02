"""Aggregate and consolidate benchmark results from multiple runs."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregates benchmark results from W&B and local logs into unified CSV."""

    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = self.output_dir / "benchmark_results.csv"

    def aggregate_from_wandb(self, project: str, entity: Optional[str] = None) -> pd.DataFrame:
        """Fetch and aggregate results from Weights & Biases."""
        try:
            import wandb
            api = wandb.Api()
            
            runs_data = []
            runs = api.runs(f"{entity}/{project}" if entity else project)
            
            for run in runs:
                run_data = {
                    'model': run.config.get('model_name', 'unknown'),
                    'dataset': run.config.get('dataset', 'unknown'),
                    'params_m': run.config.get('model_params', 0) / 1e6,
                    'lora_rank': run.config.get('lora_rank', 8),
                    'val_ppl': run.summary.get('val_ppl', None),
                    'bpt': run.summary.get('bpt', None),
                    'tokens_per_sec': run.summary.get('tokens_per_sec', None),
                    'steps': run.summary.get('steps', 0),
                    'epochs': run.config.get('num_epochs', 3),
                    'run_id': run.id,
                }
                if run_data['val_ppl'] is not None:
                    runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
        except ImportError:
            logger.warning("W&B not available. Install with: pip install wandb")
            return pd.DataFrame()

    def aggregate_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load results from existing CSV."""
        return pd.read_csv(csv_path)

    def save_aggregated(self, df: pd.DataFrame, include_timestamp: bool = True):
        """Save aggregated results to CSV."""
        # Sort by model and dataset for readability
        df = df.sort_values(['model', 'dataset']).reset_index(drop=True)
        
        # Save main results
        df.to_csv(self.results_csv, index=False)
        logger.info(f"Results saved to {self.results_csv}")
        
        # Save summary statistics
        summary = df.groupby('model').agg({
            'val_ppl': ['mean', 'min', 'max'],
            'tokens_per_sec': 'mean',
            'steps': 'mean'
        })
        summary.to_csv(self.output_dir / "summary_by_model.csv")
        logger.info(f"Summary saved to {self.output_dir / 'summary_by_model.csv'}")

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate human-readable report from results."""
        report = "\n" + "="*60 + "\n"
        report += "BENCHMARK RESULTS SUMMARY\n"
        report += "="*60 + "\n\n"
        
        report += f"Total Runs: {len(df)}\n"
        report += f"Models: {df['model'].nunique()}\n"
        report += f"Datasets: {df['dataset'].nunique()}\n\n"
        
        report += "Best Results by Model:\n"
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_run = model_df.loc[model_df['val_ppl'].idxmin()]
            report += f"  {model}: {best_run['val_ppl']:.2f} PPL on {best_run['dataset']}\n"
        
        return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--output-dir", default="outputs/results", help="Output directory")
    parser.add_argument("--csv", help="Path to results CSV to load")
    parser.add_argument("--wandb", action="store_true", help="Fetch from Weights & Biases")
    parser.add_argument("--project", default="slm-lora-benchmark", help="W&B project name")
    
    args = parser.parse_args()
    
    aggregator = ResultsAggregator(args.output_dir)
    
    if args.csv:
        df = aggregator.aggregate_from_csv(args.csv)
    elif args.wandb:
        df = aggregator.aggregate_from_wandb(args.project)
    else:
        df = pd.DataFrame()
    
    if not df.empty:
        aggregator.save_aggregated(df)
        report = aggregator.generate_report(df)
        print(report)
    else:
        logger.error("No results to aggregate")
