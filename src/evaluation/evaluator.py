"""Model evaluator"""

import torch
from tqdm import tqdm

class Evaluator:
    """Model evaluator"""
    
    def __init__(self, model, dataloader, device="cuda"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    def evaluate(self):
        """Run evaluation"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                n_tokens = (batch['labels'] != -100).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        
        avg_loss = total_loss / total_tokens
        
        from .metrics import compute_ppl, compute_bpt
        
        return {
            'loss': avg_loss,
            'ppl': compute_ppl(avg_loss),
            'bpt': compute_bpt(avg_loss)
        }

def evaluate_model(model, dataloader, device="cuda"):
    """Convenience function"""
    evaluator = Evaluator(model, dataloader, device)
    return evaluator.evaluate()
