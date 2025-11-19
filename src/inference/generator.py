"""Text generation"""

import torch

class TextGenerator:
    """Text generator"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_p=0.9):
        """Generate text"""
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

def generate_text(model, tokenizer, prompt, max_length=100, device="cuda"):
    """Convenience function"""
    generator = TextGenerator(model, tokenizer, device)
    return generator.generate(prompt, max_length)
