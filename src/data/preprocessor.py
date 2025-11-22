"""Text preprocessing utilities"""

import re
from typing import List, Optional


class TextPreprocessor:
    """Text preprocessing for language modeling"""
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_special_chars: bool = False,
        min_length: int = 32,
    ):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
    
    def preprocess(self, text: str) -> Optional[str]:
        """Preprocess single text"""
        if not text or not isinstance(text, str):
            return None
        
        if self.lowercase:
            text = text.lower()
        
        if len(text) < self.min_length:
            return None
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess batch of texts"""
        preprocessed = []
        for text in texts:
            processed = self.preprocess(text)
            if processed:
                preprocessed.append(processed)
        return preprocessed
