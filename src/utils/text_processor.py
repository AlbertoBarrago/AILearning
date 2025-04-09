"""Text processor for handling text preprocessing operations."""

from typing import List
import re

class TextProcessor:
    def __init__(self):
        """Initialize the text processor."""
        pass

    @staticmethod
    def preprocess(text: str) -> str:
        """Preprocess the input text.

        Args:
            text: Input text to process

        Returns:
            Processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split text into tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        # Simple word-based tokenization
        return text.split()