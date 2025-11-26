import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class DocumentCleaner:
    """
    A class to clean and normalize text chunks for better readability and consistency.
    """
    
    def __init__(self):
        # Compile regex patterns for better performance
        self.newline_pattern = re.compile(r'\s*\n\s*')
        self.multi_space_pattern = re.compile(r'\s{2,}')
        self.bullet_pattern = re.compile(r'^[\s•\-*]\s*')
        self.leading_trailing_space = re.compile(r'^\s+|\s+$')
        self.page_number_pattern = re.compile(r'\b(?:page|p|pg)\.?\s*\d+\b', re.IGNORECASE)
        self.header_footer_pattern = re.compile(
            r'^(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,}|'
            r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}|'
            r'confidential|private|draft|preliminary|'
            r'©|copyright|all rights? reserved|page\s+\d+\s+of\s+\d+)',
            re.IGNORECASE | re.MULTILINE
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the given text chunk.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return text
            
        # Remove common header/footer content
        text = self.header_footer_pattern.sub('', text)
        
        # Replace newlines and multiple spaces with single space
        text = self.newline_pattern.sub(' ', text)
        text = self.multi_space_pattern.sub(' ', text)
        
        # Clean up bullet points
        text = self.bullet_pattern.sub('', text)
        
        # Remove page numbers and references
        text = self.page_number_pattern.sub('', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation if missing
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase words
        
        # Remove leading/trailing spaces
        text = self.leading_trailing_space.sub('', text)
        
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?] )', text)
        text = ''.join([sentences[i].capitalize() if i == 0 or 
                       (i > 0 and len(sentences[i-1]) > 0 and 
                        sentences[i-1][-1] in '.!?') 
                       else sentences[i] 
                       for i in range(len(sentences))])
        
        return text.strip()
    
    def clean_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single chunk of text with metadata.
        
        Args:
            chunk: Dictionary containing 'text' and 'metadata' keys
            
        Returns:
            Cleaned chunk dictionary
        """
        if not isinstance(chunk, dict) or 'text' not in chunk:
            return chunk
            
        cleaned_chunk = chunk.copy()
        cleaned_chunk['text'] = self.clean_text(chunk['text'])
        
        # Clean metadata if it exists
        if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
            cleaned_metadata = chunk['metadata'].copy()
            
            # Clean any string fields in metadata
            for key, value in cleaned_metadata.items():
                if isinstance(value, str):
                    cleaned_metadata[key] = self.clean_text(value)
            
            cleaned_chunk['metadata'] = cleaned_metadata
            
        return cleaned_chunk
    
    def clean_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of cleaned chunk dictionaries
        """
        return [self.clean_chunk(chunk) for chunk in chunks if chunk.get('text')]


# Global instance for convenience
document_cleaner = DocumentCleaner()

# Convenience functions
def clean_text(text: str) -> str:
    """Clean a single text string."""
    return document_cleaner.clean_text(text)

def clean_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single chunk."""
    return document_cleaner.clean_chunk(chunk)

def clean_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a list of chunks."""
    return document_cleaner.clean_chunks(chunks)
