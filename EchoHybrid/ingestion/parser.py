import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import PyPDF2
import docx
import json as json_module
from .redactor import DocumentRedactor, RedactionRule, redact_content

class DocumentParser:
    """Parser for various document types that extracts text content and saves to JSON."""
    
    def __init__(self, redact_sensitive: bool = True, custom_rules: Optional[List[RedactionRule]] = None):
        """
        Initialize the document parser with optional redaction.
        
        Args:
            redact_sensitive: Whether to redact sensitive information
            custom_rules: Optional list of custom redaction rules
        """
        self.redact_sensitive = redact_sensitive
        self.redactor = DocumentRedactor(custom_rules) if redact_sensitive else None
    
    @staticmethod
    def parse_txt(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse text file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'file_type': 'txt',
            'content': content,
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
        }

    @staticmethod
    def parse_pdf(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse PDF file and extract text content."""
        text = []
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
            
        return {
            'file_type': 'pdf',
            'content': '\n\n'.join(text),
            'metadata': {
                'file_name': os.path.basename(file_path),
                'page_count': len(text),
                'file_size': os.path.getsize(file_path)
            }
        }

    @staticmethod
    def parse_docx(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse DOCX file and extract text content."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {str(e)}")
            
        return {
            'file_type': 'docx',
            'content': '\n\n'.join(paragraphs),
            'metadata': {
                'file_name': os.path.basename(file_path),
                'paragraph_count': len(paragraphs),
                'file_size': os.path.getsize(file_path)
            }
        }

    @staticmethod
    def parse_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse JSON file and extract content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json_module.load(f)
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")
            
        return {
            'file_type': 'json',
            'content': content,
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
        }

    @staticmethod
    def parse_markdown(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse Markdown file and extract text content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Error reading Markdown file: {str(e)}")
            
        return {
            'file_type': 'md',
            'content': content,
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
        }

    def parse_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse document based on file extension.
        
        Args:
            file_path: Path to the document to parse
            
        Returns:
            Dictionary containing parsed content and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = file_path.suffix.lower()
        
        # Parse the document
        if ext == '.pdf':
            parsed_data = self.parse_pdf(file_path)
        elif ext == '.docx':
            parsed_data = self.parse_docx(file_path)
        elif ext == '.json':
            parsed_data = self.parse_json(file_path)
        elif ext in ['.md', '.markdown']:
            parsed_data = self.parse_markdown(file_path)
        elif ext == '.txt':
            parsed_data = self.parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Apply redaction if enabled
        if self.redact_sensitive and 'content' in parsed_data:
            parsed_data['content'] = self.redactor.redact_text(parsed_data['content'])
            
        return (parsed_data)

    @staticmethod
    def get_output_path(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Determine the output path for the parsed file with timestamp.
        
        Args:
            input_path: Path to the input file
            output_path: Optional custom output path
            
        Returns:
            Path object for the output file with timestamp
        """
        from datetime import datetime
        
        input_path = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            # Default to parsed_output folder in the echohybrid directory
            project_root = Path(__file__).parent.parent  # Go up two levels to reach echohybrid
            output_dir = project_root / 'parsed_output'
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"{input_path.stem}_{timestamp}.json"
            
        output_path = Path(output_path)
        if output_path.is_dir():
            return output_path / f"{input_path.stem}_{timestamp}.json"
            
        # If output_path is a file, insert timestamp before the extension
        return output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"

    @classmethod
    def save_to_json(cls, data: Dict[str, Any], output_path: Optional[Union[str, Path]] = None, 
                    input_path: Optional[Union[str, Path]] = None) -> Path:
        """Save parsed content to JSON file in the parsed_output folder.
        
        Args:
            data: The parsed data to save
            output_path: Optional custom output path
            input_path: Optional input path used to determine default output path
            
        Returns:
            Path to the saved file
        """
        if output_path is None and input_path is None:
            raise ValueError("Either output_path or input_path must be provided")
            
        final_output_path = cls.get_output_path(input_path or output_path, output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved parsed content to: {final_output_path}")
        return final_output_path


def parse_and_save(
    input_path: str, 
    output_path: Optional[str] = None,
    redact_sensitive: bool = True,
    custom_rules: Optional[List[RedactionRule]] = None
) -> Path:
    """
    Parse a document and save its content to a JSON file in the parsed_output folder.
    
    Args:
        input_path: Path to the input document
        output_path: Optional custom output path. If not provided, will save to:
                   ./parsed_output/{input_filename}.json
                   
    Returns:
        Path to the saved JSON file
    """
    try:
        # Initialize parser with redaction settings
        parser = DocumentParser(redact_sensitive=redact_sensitive, custom_rules=custom_rules)
        
        # Parse the document
        parsed_data = parser.parse_document(input_path)
        
        # Save to parsed_output folder by default
        saved_path = parser.save_to_json(
            data=parsed_data,
            output_path=output_path,
            input_path=input_path
        )
        
        return saved_path
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        parse_and_save(input_file, output_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
