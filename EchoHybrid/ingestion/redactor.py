import re
from typing import Dict, Any, List, Pattern, Union, Optional
from dataclasses import dataclass
import json

@dataclass
class RedactionRule:
    name: str
    pattern: Pattern
    replacement: str
    description: str = ""

class DocumentRedactor:
    """
    Handles redaction of sensitive information from text content.
    """
    
    def __init__(self, redaction_rules: Optional[List[RedactionRule]] = None):
        """
        Initialize with default or custom redaction rules.
        """
        self.rules = redaction_rules or self.get_default_rules()
    
    @staticmethod
    def get_default_rules() -> List[RedactionRule]:
        """
        Returns a list of default redaction rules for common sensitive data.
        """
        return [
            # Email addresses
            RedactionRule(
                name="email",
                pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                replacement="[REDACTED_EMAIL]",
                description="Redacts email addresses"
            ),
            # Phone numbers (various formats)
            RedactionRule(
                name="phone",
                pattern=re.compile(r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                replacement="[REDACTED_PHONE]",
                description="Redacts phone numbers"
            ),
            # Copyright notices
            RedactionRule(
                name="copyright",
                pattern=re.compile(r'©|\bCopyright\b[\s\S]*?\d{4}\b', re.IGNORECASE),
                replacement="[REDACTED_COPYRIGHT]",
                description="Redacts copyright notices"
            ),
            # Credit card numbers
            RedactionRule(
                name="credit_card",
                pattern=re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11})\b'),
                replacement="[REDACTED_CREDIT_CARD]",
                description="Redacts credit card numbers"
            ),
            # SSN (US Social Security Numbers)
            RedactionRule(
                name="ssn",
                pattern=re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),
                replacement="[REDACTED_SSN]",
                description="Redacts Social Security Numbers"
            ),
            # API keys (common patterns)
            RedactionRule(
                name="api_key",
                pattern=re.compile(r'\b(sk_|pk_|AKIA|A3T|ABIA|ACCA|AGPA|AIDA|AIPA|AKIA|ANPA|ANVA|APKA|AROA|ASCA|ASIA)[A-Z0-9]{16,}\b'),
                replacement="[REDACTED_API_KEY]",
                description="Redacts API keys"
            )
        ]
    
    def add_rule(self, rule: RedactionRule) -> None:
        """Add a custom redaction rule."""
        self.rules.append(rule)
    
    def redact_text(self, text: str) -> str:
        """
        Apply all redaction rules to the given text.
        
        Args:
            text: The text to redact
            
        Returns:
            Redacted text with sensitive information replaced
        """
        if not text or not isinstance(text, str):
            return text
            
        redacted = text
        for rule in self.rules:
            redacted = rule.pattern.sub(rule.replacement, redacted)
        
        return redacted
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively redact values in a dictionary.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.redact_text(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [self.redact_dict(v) if isinstance(v, dict) else 
                             (self.redact_text(v) if isinstance(v, str) else v) 
                             for v in value]
            else:
                result[key] = value
        return result

def redact_content(content: Union[str, Dict[str, Any]], rules: Optional[List[RedactionRule]] = None) -> Union[str, Dict[str, Any]]:
    """
    Convenience function to redact content.
    
    Args:
        content: Text content or dictionary to redact
        rules: Optional list of custom redaction rules
        
    Returns:
        Redacted content of the same type as input
    """
    redactor = DocumentRedactor(rules)
    
    if isinstance(content, str):
        return redactor.redact_text(content)
    elif isinstance(content, dict):
        return redactor.redact_dict(content)
    return content