"""
Entity Extractor Component
Lightweight NER for financial entities optimized for M1 Mac
"""

import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract financial entities from user messages"""
    
    def __init__(self):
        """Initialize entity extractor with lightweight NLP model"""
        self.nlp = None
        self._initialize_nlp()
        
        # Define regex patterns for common financial entities
        self.patterns = {
            'money': [
                r'\$[\d,]+\.?\d*',  # $1,000.50
                r'[\d,]+\.?\d*\s*dollars?',  # 1000 dollars
                r'[\d,]+\.?\d*\s*bucks?',  # 500 bucks
            ],
            'percentage': [
                r'[\d.]+\s*%',  # 5.5%
                r'[\d.]+\s*percent',  # 5.5 percent
            ],
            'date': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 12/31/2024
                r'\d{1,2}\s+\w+\s+\d{2,4}',  # 31 December 2024
            ],
            'time_period': [
                r'\d+\s*days?',  # 30 days
                r'\d+\s*weeks?',  # 2 weeks
                r'\d+\s*months?',  # 6 months
                r'\d+\s*years?',  # 1 year
            ],
            'category': [
                r'\bgroceries?\b',
                r'\brent\b',
                r'\bgasoline?\b',
                r'\bfood\b',
                r'\bentertainment\b',
                r'\butilities?\b',
                r'\binsurance\b',
                r'\btransportation\b',
            ]
        }
        
        # Define financial account types
        self.account_types = [
            'checking', 'savings', 'credit', 'investment', 'retirement',
            '401k', 'ira', 'roth', 'brokerage'
        ]
        
        # Define investment types
        self.investment_types = [
            'stocks', 'bonds', 'mutual funds', 'etf', 'cryptocurrency',
            'real estate', 'commodities', 'options'
        ]
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model (lightweight)"""
        try:
            # Try to load a lightweight English model
            # If not available, will fall back to regex-only extraction
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            logger.warning("Using regex-only entity extraction")
            self.nlp = None
    
    async def extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from text"""
        entities = {
            'money': [],
            'percentage': [],
            'date': [],
            'time_period': [],
            'category': [],
            'account_type': [],
            'investment_type': [],
            'number': [],
            'person': [],
            'organization': []
        }
        
        try:
            # Extract using regex patterns
            self._extract_with_regex(text, entities)
            
            # Extract using spaCy if available
            if self.nlp:
                self._extract_with_spacy(text, entities)
            
            # Extract specialized financial entities
            self._extract_financial_entities(text, entities)
            
            # Clean and deduplicate entities
            entities = self._clean_entities(entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return entities
    
    def _extract_with_regex(self, text: str, entities: Dict[str, List]):
        """Extract entities using regex patterns"""
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities[entity_type].append({
                        'value': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8,
                        'method': 'regex'
                    })
    
    def _extract_with_spacy(self, text: str, entities: Dict[str, List]):
        """Extract entities using spaCy NER"""
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                entities['money'].append({
                    'value': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9,
                    'method': 'spacy'
                })
            elif ent.label_ == "PERCENT":
                entities['percentage'].append({
                    'value': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9,
                    'method': 'spacy'
                })
            elif ent.label_ == "DATE":
                entities['date'].append({
                    'value': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9,
                    'method': 'spacy'
                })
            elif ent.label_ == "PERSON":
                entities['person'].append({
                    'value': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9,
                    'method': 'spacy'
                })
            elif ent.label_ == "ORG":
                entities['organization'].append({
                    'value': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9,
                    'method': 'spacy'
                })
    
    def _extract_financial_entities(self, text: str, entities: Dict[str, List]):
        """Extract specialized financial entities"""
        text_lower = text.lower()
        
        # Extract account types
        for account_type in self.account_types:
            if account_type in text_lower:
                start_idx = text_lower.find(account_type)
                entities['account_type'].append({
                    'value': account_type,
                    'start': start_idx,
                    'end': start_idx + len(account_type),
                    'confidence': 0.85,
                    'method': 'financial_patterns'
                })
        
        # Extract investment types
        for investment_type in self.investment_types:
            if investment_type in text_lower:
                start_idx = text_lower.find(investment_type)
                entities['investment_type'].append({
                    'value': investment_type,
                    'start': start_idx,
                    'end': start_idx + len(investment_type),
                    'confidence': 0.85,
                    'method': 'financial_patterns'
                })
        
        # Extract standalone numbers
        number_pattern = r'\b\d+\.?\d*\b'
        for match in re.finditer(number_pattern, text):
            # Skip if already captured as money or percentage
            is_money = any(
                money_ent['start'] <= match.start() < money_ent['end'] 
                for money_ent in entities['money']
            )
            is_percentage = any(
                perc_ent['start'] <= match.start() < perc_ent['end'] 
                for perc_ent in entities['percentage']
            )
            
            if not is_money and not is_percentage:
                entities['number'].append({
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7,
                    'method': 'regex'
                })
    
    def _clean_entities(self, entities: Dict[str, List]) -> Dict[str, List]:
        """Clean and deduplicate entities"""
        cleaned = {}
        
        for entity_type, entity_list in entities.items():
            # Remove duplicates and overlapping entities
            unique_entities = []
            for entity in entity_list:
                # Check for overlaps with existing entities
                is_overlap = False
                for existing in unique_entities:
                    if self._is_overlapping(entity, existing):
                        # Keep the one with higher confidence
                        if entity['confidence'] > existing['confidence']:
                            unique_entities.remove(existing)
                        else:
                            is_overlap = True
                        break
                
                if not is_overlap:
                    unique_entities.append(entity)
            
            # Sort by position in text
            unique_entities.sort(key=lambda x: x['start'])
            cleaned[entity_type] = unique_entities
        
        return cleaned
    
    def _is_overlapping(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities overlap"""
        return not (entity1['end'] <= entity2['start'] or entity2['end'] <= entity1['start'])
    
    def normalize_money(self, money_string: str) -> Optional[float]:
        """Normalize money string to float value"""
        try:
            # Remove currency symbols and spaces
            clean_money = re.sub(r'[\$,\s]', '', money_string.lower())
            clean_money = re.sub(r'dollars?|bucks?', '', clean_money)
            
            return float(clean_money)
        except (ValueError, TypeError):
            return None
    
    def normalize_percentage(self, percentage_string: str) -> Optional[float]:
        """Normalize percentage string to float value"""
        try:
            # Remove % and percent
            clean_percentage = re.sub(r'[%\s]', '', percentage_string.lower())
            clean_percentage = re.sub(r'percent', '', clean_percentage)
            
            return float(clean_percentage)
        except (ValueError, TypeError):
            return None
    
    def normalize_time_period(self, time_string: str) -> Optional[int]:
        """Normalize time period to days"""
        try:
            # Extract number and unit
            match = re.search(r'(\d+)\s*(day|week|month|year)s?', time_string.lower())
            if not match:
                return None
            
            number = int(match.group(1))
            unit = match.group(2)
            
            # Convert to days
            if unit == 'day':
                return number
            elif unit == 'week':
                return number * 7
            elif unit == 'month':
                return number * 30  # Approximate
            elif unit == 'year':
                return number * 365  # Approximate
            
            return None
        except (ValueError, TypeError):
            return None
    
    def get_normalized_entities(self, entities: Dict[str, List]) -> Dict[str, Any]:
        """Get normalized versions of extracted entities"""
        normalized = {}
        
        # Normalize money entities
        if entities['money']:
            money_values = []
            for money_ent in entities['money']:
                normalized_value = self.normalize_money(money_ent['value'])
                if normalized_value is not None:
                    money_values.append(normalized_value)
            
            if money_values:
                normalized['money_amounts'] = money_values
                normalized['total_money'] = sum(money_values)
        
        # Normalize percentages
        if entities['percentage']:
            percentage_values = []
            for perc_ent in entities['percentage']:
                normalized_value = self.normalize_percentage(perc_ent['value'])
                if normalized_value is not None:
                    percentage_values.append(normalized_value)
            
            if percentage_values:
                normalized['percentages'] = percentage_values
        
        # Normalize time periods
        if entities['time_period']:
            time_values = []
            for time_ent in entities['time_period']:
                normalized_value = self.normalize_time_period(time_ent['value'])
                if normalized_value is not None:
                    time_values.append(normalized_value)
            
            if time_values:
                normalized['time_periods_days'] = time_values
        
        return normalized