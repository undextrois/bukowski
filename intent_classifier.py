"""
Intent Classifier Component
Lightweight implementation using DistilBERT for M1 Mac optimization
"""

import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    pipeline
)
import numpy as np
from typing import Dict, List, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Lightweight intent classifier for financial queries"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the intent classifier"""
        self.config = config
        self.model_name = config.get('intent_model', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 128)
        self.use_gpu = config.get('use_gpu', False) and torch.cuda.is_available()
        
        # Define financial intents
        self.intents = {
            'account_balance': [
                "How much money do I have?",
                "What's my current balance?",
                "Show me my account balance",
                "How much is in my account?"
            ],
            'spending_analysis': [
                "How much did I spend on groceries?",
                "What are my monthly expenses?",
                "Show me my spending breakdown",
                "Where does my money go?"
            ],
            'future_balance': [
                "What will my balance be in 30 days?",
                "How much will I have next month?",
                "Calculate my future savings",
                "Project my balance"
            ],
            'investment_advice': [
                "How can I invest in stocks?",
                "What should I invest in?",
                "Tell me about mutual funds",
                "Investment strategies"
            ],
            'financial_education': [
                "What is compound interest?",
                "Explain 401k",
                "What are bonds?",
                "How does the stock market work?"
            ],
            'budget_planning': [
                "Help me create a budget",
                "How much should I save?",
                "Budget recommendations",
                "Financial planning advice"
            ]
        }
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.classifier = None
        
        # Load or create model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model"""
        try:
            # Use a lightweight approach with text classification pipeline
            device = 0 if self.use_gpu else -1
            
            # For POC, we'll use zero-shot classification which is more flexible
            # and doesn't require training on specific intents initially
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",  # Lightweight alternative
                device=device
            )
            
            # Prepare candidate labels from our intents
            self.candidate_labels = list(self.intents.keys())
            
            logger.info(f"Intent classifier initialized with {len(self.candidate_labels)} intents")
            
        except Exception as e:
            logger.error(f"Error initializing intent classifier: {e}")
            # Fallback to rule-based classification
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback rule-based classifier"""
        logger.warning("Using fallback rule-based classifier")
        self.classifier = None
        
        # Create keyword mappings for fallback
        self.keyword_mapping = {
            'account_balance': ['balance', 'money', 'account', 'have', 'current'],
            'spending_analysis': ['spend', 'spent', 'expenses', 'cost', 'paid'],
            'future_balance': ['future', 'will', 'next', 'days', 'months', 'project'],
            'investment_advice': ['invest', 'stocks', 'mutual', 'portfolio', 'buy'],
            'financial_education': ['what is', 'explain', 'how does', 'what are', 'compound'],
            'budget_planning': ['budget', 'save', 'saving', 'plan', 'planning']
        }
    
    async def classify(self, text: str) -> Dict[str, Any]:
        """Classify the intent of input text"""
        try:
            if self.classifier is not None:
                # Use transformer-based classification
                result = self.classifier(text, self.candidate_labels)
                
                return {
                    'intent': result['labels'][0],
                    'confidence': result['scores'][0],
                    'all_scores': dict(zip(result['labels'], result['scores']))
                }
            else:
                # Use fallback rule-based classification
                return self._classify_fallback(text)
                
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based classification"""
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in self.keyword_mapping.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            # Normalize score
            scores[intent] = score / len(keywords) if keywords else 0
        
        if not scores or max(scores.values()) == 0:
            return {
                'intent': 'financial_education',  # Default to education
                'confidence': 0.3,
                'all_scores': scores
            }
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        # Boost confidence if multiple keywords match
        if confidence > 0:
            confidence = min(0.9, confidence + 0.3)
        
        return {
            'intent': best_intent,
            'confidence': confidence,
            'all_scores': scores
        }
    
    def add_training_data(self, intent: str, examples: List[str]):
        """Add training examples for a specific intent"""
        if intent not in self.intents:
            self.intents[intent] = []
        
        self.intents[intent].extend(examples)
        
        # Update candidate labels
        self.candidate_labels = list(self.intents.keys())
        
        logger.info(f"Added {len(examples)} examples for intent '{intent}'")
    
    def save_intents(self, filepath: str = "intents.json"):
        """Save current intents to file"""
        with open(filepath, 'w') as f:
            json.dump(self.intents, f, indent=2)
        
        logger.info(f"Intents saved to {filepath}")
    
    def load_intents(self, filepath: str = "intents.json"):
        """Load intents from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.intents = json.load(f)
            
            self.candidate_labels = list(self.intents.keys())
            logger.info(f"Loaded intents from {filepath}")
        else:
            logger.warning(f"Intent file {filepath} not found, using defaults")
    
    def get_intent_examples(self, intent: str) -> List[str]:
        """Get training examples for a specific intent"""
        return self.intents.get(intent, [])
    
    def list_intents(self) -> List[str]:
        """Get list of all available intents"""
        return list(self.intents.keys())