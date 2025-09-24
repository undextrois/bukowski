"""
Model Trainer Component
Lightweight training pipeline for custom financial domain adaptation
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import json
import os
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FinancialIntentDataset(Dataset):
    """Custom dataset for financial intent classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelTrainer:
    """Train and fine-tune models for financial domain"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer"""
        self.config = config
        self.epochs = config.get('epochs', 3)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.batch_size = config.get('batch_size', 8)  # Small batch size for M1 Mac
        self.warmup_steps = config.get('warmup_steps', 100)
        self.max_length = config.get('max_length', 128)
        
        # Model configuration
        self.model_name = config.get('base_model', 'distilbert-base-uncased')
        self.output_dir = config.get('output_dir', './trained_models')
        
        # Training data storage
        self.training_data = []
        self.intent_to_id = {}
        self.id_to_intent = {}
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_intent(self, intent_name: str, examples: List[str]):
        """Add training examples for a specific intent"""
        if intent_name not in self.intent_to_id:
            intent_id = len(self.intent_to_id)
            self.intent_to_id[intent_name] = intent_id
            self.id_to_intent[intent_id] = intent_name
        
        intent_id = self.intent_to_id[intent_name]
        
        for example in examples:
            self.training_data.append({
                'text': example,
                'intent': intent_name,
                'label': intent_id
            })
        
        logger.info(f"Added {len(examples)} examples for intent '{intent_name}'")
    
    def load_training_data(self, file_path: str) -> bool:
        """Load training data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'intents' in data:
                # Format: {"intents": {"intent_name": ["example1", "example2"]}}
                for intent_name, examples in data['intents'].items():
                    self.add_intent(intent_name, examples)
            elif isinstance(data, list):
                # Format: [{"text": "example", "intent": "intent_name"}]
                for item in data:
                    if 'text' in item and 'intent' in item:
                        self.add_intent(item['intent'], [item['text']])
            
            logger.info(f"Loaded training data from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return False
    
    def prepare_default_training_data(self):
        """Prepare default financial training data"""
        default_intents = {
            'account_balance': [
                "What's my current balance?",
                "How much money do I have?",
                "Show me my account balance",
                "What's in my checking account?",
                "How much do I have available?",
                "Check my balance please",
                "What's my total balance?",
                "How much is in my accounts?",
                "Balance inquiry",
                "Account balance check"
            ],
            'spending_analysis': [
                "How much did I spend this month?",
                "What are my expenses?",
                "Show me my spending breakdown",
                "Where did my money go?",
                "How much did I spend on groceries?",
                "What's my biggest expense?",
                "Analyze my spending",
                "Monthly expenses report",
                "Track my spending",
                "Expense analysis"
            ],
            'future_balance': [
                "What will my balance be next month?",
                "Project my future savings",
                "How much will I have in 30 days?",
                "Forecast my account balance",
                "Calculate future balance",
                "What's my projected balance?",
                "Estimate balance in 60 days",
                "Future account projection",
                "Balance prediction",
                "Financial forecast"
            ],
            'investment_advice': [
                "How should I invest my money?",
                "What stocks should I buy?",
                "Investment recommendations",
                "How to start investing?",
                "What about mutual funds?",
                "Should I invest in ETFs?",
                "Investment portfolio advice",
                "Stock market guidance",
                "Investment strategy help",
                "How to build wealth?"
            ],
            'financial_education': [
                "What is compound interest?",
                "Explain 401k to me",
                "How do bonds work?",
                "What's the difference between stocks and bonds?",
                "What is diversification?",
                "How does the stock market work?",
                "What is an emergency fund?",
                "Explain mutual funds",
                "What are ETFs?",
                "How does retirement planning work?"
            ],
            'budget_planning': [
                "Help me create a budget",
                "How much should I save each month?",
                "Budget planning assistance",
                "How to manage my money better?",
                "Financial planning help",
                "Create a spending plan",
                "Money management tips",
                "How to budget effectively?",
                "Personal finance planning",
                "Save money strategies"
            ]
        }
        
        for intent, examples in default_intents.items():
            self.add_intent(intent, examples)
        
        logger.info("Prepared default training data")
    
    def train(self, data_path: str = None) -> Dict[str, Any]:
        """Train the model on financial data"""
        try:
            # Load training data
            if data_path and os.path.exists(data_path):
                success = self.load_training_data(data_path)
                if not success:
                    self.prepare_default_training_data()
            else:
                self.prepare_default_training_data()
            
            if not self.training_data:
                raise ValueError("No training data available")
            
            logger.info(f"Training with {len(self.training_data)} examples across {len(self.intent_to_id)} intents")
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.intent_to_id)
            )
            
            # Prepare datasets
            texts = [item['text'] for item in self.training_data]
            labels = [item['label'] for item in self.training_data]
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            train_dataset = FinancialIntentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
            val_dataset = FinancialIntentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            
            # Training arguments optimized for M1 Mac
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=0.01,
                logging_dir=f'{self.output_dir}/logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                learning_rate=self.learning_rate,
                fp16=False,  # Disable mixed precision for compatibility
                dataloader_num_workers=0,  # Disable multiprocessing
                remove_unused_columns=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )
            
            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Evaluate the model
            eval_result = trainer.evaluate()
            
            # Save the model
            model_path = os.path.join(self.output_dir, 'financial_intent_model')
            trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            # Save intent mappings
            mappings = {
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent
            }
            
            with open(os.path.join(model_path, 'intent_mappings.json'), 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result.get('eval_accuracy', 0),
                'num_examples': len(self.training_data),
                'num_intents': len(self.intent_to_id)
            }
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy
        }
    
    def load_trained_model(self, model_path: str) -> bool:
        """Load a previously trained model"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load intent mappings
            mappings_path = os.path.join(model_path, 'intent_mappings.json')
            if os.path.exists(mappings_path):
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                
                self.intent_to_id = mappings['intent_to_id']
                self.id_to_intent = {int(k): v for k, v in mappings['id_to_intent'].items()}
            
            logger.info(f"Loaded trained model from {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent for input text using trained model"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top prediction
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()
        
        predicted_intent = self.id_to_intent.get(predicted_class_id, 'unknown')
        
        # Get all scores
        all_scores = {}
        for intent_id, intent_name in self.id_to_intent.items():
            all_scores[intent_name] = predictions[0][intent_id].item()
        
        return {
            'intent': predicted_intent,
            'confidence': confidence,
            'all_scores': all_scores
        }
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        # Load test data
        test_texts = []
        test_labels = []
        true_intents = []
        
        try:
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
            
            for item in test_data:
                if 'text' in item and 'intent' in item:
                    test_texts.append(item['text'])
                    true_intents.append(item['intent'])
                    test_labels.append(self.intent_to_id.get(item['intent'], 0))
        
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return {'error': str(e)}
        
        # Make predictions
        predicted_intents = []
        confidences = []
        
        for text in test_texts:
            prediction = self.predict(text)
            predicted_intents.append(prediction['intent'])
            confidences.append(prediction['confidence'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_intents, predicted_intents)
        avg_confidence = np.mean(confidences)
        
        # Generate classification report
        report = classification_report(true_intents, predicted_intents, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'classification_report': report,
            'num_test_examples': len(test_texts)
        }
    
    def export_model_info(self) -> Dict[str, Any]:
        """Export model information and configuration"""
        return {
            'model_name': self.model_name,
            'num_intents': len(self.intent_to_id),
            'intents': list(self.intent_to_id.keys()),
            'training_config': {
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_length': self.max_length
            },
            'num_training_examples': len(self.training_data)
        }
    
    def create_training_template(self, output_path: str):
        """Create a template file for training data"""
        template = {
            "intents": {
                "account_balance": [
                    "What's my balance?",
                    "How much money do I have?"
                ],
                "spending_analysis": [
                    "How much did I spend?",
                    "Show me my expenses"
                ],
                "investment_advice": [
                    "How should I invest?",
                    "What stocks to buy?"
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Training template created at {output_path}")