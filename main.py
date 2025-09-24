#!/usr/bin/env python3
"""
Financial Chatbot POC - Main Entry Point
Lightweight implementation optimized for M1 Mac with 8GB RAM
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialChatbot:
    """Main chatbot class with extensible architecture"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the chatbot with configuration"""
        self.config = self.load_config(config_path)
        self.intent_classifier = None
        self.entity_extractor = None
        self.response_generator = None
        self.data_manager = None
        self.model_trainer = None
        
        # Initialize components
        self._initialize_components()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "model": {
                "intent_model": "distilbert-base-uncased",
                "max_length": 128,
                "batch_size": 8,
                "use_gpu": False  # Optimized for CPU on M1
            },
            "data": {
                "user_data_api": None,
                "mock_data": True
            },
            "training": {
                "epochs": 3,
                "learning_rate": 2e-5,
                "warmup_steps": 100
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        else:
            config = default_config
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def _initialize_components(self):
        """Initialize all chatbot components"""
        from components.intent_classifier import IntentClassifier
        from components.entity_extractor import EntityExtractor
        from components.response_generator import ResponseGenerator
        from components.data_manager import DataManager
        from components.model_trainer import ModelTrainer
        
        # Initialize components with config
        self.intent_classifier = IntentClassifier(self.config['model'])
        self.entity_extractor = EntityExtractor()
        self.response_generator = ResponseGenerator()
        self.data_manager = DataManager(self.config['data'])
        self.model_trainer = ModelTrainer(self.config['training'])
    
    async def process_message(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Process user message and return response"""
        try:
            # Step 1: Classify intent
            intent_result = await self.intent_classifier.classify(message)
            
            # Step 2: Extract entities
            entities = await self.entity_extractor.extract(message)
            
            # Step 3: Get user data if needed
            user_data = await self.data_manager.get_user_data(user_id)
            
            # Step 4: Generate response
            response = await self.response_generator.generate_response(
                intent=intent_result['intent'],
                entities=entities,
                user_data=user_data,
                original_message=message
            )
            
            return {
                "response": response,
                "intent": intent_result,
                "entities": entities,
                "confidence": intent_result.get('confidence', 0.0),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your request. Please try again.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def train_custom_model(self, training_data_path: str):
        """Train custom model on financial data"""
        return self.model_trainer.train(training_data_path)
    
    def add_custom_intent(self, intent_name: str, examples: List[str]):
        """Add custom intent with training examples"""
        self.model_trainer.add_intent(intent_name, examples)
    
    async def start_interactive_session(self):
        """Start interactive terminal session"""
        print("🏦 Financial Chatbot POC - Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    continue
                
                # Process message
                print("🤖 Bot: Thinking...")
                result = await self.process_message(user_input)
                
                print(f"🤖 Bot: {result['response']}")
                
                # Show debug info if confidence is low
                if result.get('confidence', 0) < 0.5:
                    print(f"   (Intent: {result.get('intent', {}).get('intent', 'unknown')} - "
                          f"Confidence: {result.get('confidence', 0):.2f})")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show available commands and examples"""
        help_text = """
📋 Available Commands:
- quit/exit/q: Exit the chatbot
- help: Show this help message

💡 Example Questions:
Personal Finance:
- "How much money do I have right now?"
- "What's my balance after 30 days?"
- "How much will I spend on groceries this month?"

General Finance:
- "What are the benefits of compound interest?"
- "How can I start investing in stocks?"
- "Explain what a 401k is"
- "What's the difference between stocks and bonds?"

🔧 Training Commands (coming soon):
- Add custom intents
- Train on new data
- Export/import models
        """
        print(help_text)

# Components will be in separate files
def create_component_files():
    """Create component files with basic implementations"""
    
    # Create components directory
    os.makedirs("components", exist_ok=True)
    
    # Create __init__.py
    with open("components/__init__.py", "w") as f:
        f.write("# Financial Chatbot Components\n")

if __name__ == "__main__":
    import sys
    
    # Create component files if they don't exist
    create_component_files()
    
    # Initialize chatbot
    chatbot = FinancialChatbot()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            if len(sys.argv) > 2:
                print("🔄 Training custom model...")
                chatbot.train_custom_model(sys.argv[2])
            else:
                print("❌ Please provide training data path")
        elif sys.argv[1] == "test":
            # Test mode with predefined messages
            test_messages = [
                "How much money do I have?",
                "What's compound interest?",
                "Can you help me with investing?",
                "What's my spending this month?"
            ]
            
            async def test_mode():
                for msg in test_messages:
                    print(f"\n💬 Test: {msg}")
                    result = await chatbot.process_message(msg)
                    print(f"🤖 Response: {result['response']}")
                    print(f"   Intent: {result.get('intent', {}).get('intent', 'unknown')}")
            
            asyncio.run(test_mode())
    else:
        # Interactive mode
        asyncio.run(chatbot.start_interactive_session())