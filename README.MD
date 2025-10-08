# 🏦 Financial Chatbot POC

A lightweight, extensible financial chatbot built with open-source models from Hugging Face. Optimized for M1 Mac with 8GB RAM and designed for easy integration with Rasa and other chatbot frameworks.

## ✨ Features

- **Personal Finance Queries**: Account balances, spending analysis, future projections
- **Financial Education**: Compound interest, investing, budgeting explanations
- **Investment Advice**: Stock market guidance, portfolio recommendations
- **Extensible Architecture**: Easy to add new intents and customize responses
- **Lightweight**: Optimized for limited resources (8GB RAM)
- **Mock Data Support**: Test without real financial APIs
- **Custom Training**: Train on your specific financial domain data

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Run the automated setup
python install.py

# Activate virtual environment
source venv/bin/activate

# Start the chatbot
python main.py
```

### 2. Interactive Usage

```bash
🏦 Financial Chatbot POC - Interactive Mode
Type 'quit' to exit, 'help' for commands
--------------------------------------------------

💬 You: How much money do I have?
🤖 Bot: Based on your accounts, you currently have $11,000.75 available. This total includes 4 accounts.

💬 You: What's compound interest?
🤖 Bot: Let me explain Compound Interest: Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods...

💬 You: How much will I have in 60 days?
🤖 Bot: In 60 days, your projected balance will be $12,180.50. Based on your current income and spending patterns, you're saving about $19.67 per day.
```

## 📁 Project Structure

```
financial-chatbot/
├── main.py                 # Main application entry point
├── components/             # Core chatbot components
│   ├── intent_classifier.py   # Intent classification
│   ├── entity_extractor.py    # Entity extraction
│   ├── response_generator.py  # Response generation
│   ├── data_manager.py        # Data handling & API integration
│   └── model_trainer.py       # Custom model training
├── data/                   # Training and sample data
│   └── training_data.json     # Sample training data
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
├── setup.py              # Automated setup script
└── README.md             # This file
```

## 🎯 Supported Query Types

### Personal Finance
- **Account Balance**: "How much money do I have?", "What's my balance?"
- **Spending Analysis**: "How much did I spend on groceries?", "What are my expenses?"
- **Future Projections**: "What will my balance be in 30 days?"

### Financial Education
- **Investment Basics**: "What are stocks?", "How do bonds work?"
- **Retirement Planning**: "Explain 401k", "What is compound interest?"
- **Budgeting**: "How to create a budget?", "Money management tips"

### Investment Advice
- **Portfolio Guidance**: "How should I invest?", "What stocks to buy?"
- **Strategy Help**: "Investment recommendations", "How to start investing?"

## 🔧 Configuration

Edit `config.json` to customize the chatbot:

```json
{
  "model": {
    "intent_model": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 4,
    "use_gpu": false
  },
  "data": {
    "user_data_api": null,
    "mock_data": true,
    "api_timeout": 10
  },
  "training": {
    "epochs": 3,
    "learning_rate": 2e-5,
    "base_model": "distilbert-base-uncased"
  }
}
```

## 🎓 Custom Training

### 1. Prepare Training Data

Create or edit `data/training_data.json`:

```json
{
  "intents": {
    "custom_intent": [
      "Example question 1",
      "Example question 2",
      "Another way to ask the same thing"
    ]
  }
}
```

### 2. Train the Model

```bash
# Train with custom data
python main.py train data/training_data.json

# Test the trained model
python main.py test
```

### 3. Add Custom Responses

Edit `components/response_generator.py` to add custom response templates and logic for your new intents.

## 🔌 API Integration

### Replace Mock Data with Real APIs

1. **Update Configuration**:
```json
{
  "data": {
    "user_data_api": "https://your-api.com/v1",
    "mock_data": false
  }
}
```

2. **Implement API Endpoints** in `components/data_manager.py`:
   - `/users/{user_id}/accounts` - Account information
   - `/users/{user_id}/transactions` - Transaction history
   - `/users/{user_id}/profile` - User profile data

### API Response Format

```json
{
  "accounts": [
    {
      "account_id": "acc_001",
      "account_type": "checking",
      "balance": 2500.75,
      "currency": "USD"
    }
  ],
  "transactions": [
    {
      "transaction_id": "txn_001",
      "date": "2025-01-15",
      "amount": -45.67,
      "category": "groceries",
      "description": "Supermarket"
    }
  ]
}
```

## 🤖 Rasa Integration

To integrate with your existing Rasa chatbot:

### 1. As a Custom Action

```python
from rasa_sdk import Action
from main import FinancialChatbot

class ActionFinancialQuery(Action):
    def __init__(self):
        self.financial_bot = FinancialChatbot()
    
    def name(self):
        return "action_financial_query"
    
    async def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get('text')
        response = await self.financial_bot.process_message(user_message)
        
        dispatcher.utter_message(text=response['response'])
        return []
```

### 2. As a Separate Service

Run the financial chatbot as a microservice and call it from Rasa:

```python
import aiohttp

async def call_financial_service(message, user_id):
    async with aiohttp.ClientSession() as session:
        payload = {"message": message, "user_id": user_id}
        async with session.post("http://localhost:8000/query", json=payload) as response:
            return await response.json()
```

## 💡 Customization Examples

### Adding New Intents

```python
# In main.py or as a separate script
chatbot = FinancialChatbot()

# Add new intent with examples
new_examples = [
    "What's my credit score?",
    "Check my credit rating",
    "How's my credit health?"
]
chatbot.add_custom_intent("credit_score", new_examples)

# Train the model
chatbot.train_custom_model("data/training_data.json")
```

### Custom Response Templates

```python
# In components/response_generator.py
def _handle_credit_score_query(self, entities, user_data):
    # Your custom logic here
    credit_score = user_data.get('credit_score', 750)
    return f"Your current credit score is {credit_score}. That's considered {'excellent' if credit_score > 740 else 'good'}!"
```

### Adding Financial Calculations

```python
# Example: Loan payment calculator
def calculate_loan_payment(principal, rate, years):
    monthly_rate = rate / 12 / 100
    num_payments = years * 12
    
    if monthly_rate == 0:
        return principal / num_payments
    
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
              ((1 + monthly_rate)**num_payments - 1)
    
    return payment

# Usage in response generator
payment = calculate_loan_payment(250000, 4.5, 30)  # $250k loan, 4.5% APR, 30 years
response = f"Your monthly payment would be approximately ${payment:.2f}"
```

## 📊 Performance Optimization

### For Limited Resources (8GB RAM)

1. **Reduce Batch Size**: Set `batch_size: 2-4` in config
2. **Use CPU Only**: Set `use_gpu: false`
3. **Limit Model Size**: Use DistilBERT instead of BERT
4. **Cache Management**: Clear cache regularly for long-running sessions

### Memory Usage Tips

```python
# Clear cache periodically
chatbot.data_manager.clear_cache()

# Use smaller models
config['model']['intent_model'] = 'distilbert-base-uncased'  # Instead of 'bert-base-uncased'

# Reduce sequence length
config['model']['max_length'] = 64  # Instead of 128
```

## 🧪 Testing

### Run Built-in Tests

```bash
# Test with sample queries
python main.py test

# Test specific functionality
python -c "
import asyncio
from main import FinancialChatbot

async def test():
    bot = FinancialChatbot()
    result = await bot.process_message('How much money do I have?')
    print(result)

asyncio.run(test())
"
```

### Custom Test Cases

```python
test_cases = [
    "What's my balance?",
    "How much did I spend on food?",
    "What is compound interest?",
    "Should I invest in stocks?",
    "Help me create a budget"
]

async def run_tests():
    chatbot = FinancialChatbot()
    for question in test_cases:
        result = await chatbot.process_message(question)
        print(f"Q: {question}")
        print(f"A: {result['response']}")
        print(f"Intent: {result['intent']['intent']} (confidence: {result['confidence']:.2f})")
        print("-" * 50)
```

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found" error**:
   ```bash
   # Download the model manually
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased')"
   ```

2. **spaCy model missing**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory errors during training**:
   - Reduce batch size to 2 or 1
   - Use smaller model (distilbert instead of bert)
   - Reduce max_length to 64

4. **Import errors**:
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

### Performance Issues

- **Slow responses**: Enable GPU if available, or use smaller models
- **High memory usage**: Clear cache regularly, reduce batch size
- **Training too slow**: Reduce epochs to 1-2 for testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly on M1 Mac environment
5. Submit a pull request

### Areas for Contribution

- Additional financial calculators
- More sophisticated NER for financial entities
- Integration with popular financial APIs (Plaid, Yodlee)
- Support for multiple languages
- Advanced conversation flow management
- Real-time market data integration

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🙋‍♀️ Support

For questions, issues, or feature requests:

1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include your system specs and error messages

---

**Happy Financial Chatbotting! 🏦🤖💰**
