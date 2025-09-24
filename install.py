#!/usr/bin/env python3
"""
Financial Chatbot Setup Script
Automated setup for M1 Mac with 8GB RAM optimization
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"Running: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print("✅ Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error details:", e.stderr)
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Please use Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"✅ Virtual environment '{venv_path}' already exists")
        return True
    
    print(f"🔧 Creating virtual environment: {venv_path}")
    success = run_command(f"python3 -m venv {venv_path}", "Creating virtual environment")
    
    if success:
        print("\n📝 To activate the virtual environment, run:")
        print(f"   source {venv_path}/bin/activate  # On macOS/Linux")
        print(f"   {venv_path}\\Scripts\\activate     # On Windows")
    
    return success

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    # Core dependencies optimized for M1 Mac
    core_packages = [
        "torch",
        "transformers",
        "spacy",
        "pandas",
        "numpy",
        "scikit-learn",
        "aiohttp",
        "python-dateutil"
    ]
    
    # Install packages one by one for better error handling
    for package in core_packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Warning: Failed to install {package}")
    
    return True

def setup_spacy_model():
    """Download spaCy English model"""
    print("🔤 Setting up spaCy English model...")
    success = run_command("python -m spacy download en_core_web_sm", 
                         "Downloading spaCy English model")
    
    if not success:
        print("⚠️  Warning: spaCy model download failed. Entity extraction will use regex only.")
    
    return success

def create_project_structure():
    """Create necessary project directories and files"""
    print("📁 Creating project structure...")
    
    directories = [
        "components",
        "data",
        "models",
        "logs",
        "trained_models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        "components/__init__.py",
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Financial Chatbot Components\n")
            print(f"✅ Created: {init_file}")
    
    return True

def create_config_file():
    """Create default configuration file"""
    config_path = "config.json"
    
    if os.path.exists(config_path):
        print(f"✅ Configuration file '{config_path}' already exists")
        return True
    
    default_config = {
        "model": {
            "intent_model": "distilbert-base-uncased",
            "max_length": 128,
            "batch_size": 4,  # Reduced for 8GB RAM
            "use_gpu": False
        },
        "data": {
            "user_data_api": None,
            "mock_data": True,
            "api_timeout": 10
        },
        "training": {
            "epochs": 3,
            "learning_rate": 2e-5,
            "warmup_steps": 50,  # Reduced for faster training
            "base_model": "distilbert-base-uncased",
            "output_dir": "./trained_models"
        },
        "logging": {
            "level": "INFO",
            "file": "logs/chatbot.log"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"✅ Created configuration file: {config_path}")
    return True

def create_sample_training_data():
    """Create sample training data file"""
    data_path = "data/training_data.json"
    
    if os.path.exists(data_path):
        print(f"✅ Training data file '{data_path}' already exists")
        return True
    
    sample_data = {
        "intents": {
            "account_balance": [
                "What's my current balance?",
                "How much money do I have?",
                "Show me my account balance",
                "Check my balance",
                "How much is in my account?",
                "What's my total balance?",
                "Account balance inquiry",
                "Balance check please"
            ],
            "spending_analysis": [
                "How much did I spend this month?",
                "What are my monthly expenses?",
                "Show me my spending breakdown",
                "Where did my money go?",
                "Analyze my spending",
                "What's my biggest expense?",
                "Track my expenses",
                "Monthly spending report"
            ],
            "future_balance": [
                "What will my balance be next month?",
                "Project my savings in 30 days",
                "How much will I have in 60 days?",
                "Calculate my future balance",
                "Forecast my account balance",
                "Balance projection",
                "Future financial outlook",
                "Estimate balance after expenses"
            ],
            "investment_advice": [
                "How should I invest my money?",
                "What stocks should I buy?",
                "Investment recommendations please",
                "How to start investing?",
                "Best investment options",
                "Should I buy ETFs?",
                "Investment portfolio advice",
                "How to build wealth?"
            ],
            "financial_education": [
                "What is compound interest?",
                "Explain 401k to me",
                "How do bonds work?",
                "What's a mutual fund?",
                "Tell me about ETFs",
                "How does the stock market work?",
                "What is diversification?",
                "Explain retirement planning"
            ],
            "budget_planning": [
                "Help me create a budget",
                "How much should I save each month?",
                "Budget planning assistance",
                "How to manage money better?",
                "Create a spending plan",
                "Financial planning help",
                "Money management tips",
                "How to budget effectively?"
            ]
        }
    }
    
    with open(data_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✅ Created sample training data: {data_path}")
    return True

def create_startup_script():
    """Create convenient startup scripts"""
    
    # Create run script
    run_script = """#!/bin/bash
# Financial Chatbot Startup Script

echo "🏦 Starting Financial Chatbot..."
echo "Make sure you've activated the virtual environment:"
echo "source venv/bin/activate"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active"
    python main.py
else
    echo "⚠️  Virtual environment not detected"
    echo "Run: source venv/bin/activate"
    echo "Then: ./run.sh"
fi
"""
    
    with open("run.sh", 'w') as f:
        f.write(run_script)
    
    # Make executable
    os.chmod("run.sh", 0o755)
    
    print("✅ Created startup script: run.sh")
    return True

def run_quick_test():
    """Run a quick test to verify installation"""
    print("🧪 Running quick installation test...")
    
    try:
        # Test imports
        import torch
        import transformers
        import pandas as pd
        import numpy as np
        import sklearn
        import aiohttp
        
        print("✅ All core packages imported successfully!")
        
        # Test torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Test transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Test if we can create a simple model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("✅ Can load pre-trained models")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("""
🏦 Financial Chatbot POC Setup
===============================
Optimized for M1 Mac with 8GB RAM
    """)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("⚠️  Virtual environment setup failed, continuing...")
    
    # Install dependencies
    install_dependencies()
    
    # Setup spaCy model
    setup_spacy_model()
    
    # Create project structure
    create_project_structure()
    
    # Create configuration
    create_config_file()
    
    # Create sample data
    create_sample_training_data()
    
    # Create startup scripts
    create_startup_script()
    
    # Run tests
    if run_quick_test():
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Activate virtual environment:")
        print("   source venv/bin/activate")
        print("\n2. Run the chatbot:")
        print("   python main.py")
        print("\n3. Or use the startup script:")
        print("   ./run.sh")
        print("\nFor training custom models:")
        print("   python main.py train data/training_data.json")
        print("\nFor testing:")
        print("   python main.py test")
        print("="*60)
    else:
        print("\n❌ Setup completed with errors. Check the output above.")
        print("You may need to manually install some dependencies.")

if __name__ == "__main__":
    main()