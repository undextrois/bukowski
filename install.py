#!/usr/bin/env python3
"""
Financial Chatbot Setup Script
Cross-platform (macOS / Linux / Windows)
Optional Virtual Environment
"""

import subprocess
import sys
import os
import json
import platform
from pathlib import Path

IS_WINDOWS = platform.system().lower() == "windows"
USE_VENV = False  # Will be set at runtime

def run_command(command, description=""):
    """Run a shell command with error handling"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"Running: {command}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def check_python_version():
    """Ensure Python 3.8+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def ask_for_venv():
    """Ask user if they want to use a virtual environment"""
    global USE_VENV
    choice = input("\n🧰 Do you want to use a virtual environment? (Y/n): ").strip().lower()
    USE_VENV = (choice in ["", "y", "yes"])
    if USE_VENV:
        print("✅ Virtual environment will be used")
    else:
        print("⚠️  Installing globally or in the current environment (no venv)")
    return USE_VENV

def setup_virtual_environment():
    """Create virtual environment if requested"""
    if not USE_VENV:
        return True

    venv_path = Path("venv")
    if venv_path.exists():
        print(f"✅ Virtual environment '{venv_path}' already exists")
        return True

    print(f"🔧 Creating virtual environment: {venv_path}")
    success = run_command(f'"{sys.executable}" -m venv "{venv_path}"', "Creating virtual environment")

    if success:
        if IS_WINDOWS:
            print("\n📝 To activate the virtual environment:")
            print(f"   {venv_path}\\Scripts\\activate")
        else:
            print("\n📝 To activate the virtual environment:")
            print(f"   source {venv_path}/bin/activate")
    return success

def venv_pip():
    """Return pip path (venv or system)"""
    if USE_VENV:
        return Path("venv") / ("Scripts" if IS_WINDOWS else "bin") / ("pip.exe" if IS_WINDOWS else "pip")
    return sys.executable.replace("python.exe", "pip.exe") if IS_WINDOWS else "pip"

def venv_python():
    """Return Python path (venv or system)"""
    if USE_VENV:
        return Path("venv") / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python")
    return sys.executable

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")

    packages = [
        "torch",
        "transformers",
        "spacy",
        "pandas",
        "numpy",
        "scikit-learn",
        "aiohttp",
        "python-dateutil"
    ]

    pip_exec = f'"{venv_pip()}"'
    for pkg in packages:
        run_command(f"{pip_exec} install {pkg}", f"Installing {pkg}")

    return True

def setup_spacy_model():
    """Download spaCy model"""
    python_exec = venv_python()
    run_command(f'"{python_exec}" -m spacy download en_core_web_sm', "Downloading spaCy English model")

def create_project_structure():
    """Create directories and init files"""
    print("📁 Creating project structure...")
    for directory in ["components", "data", "models", "logs", "trained_models"]:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory} ready")

    init_file = Path("components/__init__.py")
    if not init_file.exists():
        init_file.write_text("# Financial Chatbot Components\n")

def create_config_file():
    """Create default config if not exists"""
    config_path = Path("config.json")
    if config_path.exists():
        print(f"✅ Configuration file '{config_path}' already exists")
        return

    default_config = {
        "model": {
            "intent_model": "distilbert-base-uncased",
            "max_length": 128,
            "batch_size": 4,
            "use_gpu": False
        },
        "data": {"user_data_api": None, "mock_data": True, "api_timeout": 10},
        "training": {
            "epochs": 3,
            "learning_rate": 2e-5,
            "warmup_steps": 50,
            "base_model": "distilbert-base-uncased",
            "output_dir": "./trained_models"
        },
        "logging": {"level": "INFO", "file": "logs/chatbot.log"}
    }
    config_path.write_text(json.dumps(default_config, indent=2))
    print(f"✅ Created {config_path}")

def create_sample_training_data():
    """Add basic intents"""
    data_path = Path("data/training_data.json")
    if data_path.exists():
        print(f"✅ {data_path} already exists")
        return
    data_path.write_text(json.dumps({
        "intents": {
            "account_balance": ["What's my balance?", "Check balance"],
            "spending_analysis": ["Spending this month?", "Show spending"]
        }
    }, indent=2))
    print(f"✅ Created {data_path}")

def create_startup_script():
    """Generate startup scripts"""

    # Linux / macOS startup script
    bash_activate = "source venv/bin/activate" if USE_VENV else "# No venv used"
    bash_script = (
        "#!/bin/bash\n"
        "echo \"🏦 Starting Financial Chatbot...\"\n"
        f"{bash_activate}\n"
        "python main.py\n"
    )
    Path("run.sh").write_text(bash_script)
    os.chmod("run.sh", 0o755)
    print("✅ Created run.sh")

    # Windows startup script
    windows_activate = "call venv\\Scripts\\activate" if USE_VENV else "rem No venv used"
    bat_script = (
        "@echo off\n"
        "echo 🏦 Starting Financial Chatbot...\n"
        f"{windows_activate}\n"
        "python main.py\n"
    )
    Path("run.bat").write_text(bat_script)
    print("✅ Created run.bat (Windows)")



def run_quick_test():
    """Quick import test"""
    python_exec = venv_python()
    cmd = f'"{python_exec}" -c "import torch, transformers, pandas, numpy, sklearn, aiohttp; print(\'✅ All core packages OK\')"'
    run_command(cmd, "Quick import test")

def main():
    print("""
🏦 Financial Chatbot POC Setup
===============================
Cross-platform installer with optional venv
    """)
    if not check_python_version():
        sys.exit(1)

    ask_for_venv()
    setup_virtual_environment()
    install_dependencies()
    setup_spacy_model()
    create_project_structure()
    create_config_file()
    create_sample_training_data()
    create_startup_script()
    run_quick_test()

    if USE_VENV:
        print("\n✅ Setup complete using virtual environment.")
        print("👉 Activate it and run the chatbot:")
        print(f"   {'venv\\Scripts\\activate' if IS_WINDOWS else 'source venv/bin/activate'}")
    else:
        print("\n✅ Setup complete without virtual environment.")
        print("👉 You can run the chatbot directly with:")
        print("   python main.py")

if __name__ == "__main__":
    main()
