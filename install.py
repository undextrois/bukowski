#!/usr/bin/env python3
"""
Financial Chatbot Setup Script
Cross-platform (macOS / Linux / Windows)
Smart dependency checking & developer-friendly
"""

import subprocess
import sys
import os
import json
import platform
from pathlib import Path
from typing import List, Tuple, Dict
import importlib.util

IS_WINDOWS = platform.system().lower() == "windows"
USE_VENV = False

# ANSI color codes for better readability
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def disable():
        """Disable colors on Windows if not supported"""
        if IS_WINDOWS:
            Colors.HEADER = ''
            Colors.OKBLUE = ''
            Colors.OKCYAN = ''
            Colors.OKGREEN = ''
            Colors.WARNING = ''
            Colors.FAIL = ''
            Colors.ENDC = ''
            Colors.BOLD = ''
            Colors.UNDERLINE = ''

# Disable colors on Windows unless using Windows Terminal
if IS_WINDOWS and not os.environ.get('WT_SESSION'):
    Colors.disable()

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER} {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def run_command(command: str, description: str = "", silent: bool = False) -> Tuple[bool, str]:
    """Run a shell command with error handling"""
    if not silent:
        print_info(f"{description}: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip() and not silent:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if not silent:
            print_error(f"Command failed: {e}")
            if e.stderr:
                print(e.stderr)
        return False, e.stderr

def check_python_version() -> bool:
    """Ensure Python 3.8+"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print_error(f"Python {version.major}.{version.minor} detected. Requires Python 3.8+")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def check_package_installed(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and return its version
    
    Args:
        package_name: Name used for pip (e.g., 'scikit-learn')
        import_name: Name used for import (e.g., 'sklearn'). Defaults to package_name
    
    Returns:
        Tuple of (is_installed, version)
    """
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    # Try to import the package
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, ""
        
        # Try to get version
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except Exception:
            return True, "unknown"
    except (ImportError, ModuleNotFoundError):
        return False, ""

def check_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """Check if spaCy model is installed"""
    try:
        import spacy
        spacy.load(model_name)
        return True
    except (ImportError, OSError):
        return False

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages and versions using pip list"""
    python_exec = get_python_path()
    success, output = run_command(
        f'"{python_exec}" -m pip list --format=json',
        "Checking installed packages",
        silent=True
    )
    
    if success:
        try:
            packages = json.loads(output)
            return {pkg['name'].lower(): pkg['version'] for pkg in packages}
        except json.JSONDecodeError:
            return {}
    return {}

def ask_for_venv() -> bool:
    """Ask user if they want to use a virtual environment"""
    global USE_VENV
    
    venv_exists = Path("venv").exists()
    
    if venv_exists:
        print_info("Existing virtual environment detected")
        choice = input(f"{Colors.OKCYAN}Use existing venv? (Y/n): {Colors.ENDC}").strip().lower()
        USE_VENV = choice in ["", "y", "yes"]
    else:
        choice = input(f"{Colors.OKCYAN}Create and use a virtual environment? (Y/n): {Colors.ENDC}").strip().lower()
        USE_VENV = choice in ["", "y", "yes"]
    
    if USE_VENV:
        print_success("Virtual environment will be used")
    else:
        print_warning("Installing globally or in current environment (no venv)")
    
    return USE_VENV

def setup_virtual_environment() -> bool:
    """Create virtual environment if requested"""
    if not USE_VENV:
        return True

    venv_path = Path("venv")
    if venv_path.exists():
        print_success(f"Virtual environment '{venv_path}' already exists")
        return True

    print_header("Creating Virtual Environment")
    success, _ = run_command(
        f'"{sys.executable}" -m venv "{venv_path}"',
        "Creating virtual environment"
    )

    if success:
        print_success("Virtual environment created")
        if IS_WINDOWS:
            print_info(f"Activate with: {venv_path}\\Scripts\\activate")
        else:
            print_info(f"Activate with: source {venv_path}/bin/activate")
    
    return success

def get_pip_path() -> str:
    """Return pip path (venv or system)"""
    if USE_VENV:
        return f'"{get_python_path()}" -m pip'
    return f'"{sys.executable}" -m pip'

def get_python_path() -> str:
    """Return Python path (venv or system)"""
    if USE_VENV:
        if IS_WINDOWS:
            return str(Path("venv") / "Scripts" / "python.exe")
        else:
            return str(Path("venv") / "bin" / "python")
    return sys.executable

def install_dependencies() -> bool:
    """Install required Python packages with smart checking"""
    print_header("Checking Dependencies")
    
    # Package mapping: (pip_name, import_name)
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("spacy", "spacy"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("aiohttp", "aiohttp"),
        ("python-dateutil", "dateutil")
    ]
    
    # Get all installed packages for faster checking
    installed_packages = get_installed_packages()
    
    packages_to_install = []
    packages_installed = []
    
    for pip_name, import_name in packages:
        # Check using pip list first (faster)
        if pip_name.lower() in installed_packages:
            version = installed_packages[pip_name.lower()]
            print_success(f"{pip_name} {version} - Already installed")
            packages_installed.append(pip_name)
        else:
            # Double-check with import
            is_installed, version = check_package_installed(pip_name, import_name)
            if is_installed:
                version_str = f" {version}" if version != "unknown" else ""
                print_success(f"{pip_name}{version_str} - Already installed")
                packages_installed.append(pip_name)
            else:
                print_warning(f"{pip_name} - Not installed")
                packages_to_install.append(pip_name)
    
    if not packages_to_install:
        print_success("All dependencies are already installed!")
        return True
    
    print_header("Installing Missing Dependencies")
    print_info(f"Packages to install: {', '.join(packages_to_install)}")
    
    choice = input(f"\n{Colors.OKCYAN}Proceed with installation? (Y/n): {Colors.ENDC}").strip().lower()
    if choice not in ["", "y", "yes"]:
        print_warning("Installation cancelled by user")
        return False
    
    pip_exec = get_pip_path()
    
    # Upgrade pip first
    print_info("Upgrading pip...")
    run_command(f"{pip_exec} install --upgrade pip", "Upgrading pip", silent=True)
    
    # Install packages
    for pkg in packages_to_install:
        print_info(f"Installing {pkg}...")
        success, _ = run_command(f"{pip_exec} install {pkg}", f"Installing {pkg}")
        if success:
            print_success(f"{pkg} installed successfully")
        else:
            print_error(f"Failed to install {pkg}")
            return False
    
    return True

def setup_spacy_model() -> bool:
    """Download spaCy model if needed"""
    print_header("Checking spaCy Model")
    
    model_name = "en_core_web_sm"
    
    if check_spacy_model(model_name):
        print_success(f"spaCy model '{model_name}' already installed")
        return True
    
    print_warning(f"spaCy model '{model_name}' not found")
    python_exec = get_python_path()
    
    success, _ = run_command(
        f'"{python_exec}" -m spacy download {model_name}',
        f"Downloading spaCy model: {model_name}"
    )
    
    if success:
        print_success(f"spaCy model '{model_name}' installed successfully")
    return success

def create_project_structure() -> bool:
    """Create directories and init files"""
    print_header("Setting Up Project Structure")
    
    directories = ["components", "data", "models", "logs", "trained_models"]
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            print_info(f"{directory}/ - Already exists")
        else:
            dir_path.mkdir(exist_ok=True)
            print_success(f"{directory}/ - Created")
    
    init_file = Path("components/__init__.py")
    if init_file.exists():
        print_info("components/__init__.py - Already exists")
    else:
        init_file.write_text("# Financial Chatbot Components\n")
        print_success("components/__init__.py - Created")
    
    return True

def create_config_file() -> bool:
    """Create default config if not exists"""
    print_header("Checking Configuration")
    
    config_path = Path("config.json")
    if config_path.exists():
        print_info(f"Configuration file '{config_path}' already exists")
        return True
    
    default_config = {
        "model": {
            "intent_model": "distilbert-base-uncased",
            "max_length": 128,
            "batch_size": 4,
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
            "warmup_steps": 50,
            "base_model": "distilbert-base-uncased",
            "output_dir": "./trained_models"
        },
        "logging": {
            "level": "INFO",
            "file": "logs/chatbot.log"
        }
    }
    
    config_path.write_text(json.dumps(default_config, indent=2))
    print_success(f"Created {config_path}")
    return True

def create_sample_training_data() -> bool:
    """Add basic intents"""
    print_header("Checking Training Data")
    
    data_path = Path("data/training_data.json")
    if data_path.exists():
        print_info(f"{data_path} already exists")
        return True
    
    sample_data = {
        "intents": {
            "account_balance": [
                "What's my balance?",
                "Check balance",
                "How much money do I have?",
                "Show my account balance"
            ],
            "spending_analysis": [
                "Spending this month?",
                "Show spending",
                "What did I spend on?",
                "Analyze my expenses"
            ],
            "transaction_history": [
                "Show recent transactions",
                "What are my latest transactions?",
                "Transaction history"
            ]
        }
    }
    
    data_path.write_text(json.dumps(sample_data, indent=2))
    print_success(f"Created {data_path}")
    return True

def create_startup_script() -> bool:
    """Generate startup scripts"""
    print_header("Creating Startup Scripts")
    
    # Linux / macOS startup script
    bash_activate = "source venv/bin/activate" if USE_VENV else "# No venv used"
    bash_script = (
        "#!/bin/bash\n"
        "echo \"Starting Financial Chatbot...\"\n"
        f"{bash_activate}\n"
        "python main.py\n"
    )
    
    run_script = Path("run.sh")
    run_script.write_text(bash_script)
    os.chmod(run_script, 0o755)
    print_success("Created run.sh (Linux/macOS)")
    
    # Windows startup script
    windows_activate = "call venv\\Scripts\\activate" if USE_VENV else "rem No venv used"
    bat_script = (
        "@echo off\n"
        "echo Starting Financial Chatbot...\n"
        f"{windows_activate}\n"
        "python main.py\n"
        "pause\n"
    )
    
    bat_file = Path("run.bat")
    bat_file.write_text(bat_script)
    print_success("Created run.bat (Windows)")
    
    return True

def create_dev_requirements() -> bool:
    """Create requirements.txt for easy installation"""
    print_header("Creating requirements.txt")
    
    requirements_path = Path("requirements.txt")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "spacy>=3.5.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "aiohttp>=3.8.0",
        "python-dateutil>=2.8.0"
    ]
    
    if requirements_path.exists():
        print_info("requirements.txt already exists")
        choice = input(f"{Colors.OKCYAN}Overwrite? (y/N): {Colors.ENDC}").strip().lower()
        if choice not in ["y", "yes"]:
            return True
    
    requirements_path.write_text("\n".join(requirements) + "\n")
    print_success("Created requirements.txt")
    print_info("Install with: pip install -r requirements.txt")
    return True

def run_quick_test() -> bool:
    """Quick import test"""
    print_header("Running Import Tests")
    
    python_exec = get_python_path()
    
    test_imports = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("aiohttp", "aiohttp")
    ]
    
    all_passed = True
    for import_name, display_name in test_imports:
        cmd = f'"{python_exec}" -c "import {import_name}; print(\'OK\')"'
        success, output = run_command(cmd, "", silent=True)
        
        if success and "OK" in output:
            print_success(f"{display_name} import test passed")
        else:
            print_error(f"{display_name} import test failed")
            all_passed = False
    
    return all_passed

def print_next_steps():
    """Print what to do next"""
    print_header("Setup Complete!")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    if USE_VENV:
        if IS_WINDOWS:
            print(f"  1. Activate virtual environment:")
            print(f"     {Colors.OKCYAN}venv\\Scripts\\activate{Colors.ENDC}\n")
        else:
            print(f"  1. Activate virtual environment:")
            print(f"     {Colors.OKCYAN}source venv/bin/activate{Colors.ENDC}\n")
    
    print(f"  2. Run the chatbot:")
    if IS_WINDOWS:
        print(f"     {Colors.OKCYAN}python main.py{Colors.ENDC} or {Colors.OKCYAN}run.bat{Colors.ENDC}\n")
    else:
        print(f"     {Colors.OKCYAN}python main.py{Colors.ENDC} or {Colors.OKCYAN}./run.sh{Colors.ENDC}\n")
    
    print(f"  3. Edit configuration:")
    print(f"     {Colors.OKCYAN}config.json{Colors.ENDC}\n")
    
    print(f"  4. Add training data:")
    print(f"     {Colors.OKCYAN}data/training_data.json{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Developer Tips:{Colors.ENDC}\n")
    print(f"  • Use {Colors.OKCYAN}requirements.txt{Colors.ENDC} for consistent installs")
    print(f"  • Re-run this script anytime to check dependencies")
    print(f"  • Check logs in: {Colors.OKCYAN}logs/chatbot.log{Colors.ENDC}")

def main():
    """Main installation flow"""
    print(f"""
{Colors.BOLD}{Colors.HEADER}╔═══════════════════════════════════════════════════════╗
║     Financial Chatbot POC - Smart Installer          ║
║     Cross-platform with dependency checking          ║
╚═══════════════════════════════════════════════════════╝{Colors.ENDC}
""")
    
    if not check_python_version():
        sys.exit(1)
    
    if not ask_for_venv():
        print_warning("Proceeding without virtual environment")
    
    steps = [
        (setup_virtual_environment, "Virtual environment setup failed"),
        (install_dependencies, "Dependency installation failed"),
        (setup_spacy_model, "spaCy model setup failed"),
        (create_project_structure, "Project structure creation failed"),
        (create_config_file, "Config file creation failed"),
        (create_sample_training_data, "Training data creation failed"),
        (create_startup_script, "Startup script creation failed"),
        (create_dev_requirements, "Requirements file creation failed"),
        (run_quick_test, "Import tests failed")
    ]
    
    for step_func, error_msg in steps:
        if not step_func():
            print_error(error_msg)
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Installation interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
