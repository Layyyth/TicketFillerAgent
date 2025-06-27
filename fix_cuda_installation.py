#!/usr/bin/env python3
"""
Script to fix CUDA compatibility issues for RTX Pro 6000 Blackwell GPU.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_current_pytorch():
    """Check current PyTorch installation."""
    print("=== Current PyTorch Installation ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch not installed")
        return False
    
    return True

def install_correct_pytorch():
    """Install PyTorch with CUDA 12.4+ support for RTX Pro 6000."""
    print("\n=== Installing PyTorch with CUDA 12.4+ Support ===")
    
    # Uninstall current PyTorch
    print("Uninstalling current PyTorch...")
    success, stdout, stderr = run_command("pip uninstall torch torchvision torchaudio -y")
    if not success:
        print(f"Warning: Failed to uninstall PyTorch: {stderr}")
    
    # Install PyTorch with CUDA 12.4+ support
    print("Installing PyTorch with CUDA 12.4+ support...")
    install_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    
    success, stdout, stderr = run_command(install_command)
    
    if success:
        print("✓ PyTorch installed successfully!")
        return True
    else:
        print(f"✗ Failed to install PyTorch: {stderr}")
        return False

def alternative_solutions():
    """Provide alternative solutions."""
    print("\n=== Alternative Solutions ===")
    print("If PyTorch installation fails, try these alternatives:")
    print()
    print("1. Use CPU training (slower but compatible):")
    print("   - The script will automatically fall back to CPU")
    print("   - Training will take much longer but will work")
    print()
    print("2. Use a different GPU if available:")
    print("   - RTX 4090, A100, V100, or other compatible GPUs")
    print()
    print("3. Use Google Colab or other cloud platforms:")
    print("   - They have compatible GPU environments")
    print()
    print("4. Wait for PyTorch to add support for RTX Pro 6000:")
    print("   - This is a new GPU architecture")
    print("   - Support should be added in future PyTorch releases")

def main():
    print("🔧 RTX Pro 6000 Blackwell CUDA Compatibility Fix")
    print("=" * 50)
    
    # Check current installation
    pytorch_installed = check_current_pytorch()
    
    if pytorch_installed:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name().lower()
                if "blackwell" in gpu_name or "rtx pro 6000" in gpu_name:
                    print("\n⚠️ Detected RTX Pro 6000 Blackwell GPU")
                    print("⚠️ Current PyTorch may not be compatible")
                    
                    response = input("\nDo you want to try installing PyTorch with CUDA 12.4+ support? (y/n): ")
                    if response.lower() == 'y':
                        if install_correct_pytorch():
                            print("\n✓ Installation complete! Try running your training script again.")
                        else:
                            alternative_solutions()
                    else:
                        alternative_solutions()
                else:
                    print("\n✓ GPU appears to be compatible with current PyTorch")
            else:
                print("\n⚠️ CUDA not available - will use CPU training")
        except Exception as e:
            print(f"Error checking GPU: {e}")
    else:
        print("\nPyTorch not installed. Installing with CUDA support...")
        install_correct_pytorch()

if __name__ == "__main__":
    main() 