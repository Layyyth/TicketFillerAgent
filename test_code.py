#!/usr/bin/env python3
"""
Test script to verify the model training code works correctly.
"""

import json
import sys
from model_training import load_json_data, create_training_prompt

def test_data_loading():
    """Test that the training data can be loaded correctly."""
    print("Testing data loading...")
    
    try:
        # Load the training data
        samples = load_json_data("training_data.json")
        print(f"✓ Successfully loaded {len(samples)} samples")
        
        # Test a few samples
        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Input: {sample['user_input'][:100]}...")
            print(f"  Ticket type: {sample['ticket_data']['ticket_type']}")
            print(f"  Severity: {sample['ticket_data']['severity']}")
        
        # Test prompt creation
        print("\nTesting prompt creation...")
        prompt = create_training_prompt(samples[0])
        print(f"✓ Prompt created successfully (length: {len(prompt)} chars)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_imports():
    """Test that all required imports are available."""
    print("Testing imports...")
    
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'peft',
        'trl',
        'pandas',
        'sklearn',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Model Training Code Test ===\n")
    
    # Test imports first
    imports_ok = test_imports()
    print()
    
    if imports_ok:
        # Test data loading
        data_ok = test_data_loading()
        
        if data_ok:
            print("\n✓ All tests passed! The code should work correctly.")
        else:
            print("\n✗ Data loading failed. Check the training_data.json file.")
            sys.exit(1)
    else:
        print("\n✗ Import test failed. Install missing packages first.")
        sys.exit(1) 