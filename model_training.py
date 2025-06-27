import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import pandas as pd
from typing import Dict, List, Optional
import re
import os

# Hugging Face login
def setup_huggingface_login():
    """Setup Hugging Face login for model downloads and uploads."""
    from huggingface_hub import login
    
    # Check if HF_TOKEN environment variable is set
    hf_token = os.getenv('HF_TOKEN')
    
    if hf_token:
        print("Using HF_TOKEN environment variable for Hugging Face login...")
        login(token=hf_token)
    else:
        print("HF_TOKEN environment variable not found.")
        print("You can either:")
        print("1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here")
        print("2. Or login interactively when prompted")
        try:
            login()
        except Exception as e:
            print(f"Login failed: {e}")
            print("Continuing without login (may fail if model requires authentication)...")

# Setup Hugging Face login
setup_huggingface_login()

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Commercial-friendly license
OUTPUT_DIR = "./banking-ticket-model"
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# Ticket Schema
TICKET_SCHEMA = {
    "ticket_type": ["complaint", "inquiry", "assistance"],
    "title": str,
    "description": str,
    "severity": ["low", "medium", "high", "critical"],
    "department_impacted": str,
    "service_impacted": str,
    "preferred_communication": ["email", "phone", "chat", "in-person"],
    "assistance_request": str  # Only for assistance tickets
}

# System prompt for banking domain
SYSTEM_PROMPT = """You are a banking customer service ticket classification and filling assistant. Your role is to:
1. Analyze customer inputs and extract relevant information
2. Fill ticket fields accurately based on the customer's request
3. Stay strictly within banking and financial services domain
4. Reject any requests outside of banking support

You must ONLY respond with a valid JSON object containing the ticket fields. Do not provide any other information or engage in conversation.

Required fields:
- ticket_type: "complaint", "inquiry", or "assistance"
- title: Brief summary of the issue
- description: Detailed description
- severity: "low", "medium", "high", or "critical"
- department_impacted: The bank department affected
- service_impacted: The specific service affected
- preferred_communication: "email", "phone", "chat", or "in-person"
- assistance_request: (ONLY if ticket_type is "assistance") - specific assistance needed

If the request is not related to banking, respond with: {"error": "Request outside banking domain"}"""

def create_training_prompt(example: Dict) -> str:
    """Create a training prompt from an example."""
    user_input = example['user_input']
    ticket_data = example['ticket_data']
    
    # Format the conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": json.dumps(ticket_data, indent=2)}
    ]
    
    # Convert to string format (adjust based on model's chat template)
    formatted = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted += f"<s>[INST] <<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
        elif msg["role"] == "user":
            formatted += f"{msg['content']} [/INST]"
        else:
            formatted += f" {msg['content']}</s>"
    
    return formatted

def load_json_data(json_path: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Validate data format
    required_fields = ["user_input", "ticket_data"]
    ticket_fields = ["ticket_type", "title", "description", "severity", 
                    "department_impacted", "service_impacted", "preferred_communication"]
    
    valid_samples = []
    skipped = 0
    
    for idx, sample in enumerate(data):
        # Check required top-level fields
        if not all(field in sample for field in required_fields):
            print(f"Skipping sample {idx}: missing required fields")
            skipped += 1
            continue
        
        ticket_data = sample["ticket_data"]
        
        # Check ticket fields
        if not all(field in ticket_data for field in ticket_fields):
            print(f"Skipping sample {idx}: missing ticket fields")
            skipped += 1
            continue
        
        # Validate ticket type
        if ticket_data["ticket_type"] not in ["complaint", "inquiry", "assistance"]:
            print(f"Skipping sample {idx}: invalid ticket type '{ticket_data['ticket_type']}'")
            skipped += 1
            continue
        
        # Check for assistance_request if ticket type is assistance
        if ticket_data["ticket_type"] == "assistance" and "assistance_request" not in ticket_data:
            # Add a default assistance request
            ticket_data["assistance_request"] = "general assistance"
        
        valid_samples.append(sample)
    
    print(f"Valid samples: {len(valid_samples)}, Skipped: {skipped}")
    
    # Add negative examples for out-of-domain detection
    negative_examples = [
        "Can you give me a recipe for chocolate cake?",
        "What's the weather like today?",
        "Tell me about the history of Rome",
        "How do I fix my computer?",
        "What movies are playing this weekend?",
        "Show me funny cat videos",
        "How to hack a website?",
        "Write me a love poem",
        "What's the score of the game?",
        "Tell me a joke"
    ]
    
    for neg_input in negative_examples:
        valid_samples.append({
            "user_input": neg_input,
            "ticket_data": {"error": "Request outside banking domain"}
        })
    
    return valid_samples

def load_real_data(csv_path: str) -> List[Dict]:
    """Load and process real complaints data from CSV."""
    import pandas as pd
    import random
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    samples = []
    skipped = 0
    
    # Map variations to standard ticket types
    ticket_type_mapping = {
        "complaint": "complaint",
        "complaints": "complaint",
        "complaining": "complaint",
        "compla intimation": "complaint",
        "inquiry": "inquiry",
        "enquiray": "inquiry",
        "assistance": "assistance",
        "request for assistance": "assistance"
    }
    
    # Map severity variations
    severity_mapping = {
        "low": "low",
        "medium": "medium",
        " medium": "medium",
        "high": "high",
        "High": "high",
        "critical": "critical"
    }
    
    for idx, row in df.iterrows():
        try:
            # Parse the output JSON
            output_data = json.loads(row['output'])
            
            # Clean and standardize ticket type
            original_type = output_data.get("Ticket Type", "").lower().strip()
            ticket_type = ticket_type_mapping.get(original_type, None)
            
            if not ticket_type:
                print(f"Unknown ticket type: {original_type}")
                skipped += 1
                continue
            
            # Clean severity
            original_severity = output_data.get("Severity", "medium").lower().strip()
            severity = severity_mapping.get(original_severity, "medium")
            
            # Extract assistance request if present
            assistance_request = None
            if ticket_type == "assistance":
                # Try to extract from description
                desc = output_data.get("Description", "")
                if "loan" in desc.lower():
                    assistance_request = "loan assistance"
                elif "account" in desc.lower():
                    assistance_request = "account assistance"
                else:
                    assistance_request = "general assistance"
            
            # Map old field names to new schema
            ticket_data = {
                "ticket_type": ticket_type,
                "title": output_data.get("Title", ""),
                "description": output_data.get("Description", ""),
                "severity": severity,
                "department_impacted": output_data.get("Department Impacted", "Customer Service"),
                "service_impacted": output_data.get("Service Impacted", "General Banking"),
                "preferred_communication": output_data.get("Preferred Communication", "email").lower()
            }
            
            # Clean preferred communication
            if ticket_data["preferred_communication"] == "not specified":
                ticket_data["preferred_communication"] = random.choice(["email", "phone", "chat"])
            
            # Add assistance request if needed
            if ticket_type == "assistance" and assistance_request:
                ticket_data["assistance_request"] = assistance_request
            
            samples.append({
                "user_input": row['input'],
                "ticket_data": ticket_data
            })
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON at row {idx}: {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped += 1
            continue
    
    print(f"Loaded {len(samples)} samples, skipped {skipped} invalid rows")
    
    # Add synthetic negative examples
    negative_examples = [
        "Can you give me a recipe for chocolate cake?",
        "What's the weather like today?",
        "Tell me about the history of Rome",
        "How do I fix my computer?",
        "What movies are playing this weekend?",
        "Show me funny cat videos",
        "How to hack a website?",
        "Write me a love poem",
        "What's the score of the game?",
        "Tell me a joke"
    ]
    
    for neg_input in negative_examples:
        samples.append({
            "user_input": neg_input,
            "ticket_data": {"error": "Request outside banking domain"}
        })
    
    return samples

def generate_synthetic_data(num_samples: int = 1000) -> List[Dict]:
    """Generate additional synthetic training data for banking tickets."""
    import random
    
    # This function can be used to augment real data with synthetic examples
    samples = []
    
    # Add some edge cases and specific scenarios
    edge_cases = [
        {
            "user_input": "URGENT!!! My account was hacked and someone stole $5000!!!",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "Account security breach - unauthorized transaction",
                "description": "Customer reports account was hacked with $5000 stolen. Urgent security issue requiring immediate attention.",
                "severity": "critical",
                "department_impacted": "Security Department",
                "service_impacted": "Online Banking",
                "preferred_communication": "phone"
            }
        },
        {
            "user_input": "I need help refinancing my mortgage to get a better rate",
            "ticket_data": {
                "ticket_type": "assistance",
                "title": "Mortgage refinancing assistance",
                "description": "Customer requesting help with mortgage refinancing to obtain better interest rate.",
                "severity": "medium",
                "department_impacted": "Loan Department",
                "service_impacted": "Mortgage Services",
                "preferred_communication": "email",
                "assistance_request": "mortgage refinancing"
            }
        }
    ]
    
    samples.extend(edge_cases)
    
    # Generate some synthetic data to balance the dataset
    # (keeping a smaller version of the original synthetic data generation)
    
    return samples

def evaluate_model(model, tokenizer, test_samples: List[Dict], output_dir: str):
    """Evaluate model accuracy on test set."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import numpy as np
    
    print("\n=== Model Evaluation ===")
    
    predictions = []
    true_labels = []
    detailed_results = []
    
    model.eval()
    
    for idx, sample in enumerate(test_samples):
        user_input = sample["user_input"]
        true_ticket = sample["ticket_data"]
        
        # Skip error samples for classification metrics
        if "error" in true_ticket:
            continue
        
        # Create prompt
        prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_input} [/INST]"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("[/INST]")[-1].strip()
        
        # Extract predicted ticket
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', assistant_response, re.DOTALL)
            if json_match:
                pred_ticket = json.loads(json_match.group())
                
                # Compare ticket types
                true_type = true_ticket.get("ticket_type", "unknown")
                pred_type = pred_ticket.get("ticket_type", "unknown")
                
                true_labels.append(true_type)
                predictions.append(pred_type)
                
                # Store detailed results
                detailed_results.append({
                    "input": user_input,
                    "true": true_ticket,
                    "predicted": pred_ticket,
                    "correct": true_type == pred_type
                })
            else:
                # Failed to parse
                predictions.append("parse_error")
                true_labels.append(true_ticket.get("ticket_type", "unknown"))
        
        except Exception as e:
            predictions.append("error")
            true_labels.append(true_ticket.get("ticket_type", "unknown"))
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx + 1}/{len(test_samples)} samples...")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    unique_labels = sorted(list(set(true_labels + predictions)))
    cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
    
    # Pretty print confusion matrix
    print(f"{'':>15}", end="")
    for label in unique_labels:
        print(f"{label:>15}", end="")
    print()
    
    for i, label in enumerate(unique_labels):
        print(f"{label:>15}", end="")
        for j in range(len(unique_labels)):
            print(f"{cm[i][j]:>15}", end="")
        print()
    
    # Save detailed results
    results_path = f"{output_dir}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "classification_report": classification_report(true_labels, predictions, output_dict=True),
            "confusion_matrix": cm.tolist(),
            "detailed_results": detailed_results[:20]  # Save first 20 for inspection
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_path}")
    
    # Field-level accuracy (for correctly classified tickets)
    field_accuracies = {
        "severity": 0.0,
        "department_impacted": 0.0,
        "service_impacted": 0.0,
        "preferred_communication": 0.0
    }
    
    correct_tickets = [r for r in detailed_results if r["correct"]]
    
    if correct_tickets:
        for result in correct_tickets:
            true_t = result["true"]
            pred_t = result["predicted"]
            
            for field in field_accuracies.keys():
                if true_t.get(field) == pred_t.get(field):
                    field_accuracies[field] += 1
        
        # Calculate percentages
        for field in field_accuracies:
            field_accuracies[field] = field_accuracies[field] / len(correct_tickets)
        
        print("\nField-level accuracy (for correctly classified tickets):")
        for field, acc in field_accuracies.items():
            print(f"  {field}: {acc:.4f}")
    
    return accuracy, detailed_results

def prepare_model_and_tokenizer():
    """Prepare model with 4-bit quantization and LoRA."""
    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_datasets(samples: List[Dict], tokenizer, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Create train, validation, and test datasets."""
    import random
    random.seed(42)
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Calculate split indices
    total_samples = len(samples)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split data
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    print(f"Dataset splits - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Format samples
    def format_dataset(samples_list):
        formatted = []
        for sample in samples_list:
            text = create_training_prompt(sample)
            formatted.append({"text": text})
        return Dataset.from_list(formatted)
    
    # Create datasets
    return DatasetDict({
        "train": format_dataset(train_samples),
        "validation": format_dataset(val_samples),
        "test": format_dataset(test_samples)
    }), test_samples  # Return test_samples for evaluation

def upload_model_to_hub(model, tokenizer, output_dir: str, model_name: str = "banking-ticket-classifier"):
    """Upload the trained model to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        
        print(f"\nUploading model to Hugging Face Hub as '{model_name}'...")
        
        # Push to hub
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
        
        print(f"✓ Model successfully uploaded to: https://huggingface.co/{model_name}")
        
        # Also save locally
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Model also saved locally to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️ Failed to upload to Hub: {e}")
        print("Model saved locally only.")
        # Ensure local save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def train_model(json_path: Optional[str] = None, csv_path: Optional[str] = None, 
                data_dir: Optional[str] = None, use_synthetic: bool = True):
    """Main training function."""
    print("Preparing model and tokenizer...")
    model, tokenizer = prepare_model_and_tokenizer()
    
    samples = []
    test_samples = []
    
    # Load data based on format
    if data_dir:
        # Load pre-split data
        print(f"Loading pre-split data from {data_dir}...")
        train_path = f"{data_dir}/train_data.json"
        val_path = f"{data_dir}/val_data.json"
        test_path = f"{data_dir}/test_data.json"
        
        with open(train_path, 'r') as f:
            train_samples = json.load(f)
        with open(val_path, 'r') as f:
            val_samples = json.load(f)
        with open(test_path, 'r') as f:
            test_samples = json.load(f)
        
        # Combine train and val for the training dataset
        samples = train_samples + val_samples
        
        print(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
    elif json_path:
        print(f"Loading JSON data from {json_path}...")
        all_samples = load_json_data(json_path)
        # Split the data
        datasets, test_samples = create_datasets(all_samples, tokenizer)
        # Get samples for synthetic augmentation
        samples = all_samples[:int(len(all_samples) * 0.9)]
    elif csv_path:
        print(f"Loading CSV data from {csv_path}...")
        all_samples = load_real_data(csv_path)
        datasets, test_samples = create_datasets(all_samples, tokenizer)
        samples = all_samples[:int(len(all_samples) * 0.9)]
    else:
        print("No data path provided, using only synthetic data")
    
    # Add synthetic data if requested and not using pre-split data
    if use_synthetic and not data_dir:
        print("Generating synthetic training data...")
        synthetic_samples = generate_synthetic_data(num_samples=1000)
        samples.extend(synthetic_samples)
    
    if not samples and not data_dir:
        raise ValueError("No training data available!")
    
    # Create datasets if not using pre-split data
    if data_dir:
        # Format the pre-split data
        def format_dataset_list(samples_list):
            formatted = []
            for sample in samples_list:
                text = create_training_prompt(sample)
                formatted.append({"text": text})
            return formatted
        
        train_formatted = format_dataset_list(train_samples)
        val_formatted = format_dataset_list(val_samples)
        
        datasets = DatasetDict({
            "train": Dataset.from_list(train_formatted),
            "validation": Dataset.from_list(val_formatted)
        })
    else:
        print("Creating datasets...")
        datasets, test_samples = create_datasets(samples, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=500,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        packing=False
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! Model saved to {OUTPUT_DIR}")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    accuracy, results = evaluate_model(model, tokenizer, test_samples, OUTPUT_DIR)
    
    # Upload model to Hugging Face Hub
    upload_model_to_hub(model, tokenizer, OUTPUT_DIR)
    
    return accuracy

# For Unsloth optimization (if available)
def train_with_unsloth():
    """Alternative training with Unsloth for faster training."""
    try:
        from unsloth import FastLanguageModel
        
        # Load model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Apply LoRA with Unsloth
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
        )
        
        print("Using Unsloth for optimized training!")
        # Continue with same training process...
        
    except ImportError:
        print("Unsloth not available, using standard training approach")
        train_model()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train banking ticket classification model")
    parser.add_argument("--json-path", type=str, default=None,
                       help="Path to JSON file with preprocessed training data")
    parser.add_argument("--csv-path", type=str, default=None,
                       help="Path to CSV file with raw training data")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Directory with pre-split train/val/test data")
    parser.add_argument("--no-synthetic", action="store_true",
                       help="Don't use synthetic data augmentation")
    parser.add_argument("--use-unsloth", action="store_true",
                       help="Use Unsloth for faster training")
    
    args = parser.parse_args()
    
    # Default to JSON if no path specified
    if not args.json_path and not args.csv_path and not args.data_dir:
        args.json_path = "training_data.json"
    
    # Check if Unsloth is available and requested
    if args.use_unsloth:
        try:
            import unsloth
            train_with_unsloth()
        except ImportError:
            print("Unsloth not available, using standard training approach")
            accuracy = train_model(
                json_path=args.json_path,
                csv_path=args.csv_path,
                data_dir=args.data_dir,
                use_synthetic=not args.no_synthetic
            )
            print(f"\nFinal test accuracy: {accuracy:.4f}")
    else:
        accuracy = train_model(
            json_path=args.json_path,
            csv_path=args.csv_path,
            data_dir=args.data_dir,
            use_synthetic=not args.no_synthetic
        )
        print(f"\nFinal test accuracy: {accuracy:.4f}")