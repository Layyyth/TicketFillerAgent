import json
import torch
import pandas as pd
from typing import Dict, List, Optional
import re
import os
import argparse
import random

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
from huggingface_hub import login
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HF_HUB_REPO_ID = "LaythAbuJafar/FillerAgent"  # Target Hugging Face Hub repository
OUTPUT_DIR = "./banking-ticket-model"
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# Ticket Schema (using string literals for JSON-like structure)
TICKET_SCHEMA = {
    "ticket_type": ["complaint", "inquiry", "assistance"],
    "title": "string",
    "description": "string",
    "severity": ["low", "medium", "high", "critical"],
    "department_impacted": "string",
    "service_impacted": "string",
    "preferred_communication": ["email", "phone", "chat", "in-person"],
    "assistance_request": "string"  # Only for assistance tickets
}

# System prompt for banking domain
SYSTEM_PROMPT = """You are a banking customer service ticket classification and filling assistant. Your role is to:
1. Analyze customer inputs and extract relevant information.
2. Fill ticket fields accurately based on the customer's request.
3. Stay strictly within the banking and financial services domain.
4. Reject any requests outside of banking support.

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

def setup_huggingface_login():
    """Setup Hugging Face login for model downloads and uploads."""
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("Using HF_TOKEN environment variable for Hugging Face login...")
        login(token=hf_token)
    else:
        print("HF_TOKEN not found. Attempting interactive login for model deployment.")
        try:
            login()
        except Exception as e:
            print(f"Login failed: {e}. Model deployment will fail.")

def check_gpu_compatibility():
    """Check GPU compatibility. Assumes A100 is target."""
    if not torch.cuda.is_available():
        print("❌ CRITICAL: CUDA not available. This script requires a GPU.")
        raise SystemExit("CUDA is not available. Please check your installation.")

    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # No specific warnings for A100 as it's a well-supported GPU for ML.
    # Flash Attention 2 is compatible with Ampere (A100) architecture.

def create_training_prompt(example: Dict, tokenizer) -> str:
    """
    Create a training prompt from an example using the tokenizer's chat template.
    This is more robust than manual string formatting.
    """
    user_input = example['user_input']
    ticket_data = example['ticket_data']
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": json.dumps(ticket_data, indent=2)}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False)

def load_json_data(json_path: str) -> List[Dict]:
    """Load and validate preprocessed JSON data."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")

    required_fields = ["user_input", "ticket_data"]
    ticket_fields = ["ticket_type", "title", "description", "severity", "department_impacted", "service_impacted", "preferred_communication"]
    valid_samples = []
    skipped = 0

    for idx, sample in enumerate(data):
        if not all(field in sample for field in required_fields):
            skipped += 1
            continue
        
        ticket_data = sample["ticket_data"]
        if "error" in ticket_data: # Handle out-of-domain examples
            valid_samples.append(sample)
            continue
        
        if not all(field in ticket_data for field in ticket_fields):
            skipped += 1
            continue
        
        valid_samples.append(sample)
    
    print(f"Validation complete. Valid samples: {len(valid_samples)}, Skipped: {skipped}")
    return valid_samples

def load_real_data(csv_path: str) -> List[Dict]:
    """Load and process real complaints data from a CSV file."""
    print(f"Loading and processing data from {csv_path}...")
    df = pd.read_csv(csv_path)
    samples = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            output_data = json.loads(row['output'])
            ticket_data = {
                "ticket_type": output_data.get("Ticket Type", "inquiry").lower(),
                "title": output_data.get("Title", ""),
                "description": output_data.get("Description", ""),
                "severity": output_data.get("Severity", "medium").lower(),
                "department_impacted": output_data.get("Department Impacted", "Customer Service"),
                "service_impacted": output_data.get("Service Impacted", "General Banking"),
                "preferred_communication": output_data.get("Preferred Communication", "email").lower()
            }
            if ticket_data["ticket_type"] == "assistance":
                ticket_data["assistance_request"] = "general assistance"
            
            samples.append({"user_input": row['input'], "ticket_data": ticket_data})
        except (json.JSONDecodeError, AttributeError):
            skipped += 1
    
    print(f"Loaded {len(samples)} valid samples, skipped {skipped} invalid rows.")
    return samples

def generate_synthetic_data() -> List[Dict]:
    """Generate a small set of synthetic edge cases."""
    print("Generating synthetic edge case data...")
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
        },
        {
            "user_input": "What's the weather like today?",
            "ticket_data": {"error": "Request outside banking domain"}
        }
    ]
    return edge_cases

def prepare_model_and_tokenizer():
    """Prepare model with 4-bit quantization, LoRA, and Flash Attention, optimized for A100."""
    print("Preparing model and tokenizer...")

    # Quantization config using FP16 compute dtype for A100
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 # Changed to float16 for A100 efficiency
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Use Flash Attention 2 for speed (compatible with A100)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def create_datasets(samples: List[Dict], tokenizer, train_ratio: float = 0.9):
    """Create train and test datasets."""
    random.seed(42)
    random.shuffle(samples)

    train_end = int(len(samples) * train_ratio)
    train_samples = samples[:train_end]
    test_samples = samples[train_end:]
    
    val_samples = test_samples[:int(len(test_samples) * 0.5)]

    print(f"Dataset splits - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    def format_dataset(samples_list):
        return Dataset.from_list([{"text": create_training_prompt(s, tokenizer)} for s in samples_list])

    dataset_dict = DatasetDict({
        "train": format_dataset(train_samples),
        "validation": format_dataset(val_samples)
    })
    
    return dataset_dict, test_samples

def train_model(model, tokenizer, datasets):
    """Configure and run the SFTTrainer, with automatic deployment to Hub."""
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=True, # Changed to FP16 for A100 efficiency (was bf16 for RTX Pro 6000)
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        push_to_hub=True,  # Enable pushing to the Hub
        hub_model_id=HF_HUB_REPO_ID,  # Specify the repo ID
        hub_strategy="every_save",  # Push on every save
    )

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
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model locally...")
    trainer.save_model(OUTPUT_DIR)
    
    print(f"Training complete! Model saved to {OUTPUT_DIR}")
    return trainer

def evaluate_model(model, tokenizer, test_samples: List[Dict]):
    """Evaluate the model's performance on the test set."""
    print("\n=== Final Model Evaluation on Test Set ===")
    predictions = []
    true_labels = []
    
    model.eval()
    for sample in test_samples:
        if "error" in sample["ticket_data"]: continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["user_input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("[/INST]")[-1].strip()

        try:
            json_match = re.search(r'\{.*\}', assistant_response, re.DOTALL)
            pred_ticket = json.loads(json_match.group()) if json_match else {}
            predictions.append(pred_ticket.get("ticket_type", "parse_error"))
        except json.JSONDecodeError:
            predictions.append("json_error")
            
        true_labels.append(sample["ticket_data"].get("ticket_type"))

    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Ticket Type Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, zero_division=0))

def main(args):
    """Main execution function."""
    setup_huggingface_login()
    check_gpu_compatibility()
    
    model, tokenizer = prepare_model_and_tokenizer()

    all_samples = []
    if args.json_path:
        all_samples.extend(load_json_data(args.json_path))
    if args.csv_path:
        all_samples.extend(load_real_data(args.csv_path))
    if args.use_synthetic:
        all_samples.extend(generate_synthetic_data())
    
    if not all_samples:
        raise ValueError("No data loaded. Please provide a data source.")
        
    datasets, test_samples = create_datasets(all_samples, tokenizer)
    
    trainer = train_model(model, tokenizer, datasets)
    
    evaluate_model(trainer.model, tokenizer, test_samples)
    
    print("\nUploading final model and tokenizer to Hugging Face Hub...")
    trainer.push_to_hub()
    print(f"✅ Successfully uploaded model to {HF_HUB_REPO_ID}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and deploy a banking ticket classification model.")
    parser.add_argument("--json-path", type=str, help="Path to JSON file with training data.")
    parser.add_argument("--csv-path", type=str, help="Path to CSV file with raw training data.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data augmentation.")
    
    class Args:
        def __init__(self):
            self.json_path = "training_data.json"
            self.csv_path = None
            self.use_synthetic = True
            
    parsed_args = parser.parse_args()
    
    if not parsed_args.json_path and not parsed_args.csv_path:
       print("No data path specified, using default 'training_data.json'")
       args = Args()
    else:
       args = parsed_args
       
    main(args)
