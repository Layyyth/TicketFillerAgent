import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

class BankingTicketEvaluator:
    """Comprehensive evaluation suite for banking ticket model."""
    
    def __init__(self, model_path: str, base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # System prompt
        self.system_prompt = """You are a banking customer service ticket classification and filling assistant. Your role is to:
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
        
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model."""
        print("Loading model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict_single(self, user_input: str) -> Dict:
        """Make a single prediction."""
        # Create prompt
        prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("[/INST]")[-1].strip()
        
        # Extract JSON
        try:
            json_match = re.search(r'\{.*\}', assistant_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Failed to parse response", "raw": assistant_response}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {str(e)}", "raw": assistant_response}
    
    def evaluate_dataset(self, test_data_path: str) -> Dict:
        """Evaluate model on a test dataset."""
        print(f"Loading test data from {test_data_path}...")
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"Evaluating {len(test_data)} samples...")
        
        results = {
            "predictions": [],
            "true_labels": [],
            "detailed_results": [],
            "field_accuracy": {},
            "error_analysis": {
                "parse_errors": 0,
                "domain_errors": 0,
                "field_errors": {}
            }
        }
        
        # Evaluate each sample
        for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            user_input = sample["user_input"]
            true_ticket = sample["ticket_data"]
            
            # Make prediction
            pred_ticket = self.predict_single(user_input)
            
            # Check for errors
            if "error" in pred_ticket and "error" not in true_ticket:
                results["error_analysis"]["parse_errors"] += 1
                results["predictions"].append("error")
                results["true_labels"].append(true_ticket.get("ticket_type", "unknown"))
            elif "error" in true_ticket and "error" in pred_ticket:
                # Correctly identified out-of-domain
                results["predictions"].append("out_of_domain")
                results["true_labels"].append("out_of_domain")
            elif "error" not in true_ticket and "error" not in pred_ticket:
                # Compare ticket types
                true_type = true_ticket.get("ticket_type", "unknown")
                pred_type = pred_ticket.get("ticket_type", "unknown")
                
                results["predictions"].append(pred_type)
                results["true_labels"].append(true_type)
                
                # Field-level comparison
                field_match = {}
                for field in ["severity", "department_impacted", "service_impacted", "preferred_communication"]:
                    field_match[field] = true_ticket.get(field) == pred_ticket.get(field)
                
                # Special handling for assistance_request
                if true_type == "assistance":
                    field_match["assistance_request"] = (
                        "assistance_request" in pred_ticket and
                        pred_ticket.get("assistance_request") is not None
                    )
                
                results["detailed_results"].append({
                    "input": user_input,
                    "true": true_ticket,
                    "predicted": pred_ticket,
                    "type_correct": true_type == pred_type,
                    "field_match": field_match
                })
            else:
                # Misclassified domain
                results["error_analysis"]["domain_errors"] += 1
                results["predictions"].append("domain_error")
                results["true_labels"].append(true_ticket.get("ticket_type", "out_of_domain"))
        
        # Calculate metrics
        results["metrics"] = self._calculate_metrics(results)
        
        return results
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Overall accuracy
        metrics["overall_accuracy"] = accuracy_score(
            results["true_labels"], 
            results["predictions"]
        )
        
        # Classification report
        metrics["classification_report"] = classification_report(
            results["true_labels"], 
            results["predictions"],
            output_dict=True
        )
        
        # Confusion matrix
        unique_labels = sorted(list(set(results["true_labels"] + results["predictions"])))
        metrics["confusion_matrix"] = confusion_matrix(
            results["true_labels"], 
            results["predictions"],
            labels=unique_labels
        ).tolist()
        metrics["confusion_matrix_labels"] = unique_labels
        
        # Field-level accuracy (for correctly classified tickets)
        if results["detailed_results"]:
            correct_tickets = [r for r in results["detailed_results"] if r["type_correct"]]
            
            if correct_tickets:
                field_accuracies = {}
                for field in ["severity", "department_impacted", "service_impacted", 
                            "preferred_communication", "assistance_request"]:
                    matches = sum(1 for r in correct_tickets if r["field_match"].get(field, False))
                    relevant = sum(1 for r in correct_tickets if field in r["field_match"])
                    if relevant > 0:
                        field_accuracies[field] = matches / relevant
                
                metrics["field_accuracies"] = field_accuracies
        
        # Error rates
        total_samples = len(results["true_labels"])
        metrics["error_rates"] = {
            "parse_error_rate": results["error_analysis"]["parse_errors"] / total_samples,
            "domain_error_rate": results["error_analysis"]["domain_errors"] / total_samples
        }
        
        return metrics
    
    def visualize_results(self, results: Dict, output_dir: str = "."):
        """Create visualizations of the evaluation results."""
        metrics = results["metrics"]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = metrics["confusion_matrix"]
        labels = metrics["confusion_matrix_labels"]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # 2. Classification Metrics
        class_report = metrics["classification_report"]
        classes = [c for c in class_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        precisions = [class_report[c]['precision'] for c in classes]
        recalls = [class_report[c]['recall'] for c in classes]
        f1_scores = [class_report[c]['f1-score'] for c in classes]
        
        x = range(len(classes))
        width = 0.25
        
        axes[0, 1].bar([i - width for i in x], precisions, width, label='Precision')
        axes[0, 1].bar(x, recalls, width, label='Recall')
        axes[0, 1].bar([i + width for i in x], f1_scores, width, label='F1-Score')
        
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Classification Metrics by Class')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1.1)
        
        # 3. Field Accuracies
        if "field_accuracies" in metrics:
            fields = list(metrics["field_accuracies"].keys())
            accuracies = list(metrics["field_accuracies"].values())
            
            axes[1, 0].bar(fields, accuracies)
            axes[1, 0].set_xlabel('Fields')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Field-level Accuracy (for correctly classified tickets)')
            axes[1, 0].set_xticklabels(fields, rotation=45)
            axes[1, 0].set_ylim(0, 1.1)
            
            # Add value labels on bars
            for i, v in enumerate(accuracies):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 4. Error Analysis
        error_types = ['Parse Errors', 'Domain Errors']
        error_rates = [
            metrics["error_rates"]["parse_error_rate"],
            metrics["error_rates"]["domain_error_rate"]
        ]
        
        axes[1, 1].pie([1 - sum(error_rates), *error_rates],
                       labels=['Correct', *error_types],
                       autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c', '#f39c12'])
        axes[1, 1].set_title('Error Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/evaluation_visualization.png", dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_dir}/evaluation_visualization.png")
        
        plt.close()
    
    def generate_report(self, results: Dict, output_path: str = "evaluation_report.txt"):
        """Generate a detailed evaluation report."""
        metrics = results["metrics"]
        
        with open(output_path, 'w') as f:
            f.write("=== Banking Ticket Model Evaluation Report ===\n\n")
            
            # Overall Performance
            f.write("1. OVERALL PERFORMANCE\n")
            f.write("-" * 50 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Total Samples Evaluated: {len(results['true_labels'])}\n\n")
            
            # Classification Report
            f.write("2. CLASSIFICATION REPORT\n")
            f.write("-" * 50 + "\n")
            
            class_report = metrics["classification_report"]
            for class_name, class_metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {class_metrics['support']}\n")
            
            # Macro and Weighted Averages
            f.write("\nMacro Average:\n")
            macro_avg = class_report.get('macro avg', {})
            f.write(f"  Precision: {macro_avg.get('precision', 0):.4f}\n")
            f.write(f"  Recall: {macro_avg.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score: {macro_avg.get('f1-score', 0):.4f}\n")
            
            # Field-level Accuracy
            if "field_accuracies" in metrics:
                f.write("\n3. FIELD-LEVEL ACCURACY\n")
                f.write("-" * 50 + "\n")
                f.write("(For correctly classified tickets)\n\n")
                
                for field, accuracy in metrics["field_accuracies"].items():
                    f.write(f"{field}: {accuracy:.4f}\n")
            
            # Error Analysis
            f.write("\n4. ERROR ANALYSIS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Parse Error Rate: {metrics['error_rates']['parse_error_rate']:.4f}\n")
            f.write(f"Domain Error Rate: {metrics['error_rates']['domain_error_rate']:.4f}\n")
            
            # Sample Predictions
            f.write("\n5. SAMPLE PREDICTIONS\n")
            f.write("-" * 50 + "\n")
            
            # Show some correct and incorrect predictions
            if results["detailed_results"]:
                correct_samples = [r for r in results["detailed_results"] if r["type_correct"]][:3]
                incorrect_samples = [r for r in results["detailed_results"] if not r["type_correct"]][:3]
                
                f.write("\nCorrect Predictions:\n")
                for i, sample in enumerate(correct_samples, 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"Input: {sample['input']}\n")
                    f.write(f"True Type: {sample['true']['ticket_type']}\n")
                    f.write(f"Predicted Type: {sample['predicted']['ticket_type']}\n")
                
                f.write("\nIncorrect Predictions:\n")
                for i, sample in enumerate(incorrect_samples, 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"Input: {sample['input']}\n")
                    f.write(f"True Type: {sample['true'].get('ticket_type', 'N/A')}\n")
                    f.write(f"Predicted Type: {sample['predicted'].get('ticket_type', 'N/A')}\n")
                    if 'error' in sample['predicted']:
                        f.write(f"Error: {sample['predicted']['error']}\n")
        
        print(f"Evaluation report saved to {output_path}")
    
    def run_comprehensive_evaluation(self, test_data_path: str, output_dir: str = "."):
        """Run a comprehensive evaluation pipeline."""
        print("\n=== Running Comprehensive Model Evaluation ===\n")
        
        # Evaluate dataset
        results = self.evaluate_dataset(test_data_path)
        
        # Generate visualizations
        self.visualize_results(results, output_dir)
        
        # Generate report
        self.generate_report(results, f"{output_dir}/evaluation_report.txt")
        
        # Save detailed results
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Overall Accuracy: {results['metrics']['overall_accuracy']:.4f}")
        print(f"Parse Error Rate: {results['metrics']['error_rates']['parse_error_rate']:.4f}")
        print(f"Domain Error Rate: {results['metrics']['error_rates']['domain_error_rate']:.4f}")
        
        if "field_accuracies" in results["metrics"]:
            print("\nField Accuracies:")
            for field, acc in results["metrics"]["field_accuracies"].items():
                print(f"  {field}: {acc:.4f}")
        
        return results

def test_model_interactive(model_path: str):
    """Interactive testing mode."""
    evaluator = BankingTicketEvaluator(model_path)
    
    print("\n=== Interactive Model Testing ===")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("\nEnter customer message: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        # Make prediction
        result = evaluator.predict_single(user_input)
        
        # Pretty print result
        print("\nPredicted Ticket:")
        print(json.dumps(result, indent=2))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate banking ticket classification model")
    parser.add_argument("--model-path", type=str, default="./banking-ticket-model",
                       help="Path to the fine-tuned model")
    parser.add_argument("--test-data", type=str, default="test_data.json",
                       help="Path to test dataset (JSON format)")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive testing mode")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Base model name")
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.interactive:
        test_model_interactive(args.model_path)
    else:
        evaluator = BankingTicketEvaluator(args.model_path, args.base_model)
        evaluator.run_comprehensive_evaluation(args.test_data, args.output_dir)

if __name__ == "__main__":
    main()