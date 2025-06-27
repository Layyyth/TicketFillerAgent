import json
import random
from typing import List, Dict, Tuple

def split_json_data(
    input_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = ".",
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split JSON data into train, validation, and test sets.
    
    Args:
        input_path: Path to the full JSON dataset
        train_ratio: Percentage for training (default: 80%)
        val_ratio: Percentage for validation (default: 10%)
        test_ratio: Percentage for testing (default: 10%)
        output_dir: Directory to save the split files
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Set random seed
    random.seed(seed)
    
    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split indices
    total_samples = len(data)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Add out-of-domain examples to test set
    out_of_domain_examples = [
        {
            "user_input": "Can you give me a recipe for chocolate cake?",
            "ticket_data": {"error": "Request outside banking domain"}
        },
        {
            "user_input": "What's the weather forecast for tomorrow?",
            "ticket_data": {"error": "Request outside banking domain"}
        },
        {
            "user_input": "Tell me a joke about programmers",
            "ticket_data": {"error": "Request outside banking domain"}
        },
        {
            "user_input": "How do I train a neural network?",
            "ticket_data": {"error": "Request outside banking domain"}
        },
        {
            "user_input": "What movies are playing this weekend?",
            "ticket_data": {"error": "Request outside banking domain"}
        }
    ]
    
    test_data.extend(out_of_domain_examples)
    
    # Analyze distribution
    def analyze_distribution(dataset: List[Dict], name: str):
        ticket_types = {}
        severities = {}
        
        for sample in dataset:
            ticket = sample.get("ticket_data", {})
            
            # Skip error samples for this analysis
            if "error" in ticket:
                continue
            
            # Count ticket types
            ticket_type = ticket.get("ticket_type", "unknown")
            ticket_types[ticket_type] = ticket_types.get(ticket_type, 0) + 1
            
            # Count severities
            severity = ticket.get("severity", "unknown")
            severities[severity] = severities.get(severity, 0) + 1
        
        print(f"\n{name} Set Analysis:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Ticket types: {ticket_types}")
        print(f"  Severities: {severities}")
    
    # Analyze each split
    analyze_distribution(train_data, "Train")
    analyze_distribution(val_data, "Validation")
    analyze_distribution(test_data, "Test")
    
    # Save splits
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    train_path = os.path.join(output_dir, "train_data.json")
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"\nTrain data saved to {train_path}")
    
    # Save validation data
    val_path = os.path.join(output_dir, "val_data.json")
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Validation data saved to {val_path}")
    
    # Save test data
    test_path = os.path.join(output_dir, "test_data.json")
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Test data saved to {test_path}")
    
    return train_data, val_data, test_data

def create_challenge_test_set(output_path: str = "challenge_test_data.json"):
    """
    Create a challenging test set with edge cases and difficult examples.
    """
    challenge_data = [
        # Mixed requests
        {
            "user_input": "I lost my card and also what's the weather like?",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "Lost card report",
                "description": "Customer reported lost card. Non-banking query ignored.",
                "severity": "high",
                "department_impacted": "Card Services",
                "service_impacted": "Card Management",
                "preferred_communication": "phone"
            }
        },
        # Urgent situations
        {
            "user_input": "HELP! Someone is using my account RIGHT NOW and making withdrawals!!!",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "Active fraud - unauthorized withdrawals",
                "description": "URGENT: Customer reports active fraud with ongoing unauthorized withdrawals. Immediate action required.",
                "severity": "critical",
                "department_impacted": "Security Department",
                "service_impacted": "Account Security",
                "preferred_communication": "phone"
            }
        },
        # Complex assistance request
        {
            "user_input": "I need help consolidating my three loans into one and also setting up automatic payments",
            "ticket_data": {
                "ticket_type": "assistance",
                "title": "Loan consolidation and autopay setup",
                "description": "Customer needs assistance with loan consolidation and automatic payment setup.",
                "severity": "medium",
                "department_impacted": "Loan Department",
                "service_impacted": "Loan Services",
                "preferred_communication": "email",
                "assistance_request": "loan consolidation"
            }
        },
        # Ambiguous severity
        {
            "user_input": "My statement has a small error but it's been happening for months",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "Recurring statement errors",
                "description": "Customer reports small but recurring errors in statements over multiple months.",
                "severity": "medium",
                "department_impacted": "Account Services",
                "service_impacted": "Statement Services",
                "preferred_communication": "email"
            }
        },
        # Multiple departments affected
        {
            "user_input": "The ATM took my deposit but didn't credit my account and the app won't let me report it",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "ATM deposit not credited, app issue",
                "description": "ATM accepted deposit but didn't credit account. Mobile app failing to submit reports.",
                "severity": "high",
                "department_impacted": "ATM Services",
                "service_impacted": "ATM Services",
                "preferred_communication": "phone"
            }
        },
        # Subtle out-of-domain
        {
            "user_input": "I need help with my credit... score on the video game I'm playing",
            "ticket_data": {"error": "Request outside banking domain"}
        },
        # Very brief input
        {
            "user_input": "Card declined",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "Card declined",
                "description": "Customer reports card was declined. Further details needed.",
                "severity": "medium",
                "department_impacted": "Card Services",
                "service_impacted": "Card Services",
                "preferred_communication": "email"
            }
        },
        # Technical banking terms
        {
            "user_input": "My ACH transfer was returned with code R01",
            "ticket_data": {
                "ticket_type": "complaint",
                "title": "ACH transfer returned - R01",
                "description": "Customer's ACH transfer returned with code R01 (Insufficient Funds).",
                "severity": "medium",
                "department_impacted": "Transaction Processing",
                "service_impacted": "Transfer Services",
                "preferred_communication": "email"
            }
        }
    ]
    
    # Save challenge set
    with open(output_path, 'w') as f:
        json.dump(challenge_data, f, indent=2)
    
    print(f"\nChallenge test set saved to {output_path}")
    print(f"Contains {len(challenge_data)} challenging examples")
    
    return challenge_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare test data for model evaluation")
    parser.add_argument("--input", type=str, default="training_data.json",
                       help="Path to the full JSON dataset")
    parser.add_argument("--output-dir", type=str, default="./data_splits",
                       help="Directory to save the split files")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Percentage for training (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Percentage for validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Percentage for testing (default: 0.1)")
    parser.add_argument("--create-challenge", action="store_true",
                       help="Create a challenge test set with edge cases")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: ratios sum to {total_ratio}, not 1.0")
    
    # Split data
    train_data, val_data, test_data = split_json_data(
        args.input,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.output_dir,
        args.seed
    )
    
    # Create challenge set if requested
    if args.create_challenge:
        challenge_path = os.path.join(args.output_dir, "challenge_test_data.json")
        create_challenge_test_set(challenge_path)
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    import os
    main()