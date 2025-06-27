import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import Dict, List, Tuple

def analyze_complaints_data(csv_path: str) -> pd.DataFrame:
    """Analyze the complaints data and generate insights."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse JSON outputs
    parsed_data = []
    parse_errors = 0
    
    for idx, row in df.iterrows():
        try:
            output_data = json.loads(row['output'])
            output_data['original_input'] = row['input']
            output_data['row_index'] = idx
            parsed_data.append(output_data)
        except json.JSONDecodeError:
            parse_errors += 1
    
    print(f"Successfully parsed: {len(parsed_data)} records")
    print(f"Parse errors: {parse_errors}")
    
    # Convert to DataFrame for analysis
    parsed_df = pd.DataFrame(parsed_data)
    
    # Analyze ticket types
    print(f"\n=== Ticket Type Distribution ===")
    ticket_types = parsed_df['Ticket Type'].value_counts()
    print(ticket_types)
    
    # Analyze severity distribution
    print(f"\n=== Severity Distribution ===")
    severities = parsed_df['Severity'].value_counts()
    print(severities)
    
    # Analyze departments
    print(f"\n=== Department Distribution ===")
    departments = parsed_df['Department Impacted'].value_counts()
    print(departments)
    
    # Analyze services
    print(f"\n=== Service Distribution ===")
    services = parsed_df['Service Impacted'].value_counts()
    print(services)
    
    # Check for assistance requests
    assistance_tickets = parsed_df[parsed_df['Ticket Type'].str.contains('assistance', case=False, na=False)]
    print(f"\n=== Assistance Tickets ===")
    print(f"Total assistance tickets: {len(assistance_tickets)}")
    
    # Analyze input lengths
    input_lengths = parsed_df['original_input'].str.len()
    print(f"\n=== Input Length Statistics ===")
    print(f"Mean: {input_lengths.mean():.2f}")
    print(f"Median: {input_lengths.median():.2f}")
    print(f"Min: {input_lengths.min()}")
    print(f"Max: {input_lengths.max()}")
    
    return parsed_df

def clean_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the parsed data."""
    df_clean = df.copy()
    
    # Standardize ticket types
    ticket_type_mapping = {
        'complaint': 'complaint',
        'complaints': 'complaint',
        'complaining': 'complaint',
        'compla intimation': 'complaint',
        'inquiry': 'inquiry',
        'enquiray': 'inquiry',
        'assistance': 'assistance',
        'request for assistance': 'assistance'
    }
    
    df_clean['Ticket Type'] = df_clean['Ticket Type'].str.lower().str.strip()
    df_clean['Ticket Type'] = df_clean['Ticket Type'].map(ticket_type_mapping).fillna('complaint')
    
    # Standardize severity
    severity_mapping = {
        'low': 'low',
        'medium': 'medium',
        ' medium': 'medium',
        'high': 'high',
        'High': 'high',
        'critical': 'critical'
    }
    
    df_clean['Severity'] = df_clean['Severity'].str.lower().str.strip()
    df_clean['Severity'] = df_clean['Severity'].map(severity_mapping).fillna('medium')
    
    # Clean preferred communication
    df_clean['Preferred Communication'] = df_clean['Preferred Communication'].str.lower().str.strip()
    df_clean.loc[df_clean['Preferred Communication'] == 'not specified', 'Preferred Communication'] = 'email'
    
    return df_clean

def extract_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract common patterns from the data."""
    patterns = {
        'complaint_keywords': [],
        'inquiry_keywords': [],
        'assistance_keywords': [],
        'high_severity_keywords': []
    }
    
    # Extract keywords for each ticket type
    for ticket_type in ['complaint', 'inquiry', 'assistance']:
        subset = df[df['Ticket Type'] == ticket_type]
        if len(subset) > 0:
            # Combine all inputs for this type
            all_text = ' '.join(subset['original_input'].tolist()).lower()
            
            # Extract common words (excluding stopwords)
            words = re.findall(r'\b[a-z]+\b', all_text)
            word_freq = Counter(words)
            
            # Filter out common stopwords
            stopwords = {'i', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'have',
                        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                        'may', 'might', 'must', 'can', 'my', 'your', 'me', 'you', 'it', 'this',
                        'that', 'what', 'how', 'when', 'where', 'why', 'who'}
            
            # Get top keywords
            keywords = [word for word, count in word_freq.most_common(20) 
                       if word not in stopwords and len(word) > 2]
            
            patterns[f'{ticket_type}_keywords'] = keywords[:10]
    
    # Extract high severity keywords
    high_severity = df[df['Severity'].isin(['high', 'critical'])]
    if len(high_severity) > 0:
        high_text = ' '.join(high_severity['original_input'].tolist()).lower()
        words = re.findall(r'\b[a-z]+\b', high_text)
        word_freq = Counter(words)
        
        stopwords = {'i', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        patterns['high_severity_keywords'] = [word for word, count in word_freq.most_common(15) 
                                             if word not in stopwords and len(word) > 2][:10]
    
    return patterns

def generate_data_quality_report(df: pd.DataFrame, output_path: str = "data_quality_report.txt"):
    """Generate a comprehensive data quality report."""
    with open(output_path, 'w') as f:
        f.write("=== Banking Ticket Data Quality Report ===\n\n")
        
        # Basic statistics
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Unique Ticket Types: {df['Ticket Type'].nunique()}\n")
        f.write(f"Unique Severities: {df['Severity'].nunique()}\n\n")
        
        # Missing values
        f.write("Missing Values:\n")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                f.write(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)\n")
        
        # Ticket type distribution
        f.write("\nTicket Type Distribution:\n")
        for tt, count in df['Ticket Type'].value_counts().items():
            f.write(f"  {tt}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Severity distribution
        f.write("\nSeverity Distribution:\n")
        for sev, count in df['Severity'].value_counts().items():
            f.write(f"  {sev}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Department distribution
        f.write("\nDepartment Distribution:\n")
        for dept, count in df['Department Impacted'].value_counts().head(10).items():
            f.write(f"  {dept}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Input length analysis
        input_lengths = df['original_input'].str.len()
        f.write(f"\nInput Length Analysis:\n")
        f.write(f"  Mean: {input_lengths.mean():.2f} characters\n")
        f.write(f"  Median: {input_lengths.median():.2f} characters\n")
        f.write(f"  Min: {input_lengths.min()} characters\n")
        f.write(f"  Max: {input_lengths.max()} characters\n")
        f.write(f"  Std Dev: {input_lengths.std():.2f} characters\n")
        
        # Check for potential issues
        f.write("\nPotential Data Quality Issues:\n")
        
        # Check for very short inputs
        very_short = df[input_lengths < 10]
        if len(very_short) > 0:
            f.write(f"  - {len(very_short)} records with very short inputs (<10 chars)\n")
        
        # Check for duplicates
        duplicates = df['original_input'].duplicated().sum()
        if duplicates > 0:
            f.write(f"  - {duplicates} duplicate inputs found\n")
        
        # Check for non-standard values
        non_standard_types = df[~df['Ticket Type'].isin(['complaint', 'inquiry', 'assistance'])]
        if len(non_standard_types) > 0:
            f.write(f"  - {len(non_standard_types)} records with non-standard ticket types\n")
    
    print(f"Data quality report saved to {output_path}")

def create_training_ready_dataset(df: pd.DataFrame, output_path: str = "training_data.json"):
    """Create a training-ready dataset in the format expected by the model."""
    training_data = []
    
    for idx, row in df.iterrows():
        # Create the expected format
        ticket_data = {
            "ticket_type": row['Ticket Type'],
            "title": row.get('Title', ''),
            "description": row.get('Description', ''),
            "severity": row['Severity'],
            "department_impacted": row.get('Department Impacted', 'Customer Service'),
            "service_impacted": row.get('Service Impacted', 'General Banking'),
            "preferred_communication": row.get('Preferred Communication', 'email')
        }
        
        # Add assistance_request for assistance tickets
        if row['Ticket Type'] == 'assistance':
            # Try to extract from description
            desc_lower = row.get('Description', '').lower()
            if 'loan' in desc_lower:
                ticket_data['assistance_request'] = 'loan assistance'
            elif 'mortgage' in desc_lower:
                ticket_data['assistance_request'] = 'mortgage assistance'
            elif 'account' in desc_lower:
                ticket_data['assistance_request'] = 'account assistance'
            else:
                ticket_data['assistance_request'] = 'general assistance'
        
        training_data.append({
            "user_input": row['original_input'],
            "ticket_data": ticket_data
        })
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Training data saved to {output_path}")
    return training_data

def visualize_data_distribution(df: pd.DataFrame, save_path: str = "data_visualization.png"):
    """Create visualizations of the data distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Ticket type distribution
    ticket_counts = df['Ticket Type'].value_counts()
    axes[0, 0].pie(ticket_counts.values, labels=ticket_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Ticket Type Distribution')
    
    # Severity distribution
    severity_counts = df['Severity'].value_counts()
    axes[0, 1].bar(severity_counts.index, severity_counts.values)
    axes[0, 1].set_title('Severity Distribution')
    axes[0, 1].set_xlabel('Severity')
    axes[0, 1].set_ylabel('Count')
    
    # Input length distribution
    input_lengths = df['original_input'].str.len()
    axes[1, 0].hist(input_lengths, bins=50, edgecolor='black')
    axes[1, 0].set_title('Input Length Distribution')
    axes[1, 0].set_xlabel('Character Count')
    axes[1, 0].set_ylabel('Frequency')
    
    # Top departments
    top_depts = df['Department Impacted'].value_counts().head(10)
    axes[1, 1].barh(top_depts.index, top_depts.values)
    axes[1, 1].set_title('Top 10 Departments Impacted')
    axes[1, 1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
def create_balanced_dataset(df: pd.DataFrame, target_samples_per_class: int = 5000) -> pd.DataFrame:
    """Create a balanced dataset by oversampling minority classes."""
    balanced_data = []
    
    for ticket_type in df['Ticket Type'].unique():
        type_data = df[df['Ticket Type'] == ticket_type]
        current_count = len(type_data)
        
        if current_count < target_samples_per_class:
            # Oversample
            additional_samples = target_samples_per_class - current_count
            oversampled = type_data.sample(n=additional_samples, replace=True, random_state=42)
            balanced_data.append(pd.concat([type_data, oversampled]))
        else:
            # Undersample
            balanced_data.append(type_data.sample(n=target_samples_per_class, random_state=42))
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"Balanced dataset created with {len(balanced_df)} samples")
    print(f"Distribution: {balanced_df['Ticket Type'].value_counts()}")
    
    return balanced_df

def main():
    """Main function to run all preprocessing steps."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and preprocess banking complaints data")
    parser.add_argument("--csv-path", type=str, default="complaints_20k_with_descriptions.csv",
                       help="Path to the CSV file")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for processed files")
    parser.add_argument("--balance-data", action="store_true",
                       help="Create a balanced dataset")
    parser.add_argument("--visualize", action="store_true",
                       help="Create data visualizations")
    
    args = parser.parse_args()
    
    # Analyze the data
    print("Analyzing complaints data...")
    parsed_df = analyze_complaints_data(args.csv_path)
    
    # Clean and standardize
    print("\nCleaning and standardizing data...")
    clean_df = clean_and_standardize_data(parsed_df)
    
    # Extract patterns
    print("\nExtracting patterns...")
    patterns = extract_patterns(clean_df)
    print("\nExtracted patterns:")
    for pattern_type, keywords in patterns.items():
        print(f"{pattern_type}: {', '.join(keywords[:5])}")
    
    # Generate quality report
    print("\nGenerating data quality report...")
    generate_data_quality_report(clean_df, f"{args.output_dir}/data_quality_report.txt")
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        visualize_data_distribution(clean_df, f"{args.output_dir}/data_visualization.png")
    
    # Balance dataset if requested
    if args.balance_data:
        print("\nCreating balanced dataset...")
        clean_df = create_balanced_dataset(clean_df)
    
    # Create training-ready dataset
    print("\nCreating training-ready dataset...")
    training_data = create_training_ready_dataset(clean_df, f"{args.output_dir}/training_data.json")
    
    print(f"\nPreprocessing complete! {len(training_data)} samples ready for training.")

if __name__ == "__main__":
    main()