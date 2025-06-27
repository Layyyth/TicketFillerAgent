# Banking Ticket Classification Model Trainer

This repository contains a fine-tuned language model for classifying and filling banking customer service tickets. The model is based on Mistral-7B-Instruct-v0.2 and uses LoRA (Low-Rank Adaptation) for efficient training.

## Features

- **Multi-class Classification**: Classifies tickets as complaint, inquiry, or assistance
- **Structured Output**: Generates JSON tickets with all required fields
- **Domain Validation**: Rejects non-banking requests
- **Efficient Training**: Uses 4-bit quantization and LoRA for memory efficiency
- **Comprehensive Evaluation**: Provides detailed accuracy metrics and confusion matrices

## Model Architecture

- **Base Model**: Mistral-7B-Instruct-v0.2
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for memory efficiency
- **Training Framework**: TRL (Transformer Reinforcement Learning)

## Ticket Schema

The model generates tickets with the following structure:

```json
{
  "ticket_type": "complaint|inquiry|assistance",
  "title": "Brief summary of the issue",
  "description": "Detailed description",
  "severity": "low|medium|high|critical",
  "department_impacted": "The bank department affected",
  "service_impacted": "The specific service affected",
  "preferred_communication": "email|phone|chat|in-person",
  "assistance_request": "Specific assistance needed (only for assistance tickets)"
}
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. **Setup Hugging Face Login** (Required for model downloads):
   
   **Option A: Using Environment Variable (Recommended)**
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```
   
   **Option B: Interactive Login**
   ```bash
   python -c "from huggingface_hub import login; login()"
   ```
   
   Get your token from: https://huggingface.co/settings/tokens

3. Verify the installation:
```bash
python test_code.py
```

## Usage

### Basic Training

Train the model using the default training data:

```bash
python model_training.py
```

### Advanced Training Options

```bash
# Train with specific data file
python model_training.py --json-path training_data.json

# Train with CSV data
python model_training.py --csv-path complaints_data.csv

# Train with pre-split data
python model_training.py --data-dir ./data_splits

# Disable synthetic data augmentation
python model_training.py --no-synthetic

# Use Unsloth for faster training (if available)
python model_training.py --use-unsloth
```

### Training Configuration

Key training parameters (modifiable in the script):

- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Max Length**: 2048 tokens
- **Learning Rate**: 2e-4
- **Batch Size**: 4
- **Epochs**: 3
- **LoRA Rank**: 32

## Data Format

### JSON Training Data

The training data should be in the following format:

```json
[
  {
    "user_input": "Customer's request text",
    "ticket_data": {
      "ticket_type": "complaint",
      "title": "Issue title",
      "description": "Detailed description",
      "severity": "high",
      "department_impacted": "Customer Service",
      "service_impacted": "Online Banking",
      "preferred_communication": "email"
    }
  }
]
```

### CSV Training Data

For CSV format, the data should have `input` and `output` columns where `output` contains JSON ticket data.

## Model Output

The trained model will be saved to `./banking-ticket-model/` and includes:

- Fine-tuned model weights
- Tokenizer configuration
- Training logs
- Evaluation results

## Evaluation

The model evaluation provides:

- Overall accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Field-level accuracy for correctly classified tickets
- Detailed results saved to JSON

## System Requirements

- **GPU**: Recommended 16GB+ VRAM (for full model)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space
- **Python**: 3.8+

### ⚠️ CUDA Compatibility Issues

**RTX Pro 6000 Blackwell GPU Users:**
- This GPU uses CUDA capability `sm_120` which may not be supported by current PyTorch versions
- The script will automatically fall back to CPU training if GPU compatibility issues are detected
- For GPU training, try installing PyTorch with CUDA 12.4+ support:
  ```bash
  python fix_cuda_installation.py
  ```

**Alternative Solutions:**
1. **CPU Training**: Slower but guaranteed to work
2. **Different GPU**: Use RTX 4090, A100, V100, or other compatible GPUs
3. **Cloud Platforms**: Google Colab, AWS, or other cloud services with compatible GPUs

## Performance

With the current configuration:
- Training time: ~2-4 hours on RTX 4090
- Memory usage: ~12GB VRAM
- Model size: ~4GB (quantized)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Import Errors**: Install missing packages with `pip install -r requirements.txt`
3. **Data Loading Issues**: Verify JSON format and file paths

### Memory Optimization

For limited GPU memory:
- Reduce `BATCH_SIZE` to 2 or 1
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Use CPU offloading if available

## License

This project uses Mistral-7B-Instruct-v0.2 which has a commercial-friendly license.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test script output
3. Verify your data format
4. Check system requirements 