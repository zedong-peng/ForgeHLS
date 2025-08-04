import os
import torch
import argparse
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
import transformers
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def train(dataset_path, base_model_path, output_dir,
          batch_size=2, learning_rate=1e-5, num_epochs=1,
          lora_r=32, lora_alpha=64, max_length=1024, device=None):

    train_dataset = load_dataset('json', data_files=dataset_path)
    print(f"Loaded dataset: {dataset_path} with {len(train_dataset['train'])} samples")
    print(f"Training output dir: {output_dir}")

    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device}" if isinstance(device, int) or 
                             (isinstance(device, str) and device.isdigit()) else device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Initializing model: {base_model_path}...")
    try:
        print(f"Loading model from: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, local_files_only=True,
            load_in_8bit=False, trust_remote_code=True
        ).to(device)
        print("Model loaded successfully")
        
        print(f"Loading tokenizer from: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, padding_side="left",
            add_eos_token=True, add_bos_token=True,
            local_files_only=True, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token = {tokenizer.pad_token}")
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return

    # Formatting function
    def formatting_func(example):
        return f'''
{example["instruction"]}

```start of code
{example["input"]}
```end of code

```start of prediction
Prediction: {example["output"]}
```end of prediction
'''
    
    # Max length setting
    print(f"Training max length: {max_length}")
    
    # Tokenization function
    def generate_and_tokenize_prompt(prompt):
        result = tokenizer(
            formatting_func(prompt),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Split and process dataset
    train_val_split = train_dataset['train'].train_test_split(test_size=0.0001, seed=42)
    print("Starting dataset tokenization...")
    tokenized_train_dataset = train_val_split['train'].map(generate_and_tokenize_prompt)
    tokenized_val_dataset = train_val_split['test'].map(generate_and_tokenize_prompt)
    print(f"Training samples: {len(tokenized_train_dataset)}, Validation samples: {len(tokenized_val_dataset)}")
    print("Dataset tokenization completed")
    
    # Prepare model for training
    print("Preparing model for training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Print trainable parameters
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    # LoRA configuration
    print(f"Configuring LoRA parameters: r={lora_r}, alpha={lora_alpha}")
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print("LoRA model preparation completed:")
    print_trainable_parameters(model)
    
    # Accelerator setup
    print("Initializing Accelerator...")
    accelerator = Accelerator()
    print(f"Accelerator device count: {accelerator.num_processes}")
    model = accelerator.prepare_model(model)
    print("Model prepared with Accelerator")

    # Training setup
    base_model_name = base_model_path.split('/')[-1]
    model_output_dir = os.path.join(output_dir, base_model_name)
    print(f"Model will be saved to: {model_output_dir}")
    
    # Calculate save frequency
    dataset_size = len(train_val_split['train'])
    save_steps = max(1, dataset_size // 2)
    print(f"Checkpoints will be saved every {save_steps} steps")

    print(f"Training parameters: batch_size={batch_size}, learning_rate={learning_rate}, epochs={num_epochs}")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        args=transformers.TrainingArguments(
            output_dir=model_output_dir,
            warmup_steps=200,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=save_steps // 10,
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=save_steps,
            do_eval=False,
            run_name=f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Disable cache during training
    model.config.use_cache = False

    print("=" * 50)
    print(f"Starting training for {base_model_name} model...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {num_epochs}")
    print("=" * 50)
    
    trainer.train()
    
    print("=" * 50)
    print(f"Training completed! Model saved to {model_output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model with LoRA")
    
    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the training dataset (jsonl file)")
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Base model ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha parameter")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./training_output",
                        help="Directory to save training outputs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (e.g., 'cuda:0', '0', 'cpu')")
    
    args = parser.parse_args()
    
    # Convert device to int if it's a digit string
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    
    train(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        device=args.device
    )