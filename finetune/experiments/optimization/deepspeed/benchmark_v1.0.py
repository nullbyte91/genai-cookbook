import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

# Define a thorough memory cleanup function
def clean_memory():
    """Thoroughly clean up GPU and CPU memory"""
    logger.info("Performing thorough memory cleanup...")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Run garbage collection multiple times
    import gc
    gc.collect()
    gc.collect()
    
    # Force CUDA synchronization
    torch.cuda.synchronize()
    
    # Wait a bit to ensure memory is freed
    import time
    time.sleep(3)
    
    # Print current memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        logger.info(f"GPU Memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
        
# Import our custom modules - using absolute imports
from profiler import (
    setup_trainer_with_memory_tracking, 
    benchmark_models, 
    compare_benchmarks,
    MemoryTracker
)
from deepspeed_utils import create_deepspeed_config, save_deepspeed_config

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Set output directory
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model_name = "openai-community/gpt2-medium"  # or any other model you want to benchmark
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset - using a small subset for benchmarking
    logger.info("Loading dataset")
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1%]")

    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        # Set up labels for causal language modeling (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Tokenize both datasets
    logger.info("Tokenizing datasets")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Ensure dataset format is compatible with the trainer
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Define models to benchmark
    logger.info("Creating model instances for benchmarking")
    models_to_benchmark = [
        (model_name, AutoModelForCausalLM.from_pretrained(model_name)),
    ]

    # Define base training arguments
    base_training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Use a small number for benchmarking
        per_device_train_batch_size=1,  # Start small
        per_device_eval_batch_size=1,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        remove_unused_columns=True,
        report_to="none",  # Disable external reporting
    )

    # Define configurations to benchmark
    # This includes standard optimizations and DeepSpeed configurations
    optimizer_configs = [
        # Baseline
        {
            "name": "baseline",
            "fp16": False,
            "gradient_checkpointing": False,
        },
        
        # Standard optimizations
        {
            "name": "fp16_gradient_checkpointing",
            "fp16": True,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 4,
        },
        
        # DeepSpeed ZeRO-1
        {
            "name": "deepspeed_zero1",
            "fp16": True,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 4,
            "deepspeed_stage": 1,
        },
        
        # DeepSpeed ZeRO-2 with optimizer offloading
        {
            "name": "deepspeed_zero2_offload",
            "fp16": True,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 4,
            "deepspeed_stage": 2,
            "offload_optimizer": True,
        }
    ]

    
    optimizer_configs_Zero_Stage_3 = [
        # DeepSpeed ZeRO-3 with full offloading
        {
            "name": "deepspeed_zero3_offload",
            "fp16": True,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 4,
            "deepspeed_stage": 3,
            "offload_optimizer": True,
            "offload_param": True,
        },
    ]
    # Run the benchmarks with our local benchmark_models function
    logger.info("Starting benchmarks")
    results = benchmark_models(
        models_to_benchmark=models_to_benchmark,
        dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        training_args=base_training_args,
        optimizer_configs=optimizer_configs,
        output_dir=output_dir,
    )
    
    clean_memory()

    results = benchmark_models(
        models_to_benchmark=models_to_benchmark,
        dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        training_args=base_training_args,
        optimizer_configs=optimizer_configs_Zero_Stage_3,
        output_dir=output_dir,
    )
    
    # Compare and visualize the benchmark results
    logger.info("Generating benchmark comparisons")
    comparison_data = compare_benchmarks(results, output_dir=output_dir)
    
    logger.info(f"All benchmarks completed. Results saved to {output_dir}")
    
if __name__ == "__main__":
    main()