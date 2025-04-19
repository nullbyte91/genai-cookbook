import logging
import sys
from IPython.display import display, HTML

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import torch
from profiler import setup_trainer_with_memory_tracking, benchmark_models, compare_benchmarks

# Load model and tokenizer
model_name = "openai-community/gpt2-medium" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

# Load dataset
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1%]")

def tokenize_function(examples):
    # Tokenize the texts
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    # Set up labels for causal language modeling (same as input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize both datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Ensure dataset format is compatible with the trainer
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define models to benchmark
models_to_benchmark = [
    (model_name, AutoModelForCausalLM.from_pretrained(model_name)),
]

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,  
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    remove_unused_columns=True,    
    report_to="none",            
)

# Define optimizer configurations to test
optimizer_configs = [
    {"name": "baseline", "fp16": False, "gradient_checkpointing": False},
    {"name": "adafactor",
     "fp16": True, 
     "gradient_checkpointing": True, 
     "gradient_accumulation_steps": 1,
     "optim": "adafactor"},
    {"name": "adamw_bnb_8bit",
     "fp16": True, 
     "gradient_checkpointing": True, 
     "gradient_accumulation_steps": 1,
     "optim": "adamw_bnb_8bit"},
    {"name": "paged_adamw_8bit",
     "fp16": True, 
     "gradient_checkpointing": True, 
     "gradient_accumulation_steps": 1,
     "optim": "paged_adamw_8bit"},
]

# Run benchmarks with the properly processed datasets
results = benchmark_models(
    models_to_benchmark=models_to_benchmark,
    dataset=tokenized_train,        
    eval_dataset=tokenized_eval,    
    training_args=training_args,
    optimizer_configs=optimizer_configs,
    output_dir="./benchmark_results",
)

# Compare and visualize the results
compare_benchmarks(results, output_dir="./benchmark_results")
