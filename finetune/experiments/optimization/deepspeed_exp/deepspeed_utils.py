import os
import json
from typing import Dict, List, Optional, Union

def create_deepspeed_config(
    stage: int = 0,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    bf16: bool = False,
    train_micro_batch_size: int = 1,
    config_name: str = None
) -> Dict:
    """
    Create a DeepSpeed configuration dictionary.
    
    Args:
        stage: ZeRO optimization stage (0, 1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_param: Whether to offload parameters to CPU (only for stage 3)
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use fp16 mixed precision
        bf16: Whether to use bf16 mixed precision
        train_micro_batch_size: Per-GPU batch size
        config_name: Optional name for the configuration
    
    Returns:
        Dictionary containing DeepSpeed configuration
    """
    config = {}
    
    # Add required batch size parameter
    config["train_micro_batch_size_per_gpu"] = train_micro_batch_size
    
    # Add ZeRO optimization config if stage > 0
    if stage > 0:
        # Set reasonable default values for bucket sizes based on model size
        # These values should work for most models but may need tuning for very large ones
        # For ZeRO-3, use smaller bucket sizes to avoid OOM during initialization
        if stage == 3:
            allgather_bucket_size = 5e7  # 50MB for ZeRO-3
            reduce_bucket_size = 5e7      # 50MB for ZeRO-3
        else:
            allgather_bucket_size = 2e8  # 200MB for ZeRO-1/2
            reduce_bucket_size = 2e8     # 200MB for ZeRO-1/2
        
        zero_config = {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": reduce_bucket_size,
            "allgather_bucket_size": allgather_bucket_size
        }
        
        # Add optimizer offloading for stage 2 and 3
        if offload_optimizer and stage >= 2:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }
            
        # Add parameter offloading for stage 3
        if offload_param and stage == 3:
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }
            
        # Add stage 3 specific configurations with lower memory footprint
        if stage == 3:
            zero_config.update({
                "sub_group_size": 1e8,               # Reduced from 1e9
                "stage3_prefetch_bucket_size": 5e7,   # Reduced to 50MB
                "stage3_param_persistence_threshold": 1e4,  # Lower threshold to offload more parameters
                "stage3_max_live_parameters": 5e8,    # Reduced from 1e9
                "stage3_max_reuse_distance": 5e8,     # Reduced from 1e9
                "stage3_gather_16bit_weights_on_model_save": True
            })
            
        config["zero_optimization"] = zero_config
    
    # Add mixed precision settings
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif bf16:
        config["bf16"] = {
            "enabled": True
        }
        
    # Add gradient accumulation
    if gradient_accumulation_steps > 1:
        config["gradient_accumulation_steps"] = gradient_accumulation_steps
    
    # For ZeRO stage 3, add a smaller activation checkpointing partition size
    if stage == 3:
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    
    return config

def save_deepspeed_config(config: Dict, output_dir: str, name: str = "ds_config.json") -> str:
    """
    Save DeepSpeed configuration to a JSON file.
    
    Args:
        config: DeepSpeed configuration dictionary
        output_dir: Directory to save the configuration
        name: Filename for the configuration
        
    Returns:
        Path to the saved configuration file
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, name)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    return config_path

def deepspeed_enabled(args) -> bool:
    """
    Check if DeepSpeed is enabled in the training arguments.
    
    Args:
        args: TrainingArguments instance
        
    Returns:
        True if DeepSpeed is enabled, False otherwise
    """
    return args.deepspeed is not None

def get_deepspeed_config(args) -> Optional[Dict]:
    """
    Get the DeepSpeed configuration from the training arguments.
    
    Args:
        args: TrainingArguments instance
        
    Returns:
        DeepSpeed configuration dictionary if enabled, None otherwise
    """
    if not deepspeed_enabled(args):
        return None
        
    # If a string path was provided, load the config
    if isinstance(args.deepspeed, str):
        try:
            with open(args.deepspeed, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading DeepSpeed config: {e}")
            return None
    
    # If a dictionary was provided, return it
    if isinstance(args.deepspeed, dict):
        return args.deepspeed
        
    return None