import os
import time
import json
import torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Callable
from torch.cuda.amp import GradScaler
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from deepspeed_utils import create_deepspeed_config, save_deepspeed_config

class MemoryTracker:
    """
    Tracks GPU memory usage during training
    """
    def __init__(self, device_id=0, log_interval=10):
        """
        Initialize NVML for memory tracking
        
        Args:
            device_id: ID of the GPU to track
            log_interval: How often to log memory (in seconds)
        """
        self.device_id = device_id
        self.log_interval = log_interval
        self.memory_logs = []
        self.start_time = None
        self.last_log_time = None
        
        # Initialize NVML
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(device_id)
            self.tracking_enabled = True
        except Exception as e:
            print(f"Warning: NVML initialization failed: {e}")
            print("Memory tracking will be disabled")
            self.tracking_enabled = False
    
    def start(self):
        """Start memory tracking"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.memory_logs = []
        
        # Log initial memory
        self.log_memory()
    
    def log_memory(self, step=None, additional_info=None):
        """
        Log current memory usage
        
        Args:
            step: Training step (if applicable)
            additional_info: Any additional info to log
        """
        if not self.tracking_enabled:
            return
        
        current_time = time.time()
        
        # If it's not time to log yet, skip
        if current_time - self.last_log_time < self.log_interval and step is None:
            return
            
        try:
            # Get memory info
            info = nvmlDeviceGetMemoryInfo(self.handle)
            
            # CPU memory via psutil
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            # Log the info
            log_entry = {
                "timestamp": current_time - self.start_time,
                "gpu_used_mb": info.used / 1024 / 1024,
                "gpu_free_mb": info.free / 1024 / 1024,
                "gpu_total_mb": info.total / 1024 / 1024,
                "gpu_utilization_pct": (info.used / info.total) * 100,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "step": step
            }
            
            # Add any additional info
            if additional_info:
                log_entry.update(additional_info)
                
            self.memory_logs.append(log_entry)
            self.last_log_time = current_time
            
        except Exception as e:
            print(f"Error logging memory: {e}")
    
    def get_peak_memory(self):
        """Get peak memory usage in MB"""
        if not self.memory_logs:
            return 0
        return max(log["gpu_used_mb"] for log in self.memory_logs)
    
    def get_average_memory(self):
        """Get average memory usage in MB"""
        if not self.memory_logs:
            return 0
        return sum(log["gpu_used_mb"] for log in self.memory_logs) / len(self.memory_logs)


@dataclass
class PerformanceMetrics:
    """Class to store performance metrics during training"""
    total_training_time: float = 0
    epochs_completed: int = 0
    steps_completed: int = 0
    peak_gpu_memory_mb: float = 0
    avg_gpu_memory_mb: float = 0
    loss_history: List[float] = field(default_factory=list)
    eval_metrics: Dict[str, List[float]] = field(default_factory=dict)
    memory_logs: List[Dict] = field(default_factory=list)
    training_config: Dict = field(default_factory=dict)
    
    def to_dict(self):
        """Convert metrics to dictionary"""
        return asdict(self)
    
    def save_to_file(self, filepath):
        """Save metrics to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def plot_memory_usage(self, output_file=None):
        """
        Plot memory usage over time
        
        Args:
            output_file: Path to save the plot. If None, the plot is displayed.
        """
        if not self.memory_logs:
            print("No memory logs to plot")
            return
            
        timestamps = [log["timestamp"] for log in self.memory_logs if "timestamp" in log]
        memory_used = [log["gpu_used_mb"] for log in self.memory_logs if "gpu_used_mb" in log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, memory_used)
        plt.title("GPU Memory Usage During Training")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Used (MB)")
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def plot_metrics(self, output_file=None):
        """
        Plot training metrics
        
        Args:
            output_file: Path to save the plot. If None, the plot is displayed.
        """
        if not self.loss_history and not self.eval_metrics:
            print("No metrics to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        if self.loss_history:
            plt.subplot(2, 1, 1)
            plt.plot(self.loss_history)
            plt.title("Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
        
        # Plot evaluation metrics
        if self.eval_metrics:
            plt.subplot(2, 1, 2)
            for metric_name, values in self.eval_metrics.items():
                plt.plot(values, label=metric_name)
            plt.title("Evaluation Metrics")
            plt.xlabel("Evaluation Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()


class PerformanceTrackingCallback(TrainerCallback):
    """
    Callback for tracking performance metrics during training
    """
    def __init__(self, log_dir="./performance_logs", memory_tracker=None):
        """
        Initialize the callback
        
        Args:
            log_dir: Directory to save logs
            memory_tracker: MemoryTracker instance
        """
        self.log_dir = log_dir
        self.metrics = PerformanceMetrics()
        
        # Create memory tracker if not provided
        if memory_tracker is None:
            self.memory_tracker = MemoryTracker()
        else:
            self.memory_tracker = memory_tracker
            
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        self.training_start_time = None
        self.epoch_start_time = None
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called when the trainer initialization ends"""
        from deepspeed_utils import deepspeed_enabled, get_deepspeed_config
        
        self.memory_tracker.start()
        
        # Check if model has gradient checkpointing enabled
        # We need to look at the model's internal state, not just if it has the method
        model = kwargs.get("model", None)
        gradient_checkpointing_enabled = False
        
        if model is not None:
            # Different models store this information in different ways
            # Check common patterns
            if hasattr(model, "is_gradient_checkpointing"):
                gradient_checkpointing_enabled = model.is_gradient_checkpointing
            elif hasattr(model, "_gradient_checkpointing"):
                gradient_checkpointing_enabled = model._gradient_checkpointing
            elif hasattr(model, "gradient_checkpointing"):
                gradient_checkpointing_enabled = model.gradient_checkpointing
            # If we can't find a direct attribute, we'll rely on the config value
        
        # Get DeepSpeed configuration if enabled
        ds_config = None
        ds_stage = 0
        if deepspeed_enabled(args):
            ds_config = get_deepspeed_config(args)
            if ds_config and "zero_optimization" in ds_config:
                ds_stage = ds_config["zero_optimization"].get("stage", 0)
                
        # Save training configuration
        training_config = {
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": gradient_checkpointing_enabled,
            "max_grad_norm": args.max_grad_norm,
            "optimizer": args.optim,  # Add the optimizer to the tracked config
            "deepspeed_enabled": deepspeed_enabled(args),
            "deepspeed_stage": ds_stage,
        }
        
        # Add DeepSpeed-specific configurations if enabled
        if ds_config:
            # Check for optimizer offloading
            if "zero_optimization" in ds_config:
                zero_config = ds_config["zero_optimization"]
                # Check if optimizer offloading is enabled
                if "offload_optimizer" in zero_config:
                    training_config["deepspeed_offload_optimizer"] = True
                    training_config["deepspeed_offload_optimizer_device"] = zero_config["offload_optimizer"].get("device", "cpu")
                else:
                    training_config["deepspeed_offload_optimizer"] = False
                
                # Check if parameter offloading is enabled (ZeRO-3 only)
                if "offload_param" in zero_config:
                    training_config["deepspeed_offload_param"] = True
                    training_config["deepspeed_offload_param_device"] = zero_config["offload_param"].get("device", "cpu")
                else:
                    training_config["deepspeed_offload_param"] = False
        
        # Check if a model is passed and add info if available
        if model is not None:
            config = getattr(model, "config", None)
            if config is not None:
                model_config = {
                    "model_type": getattr(config, "model_type", "unknown"),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                }
                training_config.update(model_config)
        
        self.metrics.training_config = training_config
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.training_start_time = time.time()
        self.memory_tracker.log_memory(step=0, additional_info={"event": "train_begin"})
        
        # Log the start of training
        print("\n" + "="*50)
        print(f"Starting training with config:")
        for k, v in self.metrics.training_config.items():
            print(f"  {k}: {v}")
        print("="*50 + "\n")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch"""
        self.epoch_start_time = time.time()
        self.memory_tracker.log_memory(
            step=state.global_step, 
            additional_info={"event": "epoch_begin", "epoch": state.epoch}
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step"""
        # Log memory usage periodically
        self.memory_tracker.log_memory(
            step=state.global_step,
            additional_info={"event": "step_end", "epoch": state.epoch}
        )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch"""
        epoch_time = time.time() - self.epoch_start_time
        
        self.metrics.epochs_completed += 1
        self.memory_tracker.log_memory(
            step=state.global_step, 
            additional_info={
                "event": "epoch_end", 
                "epoch": state.epoch,
                "epoch_time": epoch_time
            }
        )
        
        # Print epoch stats
        print(f"\nEpoch {state.epoch} completed in {epoch_time:.2f} seconds")
        print(f"Current peak GPU memory: {self.memory_tracker.get_peak_memory():.2f} MB\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics"""
        if logs is None:
            return
            
        # Extract and store loss
        if "loss" in logs:
            self.metrics.loss_history.append(logs["loss"])
        
        # Extract and store eval metrics
        for key, value in logs.items():
            if key.startswith("eval_"):
                metric_name = key[5:]  # Remove 'eval_' prefix
                if metric_name not in self.metrics.eval_metrics:
                    self.metrics.eval_metrics[metric_name] = []
                self.metrics.eval_metrics[metric_name].append(value)
                
        # Track step progress
        self.metrics.steps_completed = state.global_step
        
        # Log memory with current metrics
        self.memory_tracker.log_memory(
            step=state.global_step, 
            additional_info={
                "event": "log", 
                "epoch": state.epoch,
                "logs": logs
            }
        )
        
        # Save intermediate metrics
        self.metrics.memory_logs = self.memory_tracker.memory_logs
        self.metrics.peak_gpu_memory_mb = self.memory_tracker.get_peak_memory()
        self.metrics.avg_gpu_memory_mb = self.memory_tracker.get_average_memory()
        
        # Save progress so far
        metrics_path = os.path.join(self.log_dir, f"metrics_step_{state.global_step}.json")
        self.metrics.save_to_file(metrics_path)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        # Calculate total training time
        self.metrics.total_training_time = time.time() - self.training_start_time
        
        # Log final memory usage
        self.memory_tracker.log_memory(
            step=state.global_step, 
            additional_info={"event": "train_end"}
        )
        
        # Update final metrics
        self.metrics.memory_logs = self.memory_tracker.memory_logs
        self.metrics.peak_gpu_memory_mb = self.memory_tracker.get_peak_memory()
        self.metrics.avg_gpu_memory_mb = self.memory_tracker.get_average_memory()
        
        # Save final metrics
        final_metrics_path = os.path.join(self.log_dir, "final_metrics.json")
        self.metrics.save_to_file(final_metrics_path)
        
        # Generate plots
        self.metrics.plot_memory_usage(os.path.join(self.log_dir, "memory_usage.png"))
        self.metrics.plot_metrics(os.path.join(self.log_dir, "training_metrics.png"))
        
        # Print summary
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Total training time: {self.metrics.total_training_time:.2f} seconds")
        print(f"Epochs completed: {self.metrics.epochs_completed}")
        print(f"Steps completed: {self.metrics.steps_completed}")
        print(f"Peak GPU memory: {self.metrics.peak_gpu_memory_mb:.2f} MB")
        print(f"Average GPU memory: {self.metrics.avg_gpu_memory_mb:.2f} MB")
        
        # Print DeepSpeed info if enabled
        if self.metrics.training_config.get("deepspeed_enabled", False):
            print(f"DeepSpeed ZeRO Stage: {self.metrics.training_config.get('deepspeed_stage', 0)}")
            if self.metrics.training_config.get("deepspeed_offload_optimizer", False):
                print(f"DeepSpeed Optimizer Offload: Enabled (Device: {self.metrics.training_config.get('deepspeed_offload_optimizer_device', 'cpu')})")
            if self.metrics.training_config.get("deepspeed_offload_param", False):
                print(f"DeepSpeed Parameter Offload: Enabled (Device: {self.metrics.training_config.get('deepspeed_offload_param_device', 'cpu')})")
                
        print("="*50 + "\n")
        print(f"Performance logs saved to: {self.log_dir}")


def measure_peak_memory_usage(func):
    """
    Decorator to measure peak memory usage of a function
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that measures memory usage
    """
    def wrapper(*args, **kwargs):
        # Initialize memory tracking
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Record starting memory
        start_memory = torch.cuda.memory_allocated()
        
        # Run the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory / (1024 * 1024)
        
        # Calculate memory usage and execution time
        memory_used = peak_memory - start_memory
        memory_used_mb = memory_used / (1024 * 1024)
        execution_time = end_time - start_time
        
        print(f"Function: {func.__name__}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        print(f"Memory increase: {memory_used_mb:.2f} MB")
        
        return result
    return wrapper


def setup_trainer_with_memory_tracking(
    model,
    args,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    log_dir="./performance_logs",
    additional_callbacks=None,
    **trainer_kwargs
):
    """
    Set up a Trainer with memory tracking enabled
    
    Args:
        model: The model to train
        args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator function
        log_dir: Directory to save logs
        additional_callbacks: Additional TrainerCallbacks to include
        **trainer_kwargs: Additional arguments to pass to the Trainer
        
    Returns:
        A Trainer instance with memory tracking enabled
    """
    # Create memory tracker
    memory_tracker = MemoryTracker()
    
    # Create performance tracking callback
    performance_callback = PerformanceTrackingCallback(log_dir=log_dir, memory_tracker=memory_tracker)
    
    # Combine callbacks
    callbacks = [performance_callback]
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        **trainer_kwargs
    )
    
    return trainer, performance_callback

def benchmark_models(
    models_to_benchmark,
    dataset,
    eval_dataset=None,
    training_args=None,
    optimizer_configs=None,
    output_dir="./benchmarks",
    **trainer_kwargs
):
    """
    Benchmark multiple models with different configurations
    
    Args:
        models_to_benchmark: List of (model_name, model) tuples
        dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        training_args: Base TrainingArguments to use
        optimizer_configs: List of optimizer configurations to test
        output_dir: Directory to save benchmark results
        **trainer_kwargs: Additional arguments to pass to the Trainer
        
    Returns:
        Dictionary of benchmark results
    """
    import os
    import json
    import torch
    from transformers import TrainingArguments, Trainer
    from profiler import setup_trainer_with_memory_tracking
    
    os.makedirs(output_dir, exist_ok=True)
    
    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            logging_steps=10,
            save_steps=1000,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
        )
    
    if optimizer_configs is None:
        # Default configurations to benchmark
        optimizer_configs = [
            {
                "name": "baseline", 
                "fp16": False, 
                "bf16": False, 
                "gradient_checkpointing": False,
            },
            {
                "name": "fp16", 
                "fp16": True, 
                "bf16": False, 
                "gradient_checkpointing": False,
            },
            {
                "name": "gradient_checkpointing", 
                "fp16": False, 
                "bf16": False, 
                "gradient_checkpointing": True,
            },
            {
                "name": "fp16_gradient_checkpointing", 
                "fp16": True, 
                "bf16": False, 
                "gradient_checkpointing": True,
            },
        ]
    
    benchmark_results = {}
    
    for model_name, model in models_to_benchmark:
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        model_results = {}
        
        for config in optimizer_configs:
            config_name = config["name"]
            config_output_dir = os.path.join(model_output_dir, config_name)
            os.makedirs(config_output_dir, exist_ok=True)
            
            print(f"\n\nBenchmarking {model_name} with configuration: {config_name}")
            print("=" * 80)
            
            # Create a copy of the model for this configuration
            model_copy = type(model)(model.config)
            
            # Apply gradient checkpointing if requested
            if config.get("gradient_checkpointing", False):
                if hasattr(model_copy, "gradient_checkpointing_enable"):
                    model_copy.gradient_checkpointing_enable()
                    print("Enabled gradient checkpointing")
                else:
                    print("Warning: Model does not support gradient checkpointing")
            
            grad_accum_steps = config.get("gradient_accumulation_steps", 1)
            optimizer = config.get("optim", "adamw_torch")
            
            # Create custom training arguments for this configuration
            config_args_dict = {
                "output_dir": config_output_dir,
                "optim": optimizer,
                "gradient_accumulation_steps": grad_accum_steps,
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "logging_steps": training_args.logging_steps,
                "save_steps": training_args.save_steps,
                "eval_strategy": training_args.eval_strategy,
                "eval_steps": training_args.eval_steps,
                "fp16": config.get("fp16", False),
                "bf16": config.get("bf16", False),
                "report_to": "none",  # Disable wandb, tensorboard, etc.
            }
            
            # Handle DeepSpeed configuration
            if "deepspeed_stage" in config:
                ds_stage = config.get("deepspeed_stage", 0)
                if ds_stage > 0:
                    # Create DeepSpeed config with the batch size included
                    ds_config = create_deepspeed_config(
                        stage=ds_stage,
                        offload_optimizer=config.get("offload_optimizer", False),
                        offload_param=config.get("offload_param", False),
                        gradient_accumulation_steps=grad_accum_steps,
                        fp16=config.get("fp16", False),
                        bf16=config.get("bf16", False),
                        train_micro_batch_size=training_args.per_device_train_batch_size
                    )
                    
                    # Save DeepSpeed config to a file
                    ds_config_path = save_deepspeed_config(
                        ds_config, 
                        config_output_dir, 
                        f"ds_config_{config_name}.json"
                    )
                    
                    # Add DeepSpeed config to training arguments
                    config_args_dict["deepspeed"] = ds_config_path
                    print(f"Enabled DeepSpeed with configuration: {ds_config_path}")
            
            # Create TrainingArguments with the configured settings
            config_args = TrainingArguments(**config_args_dict)
            
            # Setup trainer with memory tracking
            trainer, performance_callback = setup_trainer_with_memory_tracking(
                model=model_copy,
                args=config_args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                log_dir=config_output_dir,
                **trainer_kwargs
            )
            
            # Train the model
            try:
                trainer.train()
                
                # Get metrics
                metrics = performance_callback.metrics.to_dict()
                model_results[config_name] = metrics
                
                # Save metrics
                results_file = os.path.join(config_output_dir, "benchmark_results.json")
                with open(results_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                    
                print(f"Benchmark results saved to {results_file}")
                
            except Exception as e:
                print(f"Error benchmarking {model_name} with config {config_name}: {e}")
                model_results[config_name] = {"error": str(e)}
            
            # Clear cache
            torch.cuda.empty_cache()
        
        benchmark_results[model_name] = model_results
        
        # Save summary of this model's results
        model_summary_file = os.path.join(model_output_dir, "benchmark_summary.json")
        with open(model_summary_file, 'w') as f:
            json.dump(model_results, f, indent=2)
    
    # Save overall benchmark results
    benchmark_summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(benchmark_summary_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
        
    print(f"\nAll benchmark results saved to {output_dir}")
    return benchmark_results

# def benchmark_models(
#     models_to_benchmark,
#     dataset,
#     eval_dataset=None,
#     training_args=None,
#     optimizer_configs=None,
#     output_dir="./benchmarks",
#     **trainer_kwargs
# ):
#     """
#     Benchmark multiple models with different configurations
    
#     Args:
#         models_to_benchmark: List of (model_name, model) tuples
#         dataset: Training dataset
#         eval_dataset: Evaluation dataset (optional)
#         training_args: Base TrainingArguments to use
#         optimizer_configs: List of optimizer configurations to test
#         output_dir: Directory to save benchmark results
#         **trainer_kwargs: Additional arguments to pass to the Trainer
        
#     Returns:
#         Dictionary of benchmark results
#     """
#     from deepspeed_utils import create_deepspeed_config, save_deepspeed_config
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     if training_args is None:
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=1,
#             per_device_train_batch_size=8,
#             logging_steps=10,
#             save_steps=1000,
#             eval_strategy="steps" if eval_dataset else "no",
#             eval_steps=100 if eval_dataset else None,
#         )
    
#     if optimizer_configs is None:
#         # Default configurations to benchmark
#         optimizer_configs = [
#             {
#                 "name": "baseline", 
#                 "fp16": False, 
#                 "bf16": False, 
#                 "gradient_checkpointing": False,
#             },
#             {
#                 "name": "fp16", 
#                 "fp16": True, 
#                 "bf16": False, 
#                 "gradient_checkpointing": False,
#             },
#             {
#                 "name": "gradient_checkpointing", 
#                 "fp16": False, 
#                 "bf16": False, 
#                 "gradient_checkpointing": True,
#             },
#             {
#                 "name": "fp16_gradient_checkpointing", 
#                 "fp16": True, 
#                 "bf16": False, 
#                 "gradient_checkpointing": True,
#             },
#         ]
    
#     benchmark_results = {}
    
#     for model_name, model in models_to_benchmark:
#         model_output_dir = os.path.join(output_dir, model_name)
#         os.makedirs(model_output_dir, exist_ok=True)
        
#         model_results = {}
        
#         for config in optimizer_configs:
#             config_name = config["name"]
#             config_output_dir = os.path.join(model_output_dir, config_name)
#             os.makedirs(config_output_dir, exist_ok=True)
            
#             print(f"\n\nBenchmarking {model_name} with configuration: {config_name}")
#             print("=" * 80)
            
#             # Create a copy of the model for this configuration
#             model_copy = type(model)(model.config)
            
#             # Apply gradient checkpointing if requested
#             if config.get("gradient_checkpointing", False):
#                 if hasattr(model_copy, "gradient_checkpointing_enable"):
#                     model_copy.gradient_checkpointing_enable()
#                     print("Enabled gradient checkpointing")
#                 else:
#                     print("Warning: Model does not support gradient checkpointing")
            
#             grad_accum_steps = config.get("gradient_accumulation_steps", 1)
#             optimizer = config.get("optim", "adamw_torch")
            
#             # Create custom training arguments for this configuration
#             config_args_dict = {
#                 "output_dir": config_output_dir,
#                 "optim": optimizer,
#                 "gradient_accumulation_steps": grad_accum_steps,
#                 "num_train_epochs": training_args.num_train_epochs,
#                 "per_device_train_batch_size": training_args.per_device_train_batch_size,
#                 "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
#                 "logging_steps": training_args.logging_steps,
#                 "save_steps": training_args.save_steps,
#                 "eval_strategy": training_args.eval_strategy,
#                 "eval_steps": training_args.eval_steps,
#                 "fp16": config.get("fp16", False),
#                 "bf16": config.get("bf16", False),
#                 "report_to": "none",  # Disable wandb, tensorboard, etc.
#             }
            
#             # Handle DeepSpeed configuration
#             if "deepspeed_stage" in config:
#                 ds_stage = config.get("deepspeed_stage", 0)
#                 if ds_stage > 0:
#                     # Create DeepSpeed config
#                     ds_config = create_deepspeed_config(
#                         stage=ds_stage,
#                         offload_optimizer=config.get("offload_optimizer", False),
#                         offload_param=config.get("offload_param", False),
#                         gradient_accumulation_steps=grad_accum_steps,
#                         fp16=config.get("fp16", False),
#                         bf16=config.get("bf16", False),
#                         train_micro_batch_size=training_args.per_device_train_batch_size  # Pass the batch size
#                     )
                    
#                     # Save DeepSpeed config to a file
#                     ds_config_path = save_deepspeed_config(
#                         ds_config, 
#                         config_output_dir, 
#                         f"ds_config_{config_name}.json"
#                     )
                    
#                     # Add DeepSpeed config to training arguments
#                     config_args_dict["deepspeed"] = ds_config_path
#                     print(f"Enabled DeepSpeed with configuration: {ds_config_path}")
            
#             # Create TrainingArguments with the configured settings
#             config_args = TrainingArguments(**config_args_dict)
            
#             # Setup trainer with memory tracking
#             trainer, performance_callback = setup_trainer_with_memory_tracking(
#                 model=model_copy,
#                 args=config_args,
#                 train_dataset=dataset,
#                 eval_dataset=eval_dataset,
#                 log_dir=config_output_dir,
#                 **trainer_kwargs
#             )
            
#             # Train the model
#             try:
#                 trainer.train()
                
#                 # Get metrics
#                 metrics = performance_callback.metrics.to_dict()
#                 model_results[config_name] = metrics
                
#                 # Save metrics
#                 results_file = os.path.join(config_output_dir, "benchmark_results.json")
#                 with open(results_file, 'w') as f:
#                     json.dump(metrics, f, indent=2)
                    
#                 print(f"Benchmark results saved to {results_file}")
                
#             except Exception as e:
#                 print(f"Error benchmarking {model_name} with config {config_name}: {e}")
#                 model_results[config_name] = {"error": str(e)}
            
#             # Clear cache
#             torch.cuda.empty_cache()
        
#         benchmark_results[model_name] = model_results
        
#         # Save summary of this model's results
#         model_summary_file = os.path.join(model_output_dir, "benchmark_summary.json")
#         with open(model_summary_file, 'w') as f:
#             json.dump(model_results, f, indent=2)
    
#     # Save overall benchmark results
#     benchmark_summary_file = os.path.join(output_dir, "benchmark_summary.json")
#     with open(benchmark_summary_file, 'w') as f:
#         json.dump(benchmark_results, f, indent=2)
        
#     print(f"\nAll benchmark results saved to {output_dir}")
#     return benchmark_results


def compare_benchmarks(benchmark_results, output_dir="./benchmarks"):
    """
    Compare and visualize benchmark results
    
    Args:
        benchmark_results: Dictionary of benchmark results
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for comparison
    model_names = []
    config_names = []
    training_times = []
    peak_memories = []
    
    for model_name, model_results in benchmark_results.items():
        for config_name, metrics in model_results.items():
            if "error" in metrics:
                continue
                
            model_names.append(model_name)
            config_names.append(config_name)
            training_times.append(metrics.get("total_training_time", 0))
            peak_memories.append(metrics.get("peak_gpu_memory_mb", 0))
    
    # Create comparison dataframe
    comparison_data = {
        "model": model_names,
        "config": config_names,
        "training_time": training_times,
        "peak_memory_mb": peak_memories,
    }
    
    # Plot training time comparison
    plt.figure(figsize=(12, 6))
    positions = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(positions, training_times, width)
    plt.xticks(positions, [f"{m} ({c})" for m, c in zip(model_names, config_names)], rotation=45, ha="right")
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"))
    
    # Plot memory usage comparison
    plt.figure(figsize=(12, 6))
    plt.bar(positions, peak_memories, width)
    plt.xticks(positions, [f"{m} ({c})" for m, c in zip(model_names, config_names)], rotation=45, ha="right")
    plt.title("Peak Memory Usage Comparison")
    plt.ylabel("Memory (MB)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
    
    # Create a markdown report
    report = f"""# Benchmark Comparison Report

## Summary
- Number of models benchmarked: {len(set(model_names))}
- Number of configurations tested: {len(set(config_names))}

## Training Time Comparison
![Training Time Comparison](training_time_comparison.png)

## Memory Usage Comparison
![Memory Usage Comparison](memory_usage_comparison.png)

## Detailed Results

| Model | Configuration | Training Time (s) | Peak Memory (MB) |
|-------|--------------|-------------------|-----------------|
"""
    
    for i in range(len(model_names)):
        report += f"| {model_names[i]} | {config_names[i]} | {training_times[i]:.2f} | {peak_memories[i]:.2f} |\n"
    
    with open(os.path.join(output_dir, "benchmark_report.md"), "w") as f:
        f.write(report)
    
    print(f"Comparison report saved to {os.path.join(output_dir, 'benchmark_report.md')}")
    
    return comparison_data