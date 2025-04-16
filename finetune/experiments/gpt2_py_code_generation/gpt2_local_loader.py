# import os
# import sys
# import torch
# from typing import Dict, Tuple, Any, List, Optional

# # Add project root to path for imports
# project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from utils.gpt2_utils import download_and_load_gpt2, load_weights_into_gpt
# from models.gpt2.gpt_model_v1 import GPTModel
# from transformers import GPT2Tokenizer

# def get_gpt_config(model_size: str) -> Dict[str, Any]:
#     """
#     Get configuration for different GPT model sizes.
    
#     Args:
#         model_size: Size of the GPT model ("124M", "355M", "774M", "1.5B")
    
#     Returns:
#         Dictionary with model configuration
#     """
#     configs = {
#         "124M": {
#             "vocab_size": 50257,  # Standard GPT-2 vocabulary size
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 768,
#             "n_layers": 12,
#             "n_heads": 12,
#         },
#         "355M": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 1024,
#             "n_layers": 24,
#             "n_heads": 16,
#         },
#         "774M": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 1280,
#             "n_layers": 36,
#             "n_heads": 20,
#         },
#         "1.5B": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 1600,
#             "n_layers": 48,
#             "n_heads": 25,
#         },
#     }
    
#     if model_size not in configs:
#         raise ValueError(f"Model size {model_size} not supported. Choose from: {list(configs.keys())}")
    
#     return configs[model_size]

# def load_gpt_model(
#     model_size: str = "124M", 
#     weights_dir: str = "gpt2_weights",
#     device: Optional[torch.device] = None,
#     freeze_layers: bool = False,
#     num_classes: Optional[int] = None
# ) -> Tuple[GPTModel, GPT2Tokenizer]:
#     """
#     Load a GPT-2 model of specified size.
    
#     Args:
#         model_size: Size of the model to load ("124M", "355M", "774M", "1.5B")
#         weights_dir: Directory to store/load the weights
#         device: Device to load the model on (defaults to CUDA if available, else CPU)
#         freeze_layers: Whether to freeze most of the model's layers (useful for fine-tuning)
#         num_classes: If provided, attaches a classification head with this many classes
        
#     Returns:
#         Tuple of (model, tokenizer)
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Get model configuration for the specified size
#     config = get_gpt_config(model_size)
    
#     # Initialize tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # Update config with tokenizer vocab size (should be 50257)
#     config["vocab_size"] = len(tokenizer)
    
#     # Download and load weights
#     settings, params = download_and_load_gpt2(model_size, weights_dir)
    
#     # Initialize model
#     model = GPTModel(config)
    
#     # Load weights
#     load_weights_into_gpt(model, params)
    
#     # Attach classification head if needed
#     if num_classes is not None:
#         model.out_head = torch.nn.Linear(config["emb_dim"], num_classes)
    
#     # Freeze layers if requested
#     if freeze_layers:
#         # Freeze all layers except final transformer block and any classification head
#         for p in model.parameters():
#             p.requires_grad = False
            
#         # Unfreeze final transformer block
#         for p in model.trf_blocks[-1].parameters():
#             p.requires_grad = True
            
#         # Unfreeze final normalization layer
#         for p in model.final_norm.parameters():
#             p.requires_grad = True
            
#         # Unfreeze classification head if it exists
#         if hasattr(model, 'out_head'):
#             for p in model.out_head.parameters():
#                 p.requires_grad = True
    
#     # Move model to device
#     model.to(device)
    
#     return model, tokenizer

# def gpt_generate(
#     model: GPTModel,
#     tokenizer: GPT2Tokenizer,
#     prompt: str,
#     max_tokens: int = 100,
#     temperature: float = 0.7,
#     top_k: int = 40,
#     top_p: float = 0.9,
#     device: Optional[torch.device] = None
# ) -> str:
#     """
#     Generate text using a GPT model.
    
#     Args:
#         model: The GPT model
#         tokenizer: The tokenizer
#         prompt: The prompt to continue from
#         max_tokens: Maximum number of tokens to generate
#         temperature: Sampling temperature (higher = more random)
#         top_k: Number of highest probability tokens to consider
#         top_p: Cumulative probability threshold for token consideration
#         device: Device to run generation on
        
#     Returns:
#         Generated text string
#     """
#     if device is None:
#         device = next(model.parameters()).device
    
#     model.eval()
    
#     # Tokenize prompt
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
#     # Generate tokens
#     with torch.no_grad():
#         for _ in range(max_tokens):
#             # Forward pass
#             outputs = model(input_ids)
            
#             # Get logits for the next token
#             next_token_logits = outputs[:, -1, :]
            
#             # Apply temperature
#             next_token_logits = next_token_logits / temperature
            
#             # Apply top-k filtering
#             if top_k > 0:
#                 indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
#                 next_token_logits[indices_to_remove] = float('-inf')
            
#             # Apply top-p (nucleus) filtering
#             if top_p < 1.0:
#                 sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
#                 # Remove tokens with cumulative probability above the threshold
#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 # Shift the indices to the right to keep also the first token above the threshold
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                 sorted_indices_to_remove[..., 0] = 0
                
#                 indices_to_remove = sorted_indices[sorted_indices_to_remove]
#                 next_token_logits[0, indices_to_remove] = float('-inf')
            
#             # Sample from the filtered distribution
#             probs = torch.softmax(next_token_logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
            
#             # Append to input_ids
#             input_ids = torch.cat((input_ids, next_token), dim=1)
            
#             # If EOS token is generated, stop
#             if next_token.item() == tokenizer.eos_token_id:
#                 break
    
#     # Decode the generated text
#     generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
#     return generated_text
# import os
# import sys
# import torch
# from typing import Dict, Tuple, Any, Optional
# from transformers import GPT2Tokenizer

# # Project path setup
# project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from utils.gpt2_utils import download_and_load_gpt2, load_weights_into_gpt
# from models.gpt2.gpt_model_v1 import GPTModel

# # -------------------------------------------------------------
# # GPT-2 Configuration Presets
# # -------------------------------------------------------------
# def get_gpt_config(model_size: str) -> Dict[str, Any]:
#     configs = {
#         "124M": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 768,
#             "n_layers": 12,
#             "n_heads": 12,
#         },
#         "355M": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 1024,
#             "n_layers": 24,
#             "n_heads": 16,
#         },
#         "774M": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 1280,
#             "n_layers": 36,
#             "n_heads": 20,
#         },
#         "1.5B": {
#             "vocab_size": 50257,
#             "context_length": 1024,
#             "drop_rate": 0.1,
#             "qkv_bias": True,
#             "emb_dim": 1600,
#             "n_layers": 48,
#             "n_heads": 25,
#         },
#     }

#     if model_size not in configs:
#         raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(configs.keys())}")

#     return configs[model_size]

# # -------------------------------------------------------------
# # Layer Freezing Utility
# # -------------------------------------------------------------
# def freeze_bottom_layers(model: GPTModel, num_unfrozen: int):
#     total_layers = len(model.trf_blocks)
#     freeze_upto = total_layers - num_unfrozen

#     for i, block in enumerate(model.trf_blocks):
#         for param in block.parameters():
#             param.requires_grad = i >= freeze_upto  # Unfreeze top-k blocks only

#     # Always unfreeze final norm
#     for p in model.final_norm.parameters():
#         p.requires_grad = True

#     # Optional: unfreeze head
#     if hasattr(model, "out_head"):
#         for p in model.out_head.parameters():
#             p.requires_grad = True

# # -------------------------------------------------------------
# # Main Loader
# # -------------------------------------------------------------
# def load_gpt_model(
#     model_size: str = "124M",
#     weights_dir: str = "gpt2_weights",
#     device: Optional[torch.device] = None,
#     freeze_layers: bool = False,
#     num_unfrozen_layers: int = 1,
#     num_classes: Optional[int] = None
# ) -> Tuple[GPTModel, GPT2Tokenizer]:
#     """
#     Loads a GPT model of the specified size with optional layer freezing.

#     Args:
#         model_size (str): e.g., "355M"
#         weights_dir (str): local weights path
#         device (torch.device): target device (default: cuda/cpu auto)
#         freeze_layers (bool): freeze all but top-k transformer blocks
#         num_unfrozen_layers (int): number of final transformer blocks to keep trainable
#         num_classes (int): if classification head is needed

#     Returns:
#         model (GPTModel), tokenizer (GPT2Tokenizer)
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     config = get_gpt_config(model_size)

#     # Tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
#     config["vocab_size"] = len(tokenizer)

#     # Load weights
#     settings, params = download_and_load_gpt2(model_size, weights_dir)
#     model = GPTModel(config)
#     model.config = config
#     load_weights_into_gpt(model, params)

#     # Optional: Classification head
#     if num_classes is not None:
#         model.out_head = torch.nn.Linear(config["emb_dim"], num_classes)

#     # Optional: Freeze layers
#     if freeze_layers:
#         freeze_bottom_layers(model, num_unfrozen=num_unfrozen_layers)

#     # Move to device
#     model.to(device)
#     return model, tokenizer

# # -------------------------------------------------------------
# # Device Getter
# # -------------------------------------------------------------
# def get_device() -> torch.device:
#     """
#     Returns the default device: CUDA if available, else CPU.
#     """
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import sys
import torch
from typing import Dict, Tuple, Any, Optional
from transformers import GPT2Tokenizer

# Project path setup
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.gpt2_utils import download_and_load_gpt2, load_weights_into_gpt
from models.gpt2.gpt_model_v1 import GPTModel

# -------------------------------------------------------------
# GPT-2 Configuration Presets
# -------------------------------------------------------------
def get_gpt_config(model_size: str) -> Dict[str, Any]:
    configs = {
        "124M": {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.1,
            "qkv_bias": True,
            "emb_dim": 768,
            "n_layers": 12,
            "n_heads": 12,
        },
        "355M": {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.1,
            "qkv_bias": True,
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16,
        },
        "774M": {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.1,
            "qkv_bias": True,
            "emb_dim": 1280,
            "n_layers": 36,
            "n_heads": 20,
        },
        "1.5B": {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.1,
            "qkv_bias": True,
            "emb_dim": 1600,
            "n_layers": 48,
            "n_heads": 25,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(configs.keys())}")

    return configs[model_size]

# -------------------------------------------------------------
# Layer Freezing Utility
# -------------------------------------------------------------
def freeze_bottom_layers(model: GPTModel, num_unfrozen: int):
    total_layers = len(model.trf_blocks)
    freeze_upto = total_layers - num_unfrozen

    for i, block in enumerate(model.trf_blocks):
        for param in block.parameters():
            param.requires_grad = i >= freeze_upto  # Unfreeze top-k blocks only

    # Always unfreeze final norm
    for p in model.final_norm.parameters():
        p.requires_grad = True

    # Optional: unfreeze head
    if hasattr(model, "out_head"):
        for p in model.out_head.parameters():
            p.requires_grad = True

# -------------------------------------------------------------
# Main Loader
# -------------------------------------------------------------
def load_gpt_model(
    model_size: str = "124M",
    weights_dir: Optional[str] = "gpt2_weights",
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    freeze_layers: bool = False,
    num_unfrozen_layers: int = 1,
    num_classes: Optional[int] = None
) -> Tuple[GPTModel, GPT2Tokenizer]:
    """
    Loads a GPT model of the specified size with optional layer freezing.
    You can provide either `weights_dir` for loading raw weights or `checkpoint_path` to load a full .pth file.
    """
    if device is None:
        device = get_device()

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("#################")
    print(checkpoint_path)
    if checkpoint_path:
        print("##############")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        model = GPTModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.config = config
    else:
        config = get_gpt_config(model_size)
        config["vocab_size"] = len(tokenizer)
        settings, params = download_and_load_gpt2(model_size, weights_dir)
        model = GPTModel(config)
        model.config = config
        load_weights_into_gpt(model, params)

    # Optional: Classification head
    if num_classes is not None:
        model.out_head = torch.nn.Linear(config["emb_dim"], num_classes)

    if freeze_layers:
        freeze_bottom_layers(model, num_unfrozen=num_unfrozen_layers)

    model.to(device)
    return model, tokenizer

# -------------------------------------------------------------
# Device Getter
# -------------------------------------------------------------
def get_device() -> torch.device:
    """
    Returns the default device: CUDA if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gpt_generate(
    model: GPTModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate text using a GPT model.

    Args:
        model: GPT model instance
        tokenizer: GPT2Tokenizer
        prompt: Initial prompt string
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling cutoff
        top_p: Top-p (nucleus) sampling cutoff
        device: Optional device override

    Returns:
        Generated text string (prompt + generated continuation)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[..., -1, None]] = float('-inf')

            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
