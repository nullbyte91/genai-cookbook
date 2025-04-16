import torch
from transformers import GPT2Tokenizer
from gpt2_local_loader import load_gpt_model  # your local module
import torch
from typing import Optional
from transformers import GPT2Tokenizer
from torch import nn

# ----------------------------
# Load model + tokenizer
# ----------------------------
checkpoint_path = "gpt2_355m_alpaca_instruction_ft.pth"  # ← your model checkpoint
model, tokenizer = load_gpt_model(
    model_size="355M",
    weights_dir="gpt2_weights",
    checkpoint_path=checkpoint_path,  # <--- custom fine-tuned weights
    freeze_layers=False,
    num_unfrozen_layers=None,
)

tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# Token helper functions
# ----------------------------
def text_to_token_ids(text):
    return tokenizer.encode(text, return_tensors="pt").to(device)

def token_ids_to_text(token_ids):
    return tokenizer.decode(token_ids[0], skip_special_tokens=True)

# ----------------------------
# Generation function (Temp + Top-k)
# ----------------------------
def gpt_generate(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate text using a GPT model with temperature, top-k, and top-p sampling.

    Args:
        model: The GPT model (should return raw logits).
        tokenizer: Tokenizer instance (e.g., GPT2Tokenizer).
        prompt: Prompt text to begin generation.
        max_tokens: Number of tokens to generate.
        temperature: Softmax temperature (higher = more diverse).
        top_k: Top-k sampling threshold (0 disables).
        top_p: Top-p (nucleus) sampling threshold (1.0 disables).
        device: CUDA or CPU device (optional).

    Returns:
        Generated text string.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :]  # get last token's logits

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                min_vals = top_k_vals[..., -1, None]
                logits = torch.where(logits < min_vals, float('-inf'), logits)

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()  # ✅ FIXED HERE
                sorted_mask[..., 0] = 0  # Always keep at least one token

                indices_to_remove = sorted_indices[sorted_mask]
                logits[0, indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
# ----------------------------
# Inference Entry Point
# ----------------------------
if __name__ == "__main__":
    prompt = (
        "Write a Python function to find all the prime numbers below a given number.\n\n"
    )

    input_ids = text_to_token_ids(prompt)

    torch.manual_seed(42)
    output = gpt_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=100,
        temperature=1.2,   # <- this is the new addition
        top_k=50,
        top_p=0.9
    )

    print("Generated text:\n", output)
