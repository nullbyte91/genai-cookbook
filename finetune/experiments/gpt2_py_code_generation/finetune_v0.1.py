# Alpaca-style Instruction Fine-tuning on GPT-2 Medium (355M)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
import os
from gpt2_local_loader import load_gpt_model, get_device

# ------------------ Tokenizer ------------------ #
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

# ------------------ Load and Filter Dataset ------------------ #
alpaca_data = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")

python_keywords = ['def ', 'import ', 'lambda ']
def is_python_code(text):
    return any(keyword in text for keyword in python_keywords)

python_dataset = alpaca_data.filter(lambda example: is_python_code(example['completion']))

# ------------------ Prompt Formatting ------------------ #
def format_input(entry):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['prompt']}"
    )

def format_output(entry):
    return f"\n\n### Response:\n{entry['completion']}"

# ------------------ Dataset Class ------------------ #
class AlpacaInstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.encoded_texts = [
            tokenizer.encode(format_input(entry) + format_output(entry))
            for entry in data
        ]

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# ------------------ Collate Function ------------------ #
def collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=1024, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy() + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)

# ------------------ Split Data ------------------ #
train_size = int(len(python_dataset) * 0.85)
test_size = int(len(python_dataset) * 0.1)

train_data = python_dataset.select(range(train_size))
test_data = python_dataset.select(range(train_size, train_size + test_size))
val_data = python_dataset.select(range(train_size + test_size, len(python_dataset)))

# ------------------ Model Loading ------------------ #
model, tokenizer = load_gpt_model(
    model_size="355M",
    weights_dir="gpt2_weights",
    freeze_layers=True,
    num_unfrozen_layers=4
)
device = get_device()

# ------------------ DataLoaders ------------------ #
train_dataset = AlpacaInstructionDataset(train_data, tokenizer)
val_dataset = AlpacaInstructionDataset(val_data, tokenizer)
test_dataset = AlpacaInstructionDataset(test_data, tokenizer)

customized_collate_fn = partial(collate_fn, device=device, allowed_max_length=512)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=customized_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=customized_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=customized_collate_fn)

# ------------------ Loss Calculation ------------------ #
@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=5):
    model.eval()
    total_loss = 0
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        logits = model(inputs.to(device))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        total_loss += loss.item()
    return total_loss / (i + 1)

# ------------------ Response Generator ------------------ #
def generate_response(model, tokenizer, prompt, device, max_new_tokens=128):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# ------------------ Training Loop ------------------ #
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=3, eval_freq=100, eval_iter=5, tokenizer=None, start_context=None):
    model.to(device)
    train_losses, val_losses, tokens_seen = [], [], []
    global_step = token_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for step, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(inputs.to(device))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to(device).view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()

            if global_step % eval_freq == 0:
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {loss.item():.3f}, Val loss {val_loss:.3f}")
                if tokenizer and val_loader:
                    sample_idx = torch.randint(0, len(val_loader.dataset), (1,)).item()
                    example = val_loader.dataset[sample_idx]
                    sample_prompt = tokenizer.decode(example[:1024])  # Truncate if needed

                    response = generate_response(
                        model, tokenizer, sample_prompt, device
                    )
                    print("\nSample Generation:\n", response)
                train_losses.append(loss.item())
                val_losses.append(val_loss)
                tokens_seen.append(token_counter)

            global_step += 1
            token_counter += inputs.numel()

    return train_losses, val_losses, tokens_seen

# ------------------ Launch Training ------------------ #
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
start_prompt = format_input(val_data[0])

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=3, eval_freq=100, eval_iter=5,
    tokenizer=tokenizer, start_context=start_prompt
)

# ------------------ Save Model ------------------ #
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": model.config,
}
torch.save(checkpoint, "gpt2_355m_alpaca_instruction_ft.pth")
print("\nModel checkpoint saved to 'gpt2_355m_alpaca_instruction_ft.pth'")