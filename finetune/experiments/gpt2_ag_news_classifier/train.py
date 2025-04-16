import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch.nn.functional as F
import os, sys

# ------------------ Project Root for Imports ------------------ #
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# ------------------ Configuration ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 4
MAX_LENGTH = 128
BATCH_SIZE = 8

# ------------------ Load Dataset ------------------ #
raw_dataset = load_dataset("fancyzhx/ag_news")
train_data = raw_dataset["train"].shuffle(seed=42).select(range(1000))
test_size = int(0.2 * len(raw_dataset["test"]))
test_data = raw_dataset["test"].shuffle(seed=123).select(range(test_size))

# ------------------ Tokenizer ------------------ #
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Reuse eos as pad token
pad_token_id = tokenizer.pad_token_id      # Should be 50256

# ------------------ Dataset Wrapper ------------------ #
class AGNewsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ------------------ DataLoaders ------------------ #
def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch])
    }

train_dataset = AGNewsDataset(train_data, tokenizer, max_length=MAX_LENGTH)
test_dataset = AGNewsDataset(test_data, tokenizer, max_length=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ------------------ Model Setup ------------------ #
from utils.gpt2_utils import download_and_load_gpt2, load_weights_into_gpt
from models.gpt2.gpt_model_v1 import GPTModel

BASE_CONFIG = {
    "vocab_size": len(tokenizer),  # Should remain 50257
    "context_length": 1024,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}

settings, params = download_and_load_gpt2("124M", "gpt2_weights")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)  # No shape mismatch when not resizing embedding

# Attach classification head
model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], NUM_CLASSES)

# Freeze all except final transformer block and classification head
for p in model.parameters():
    p.requires_grad = False
for p in model.trf_blocks[-1].parameters():
    p.requires_grad = True
for p in model.final_norm.parameters():
    p.requires_grad = True
for p in model.out_head.parameters():
    p.requires_grad = True

model.to(device)

# ------------------ Training & Eval ------------------ #
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return F.cross_entropy(logits, target_batch)

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = sum(calc_loss_batch(x["input_ids"], x["labels"], model, device).item()
                         for i, x in enumerate(train_loader) if i < eval_iter) / eval_iter
        val_loss = sum(calc_loss_batch(x["input_ids"], x["labels"], model, device).item()
                       for i, x in enumerate(val_loader) if i < eval_iter) / eval_iter
    model.train()
    return train_loss, val_loss

def train_instruction_classifier(model, train_loader, val_loader, device, num_epochs=3, eval_freq=50, eval_iter=5):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.1)
    global_step = 0

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(batch["input_ids"], batch["labels"], model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                print(f"Epoch {epoch+1}, Step {global_step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    return model

trained_model = train_instruction_classifier(model, train_loader, val_loader, device)

# ------------------ Inference ------------------ #
def classify_review(text, model, tokenizer, device, max_length=128):
    model.eval()
    input_ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
    return LABELS[predicted_label]

# Example
print(classify_review("Apple unveils new AI-powered MacBooks.", trained_model, tokenizer, device))
