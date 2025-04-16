# Fine-Tuning GPT-2 for code generation
This example demonstrates how to fine-tune GPT-2 on token generation task using the CodeAlpaca_20K dataset from Hugging Face.

---
##  Dataset Preparation and Tokenization
I use the CodeAlpaca_20K dataset from Hugging Face. Each data entry has a structure similar to:
```bash
{
  "prompt": "Explain how a for loop works in Python",
  "completion": "Here is a basic Python code snippet showing a for loop..."
}
```
```python
from datasets import load_dataset

alpaca_data = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
```
---
### Prompt Formatting
For instruction/data tasks like CodeAlpaca, we typically want the prompt to include instructions and the response to contain the model’s completion. The function format_prompt was defined as:

```python
def format_prompt(entry):
    instr = f"### Instruction:\n{entry['prompt']}"
    output = f"\n\n### Response:\n{entry['completion']}"
    return instr + output
```
---
### Tokenization
We used the GPT-2 tokenizer:
```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
```
Because GPT-2 is decoder-only and was never trained with a <pad> token, many implementations simply reuse eos_token (<|endoftext|>) as the padding token.

During the tokenization in your custom Dataset class, you perform:

```python
self.tokenizer(
    prompt,
    truncation=True,
    padding="max_length",
    max_length=self.max_length,
    return_tensors="pt"
)
```
which yields:
* input_ids
* attention_mask
* (optionally) labels
The attention_mask ensures the model does not pay attention to padding tokens.

---

### Dataset Class and DataLoader
```python
class CodeAlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        ...
    def __getitem__(self, idx):
        prompt = format_prompt(self.data[idx])
        tokens = self.tokenizer( ... )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": tokens["input_ids"].squeeze(0)
        }

```
Then you wrap this in a DataLoader, specifying a custom collate_fn to batch up items:
```python
train_dataset = CodeAlpacaDataset(alpaca_data.select(range(1000)), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
```

## Model Architecture and Layer Freezing
### Loading GPT-2
The GPT-2 model is loaded from local that we have created, 
```python
model, tokenizer = load_gpt_model(
    model_size="355M",
    weights_dir="gpt2_weights",
    freeze_layers=True,
    num_unfrozen_layers=4
)
```
Under the Hood: load_gpt_model
#### Retrieve the Config
A config dictionary (e.g., {"emb_dim": 1024, "n_layers": 24, ...}) is loaded based on the specified model size (124M, 355M, 774M, etc.).

#### Load Raw Weights
If no checkpoint_path is provided, the function downloads the raw GPT-2 weights (or loads them from your local path) and manually populates a custom GPTModel instance.

#### Layer Freezing
The function freeze_bottom_layers sets requires_grad to False for all but the top num_unfrozen_layers Transformer blocks. This way, only the last few blocks, final layer norm, and (optionally) an output head remain trainable.

This drastically reduces memory usage and training time, while still allowing some adaptation.

```python
freeze_layers=True,
num_unfrozen_layers=4
```
Fine-tuning the entire GPT-2 (especially larger versions) is expensive in terms of GPU memory and can lead to catastrophic forgetting. Freezing lower layers (which have already learned strong language representations) is a common strategy to:
* Reduce training cost
* Preserve language capacity
* Focus training on the highest-level “reasoning/semantic” layers

### GPT-2 Forward Logic for Language Modeling
GPT-2 is a decoder-only model. Each forward pass typically returns logits for each token in the sequence. If our input has shape [batch_size, seq_len], then the output logits typically have shape:
```bash
[batch_size, seq_len, vocab_size]
```

```python
logits = model(input_ids)
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), 
    labels.view(-1), 
    ignore_index=pad_token_id
)
```
* We flatten logits and labels so cross-entropy compares the predicted distribution for each token with the ground truth.
* We ignore positions labeled with pad_token_id so that padded tokens do not contribute to loss.
---

## Training Loop
```python
def train_code_model(model, train_loader, device, num_epochs=3, eval_freq=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1),
                                   ignore_index=pad_token_id)
            loss.backward()
            optimizer.step()

            if step % eval_freq == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
```
1. Batch Processing
input_ids, attention_mask, and labels are loaded onto the GPU.

2. Forward Pass
logits = model(input_ids) yields [B, L, V] where B = batch_size, L = seq_len, and V = vocab_size (50257 for GPT-2).

3. Loss Calculation
Flattening shapes to [B*L, V] vs. [B*L] for cross-entropy.
ignore_index=pad_token_id ensures padded tokens are excluded from the loss.
4. Backward + Optim Step
Standard PyTorch procedure: loss.backward() → optimizer.step().

## Inference
### Hyperparameter
#### Temperature Scaling:
$$
\text{AdjustedLogit}_i = \frac{\text{Logit}_i}{\tau}
$$
τ=0.7: sharper distribution, more likely to pick top probable tokens.

τ=1.2: flatter distribution, more random explorations.

#### Top-k Truncation
After we apply temperature scaling, we sort tokens by their log probabilities, keep only the top k, set everything else to 
−
∞

```python
top_k_vals, _ = torch.topk(logits, top_k)
min_vals = top_k_vals[..., -1, None]
logits = torch.where(logits < min_vals, float('-inf'), logits)
```

#### Top-p (Nucleus) Truncation
Alternatively, or additionally, we can do nucleus sampling:
1. Sort tokens by descending log probability.
2. Compute the cumulative sum of their probabilities.
3. Retain tokens until we exceed p in total.
Set the rest to 
−
∞
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
probs = torch.softmax(sorted_logits, dim=-1)
cumulative_probs = torch.cumsum(probs, dim=-1)

sorted_mask = cumulative_probs > top_p
indices_to_remove = sorted_indices[sorted_mask]
logits[0, indices_to_remove] = float('-inf')
```
