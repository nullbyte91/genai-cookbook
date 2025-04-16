# Fine-Tuning GPT-2 for AG News Classification
This example demonstrates how to fine-tune GPT-2 on a classification task using the AG News dataset. GPT-2 is a decoder-only language model, originally designed for text generation — but we adapt it here for multi-class classification.

---

## Task

- Input: News headlines (`text`)
- Output: One of four classes — `['World', 'Sports', 'Business', 'Sci/Tech']`

---

## Phase 1
## Key Implementation Highlights
### Model Architecture Change

GPT-2 typically outputs a distribution over vocabulary tokens for **next-token prediction**. For classification, we **replace the language modeling head** with a **linear classification head**:

```python
model.out_head = nn.Linear(hidden_dim, num_classes)
```
Because instead of predicting the next word, we now want to map the final hidden state to one of num_classes.

###  Tokenization and Padding
When working with datasets that contain text sequences of varying lengths, it's crucial to standardize them for efficient batch processing during training or inference. Deep learning models expect input tensors of uniform shape, so we need a consistent strategy for managing sequence lengths.
There are two common approaches:
1. Truncation
All sequences are truncated to a fixed length — typically the shortest or an average length in the dataset or batch.
Pros: Computationally efficient; reduced memory usage.
Cons: May lead to significant information loss, especially if important content exists at the end of long texts.
2. Padding
Shorter sequences are padded to match the length of the longest sequence in the batch or dataset.
Pros: Preserves full input content.
Cons: Increases computation (especially with large padding); padding tokens must be ignored during attention.

GPT-2 doesn’t define a pad_token by default. We considered two options:
Option 1: Reuse <|endoftext|> (eos_token) as pad_token
Option 2 (preferred): Add a distinct [PAD] token to the tokenizer and resize embeddings
```python
# Option 1
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Reuse eos as pad token
pad_token_id = tokenizer.pad_token_id      # Should be 50256

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
# here:
# Pads the sequence to max_length
# Uses the pad_token_id under the hood (which you've defined as eos_token_id)
# Also returns an attention_mask indicating which tokens are real vs. padding
```
As you see here it return the attention_mask, what is attention mask?
An attention mask is a tensor of the same shape as your input sequence:
1 → real token
0 → padding token
Example:
```bash 
Input:         [50256, 234, 789, 42, 0, 0]
Attention Mask: [1,     1,   1,  1, 0, 0]
```
This tells the model:
“Pay attention only to the first 4 tokens. Ignore the padding tokens.”
Why Is the Attention Mask Critical in Training?
Because:
GPT-2 (and most transformer models) perform self-attention over the entire input.
Without a mask, the model will treat pad tokens as real content, leading to:
* Learning noise
* Incorrect gradients
* Poor generalization

And this attention mask pass it to the model during training.

###  Loss Function – Cross Entropy
For classification, we use the standard cross-entropy loss between predicted logits and ground-truth labels.
```python
logits = model(input_ids)[:, -1, :]  # Use final token's hidden state
loss = F.cross_entropy(logits, labels)
```
Why last token?
GPT-2 is autoregressive. We extract the final hidden state (i.e., from the last token in the input) as a summary representation to classify.

### Freezing Strategy
To avoid training the entire 124M parameter model, we freeze most of GPT-2 and only train:
* The classification head
* The final transformer block
* The final normalization layer
```python
for p in model.parameters():
    p.requires_grad = False
for p in model.trf_blocks[-1].parameters():
    p.requires_grad = True
for p in model.final_norm.parameters():
    p.requires_grad = True
for p in model.out_head.parameters():
    p.requires_grad = True
```
---