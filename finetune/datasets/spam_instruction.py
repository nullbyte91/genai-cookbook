import torch
from torch.utils.data import Dataset

class SpamInstructionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, pad_token_id=50256):
        import pandas as pd
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = []
        self.labels = []

        for _, row in self.data.iterrows():
            instruction = (
                "Classify the message as spam or not spam.\n\n"
                f"### Input:\n{row['Text']}\n\n### Response:\n"
            )
            encoded = tokenizer.encode(instruction)

            # Truncate and pad
            encoded = encoded[:max_length]
            encoded += [pad_token_id] * (max_length - len(encoded))

            self.encoded_texts.append(torch.tensor(encoded))
            self.labels.append(torch.tensor(row["Label"]))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.labels[index]

    def __len__(self):
        return len(self.data)
