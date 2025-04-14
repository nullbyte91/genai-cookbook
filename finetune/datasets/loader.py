from torch.utils.data import DataLoader
from functools import partial
from utils.collator import custom_collate_fn
from datasets.spam_instruction import SpamInstructionDataset

def get_data_loaders(train_data, val_data, test_data, tokenizer, device, batch_size=8):
    from datasets.instruction import InstructionDataset

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    num_workers = 0

    train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=batch_size,
                              collate_fn=customized_collate_fn, shuffle=True, drop_last=True,
                              num_workers=num_workers)

    val_loader = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=batch_size,
                            collate_fn=customized_collate_fn, shuffle=False, drop_last=False,
                            num_workers=num_workers)

    test_loader = DataLoader(InstructionDataset(test_data, tokenizer), batch_size=batch_size,
                             collate_fn=customized_collate_fn, shuffle=False, drop_last=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_spam_instruction_loaders(train_path, val_path, test_path, tokenizer, device, batch_size=8):
    train_ds = SpamInstructionDataset(train_path, tokenizer)
    val_ds = SpamInstructionDataset(val_path, tokenizer, max_length=train_ds.encoded_texts[0].size(0))
    test_ds = SpamInstructionDataset(test_path, tokenizer, max_length=train_ds.encoded_texts[0].size(0))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )
    