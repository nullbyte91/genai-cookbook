from torch.utils.data import DataLoader
from functools import partial
from utils.collator import custom_collate_fn

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
