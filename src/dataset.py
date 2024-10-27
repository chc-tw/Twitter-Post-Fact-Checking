import pandas as pd
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, mode, tokenizer):
        self.mode = mode
        self.df = pd.read_json(mode + ".json", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)

        inputs = self.tokenizer.encode_plus(
            text_a,
            text_b,
            truncation="longest_first",
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        
        if self.mode == 'test':
            return (torch.tensor(inputs['input_ids']), 
                   torch.tensor(inputs['token_type_ids']),
                   torch.tensor(inputs['attention_mask']))
        else:
            return (torch.tensor(inputs['input_ids']), 
                   torch.tensor(inputs['token_type_ids']),
                   torch.tensor(inputs['attention_mask']), 
                   label_tensor)

    def __len__(self):
        return self.len
