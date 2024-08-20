import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .labelled import PosInst

class InverseDataset(Dataset):
    def __init__(self, data: list[tuple[list[int], list[tuple[int, ...]], PosInst, list[int]]], seq_len):
        super().__init__()
        # seq_len is the length of the sequence (used for padding)
        self.seq_len = seq_len

        self.data = data
        self.pad_token = torch.tensor([0], dtype=torch.int64)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        term_data, pos_list, pos_inst, target_data = self.data[idx]
        
        term_data = torch.tensor(term_data, dtype=torch.int64)

        # pad the term data
        enc_num_padding_tokens = self.seq_len - len(term_data)

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        input_encoding = torch.cat(
            [
                term_data, 
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0,
        )

        # get the mask for padding
        input_mask = (input_encoding != self.pad_token).int().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len)

        label = torch.tensor(target_data, dtype=torch.int64)

        # pad the label
        label = torch.cat(
            [
                label, 
                torch.tensor([0] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0,
        )
        

        return {
            "input" : input_encoding,
            "input_mask" : input_mask,
            "pos_inst" : pos_inst,
            "pos_list" : pos_list,
            "label" : label
        }
