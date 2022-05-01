import torch
from torch.utils.data import TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import copy


class MyDataset(Dataset):
    def __init__(self, all_input_ids, input_attention_mask, child_indices, head_indices, mask_entail, mask_cont, num_dependencies, arcs, sentence_label, sent_indices, sent_basis_label, hypo_cls_idx, padding_idx):
        self.all_input_ids = all_input_ids
        self.input_attention_mask = input_attention_mask
        self.child_indices = child_indices
        self.head_indices = head_indices
        self.mask_entail = mask_entail 
        self.mask_cont = mask_cont
        self.num_dependencies = num_dependencies
        self.arcs = arcs
        self.sentence_label = sentence_label
        self.sent_indices = sent_indices
        self.sent_basis_label = sent_basis_label
        self.hypo_cls_idx = hypo_cls_idx
        self.padding_idx = padding_idx
        
    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.input_attention_mask[idx], self.child_indices[idx], self.head_indices[idx], self.mask_entail[idx], self.mask_cont[idx], self.num_dependencies[idx], self.arcs[idx], self.sentence_label[idx], copy.deepcopy(self.sent_indices[idx]), self.sent_basis_label[idx], self.hypo_cls_idx[idx], self.padding_idx

def collate_fn(batch):
    all_input_ids, input_attention_mask, child_indices, head_indices, mask_entail, mask_cont, num_dependencies, arcs, sentence_label, sent_indices, sent_basis_label, hypo_cls_idx, padding_idx = list(zip(*batch))
    
    batch_size = len(padding_idx)
    padding_idx = padding_idx[0]
    

    child_indices = [torch.tensor(indices, dtype=torch.long) for b_indices in child_indices for indices in b_indices]
    head_indices = [torch.tensor(indices, dtype=torch.long) for b_indices in head_indices for indices in b_indices]
    indices = child_indices + head_indices
    indices = pad_sequence(indices, batch_first=True, padding_value=padding_idx).view(2, batch_size, 20, -1)
    
    child_indices = indices[0]
    head_indices = indices[1]
    
    max_num_sent = 0
    max_sent_len = 0
    for i in range(len(sent_indices)):
        if len(sent_indices[i]) > max_num_sent:
            max_num_sent = len(sent_indices[i])
        for j in range(len(sent_indices[i])):
            if len(sent_indices[i][j]) > max_sent_len:
                max_sent_len = len(sent_indices[i][j])
                
    padding_vec = [padding_idx] * max_sent_len
    for i in range(len(sent_indices)):
        if (max_num_sent - len(sent_indices[i])) != 0:
            sent_indices[i].extend([padding_vec] * (max_num_sent - len(sent_indices[i])))
    
    sent_indices = [torch.tensor(indices, dtype=torch.long) for b_indices in sent_indices for indices in b_indices]
    sent_indices = pad_sequence(sent_indices, batch_first=True, padding_value=padding_idx).view(batch_size, max_num_sent, -1)
    
    sent_basis_label = pad_sequence(sent_basis_label, batch_first=True, padding_value=-1)
    
    return torch.tensor(all_input_ids, dtype=torch.long), torch.tensor(input_attention_mask, dtype=torch.long), child_indices, head_indices, torch.tensor(mask_entail, dtype=torch.long), torch.tensor(mask_cont, dtype=torch.long), torch.tensor(num_dependencies, dtype=torch.long), torch.tensor(arcs, dtype=torch.long), torch.tensor(sentence_label, dtype=torch.long), sent_indices, sent_basis_label, torch.tensor(hypo_cls_idx, dtype=torch.long)

