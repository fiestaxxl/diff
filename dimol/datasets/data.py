
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, max_len, num_samples=100):
        self.max_len = max_len
        self.tokens = torch.randint(0, 1000, (num_samples, max_len)).long()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        results = {'token_ids': self.tokens[idx]}
        return results

class SmilesDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str):
        assert split in {'train', 'val', 'test'}, f"Split {split} must be in {('train', 'val', 'test')}"
        data_dir = Path(data_dir)
        self.tokens = np.load(data_dir / f"{split}_tokens.npy")        # (N, T) uint16
        self.attn_mask = np.load(data_dir / f"{split}_attn_mask.npy")  # (N, T) uint8

        assert self.tokens.shape == self.attn_mask.shape
        self.num_samples, self.max_len = self.tokens.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "token_ids": torch.from_numpy(self.tokens[idx].astype(np.int64)),
            "attention_mask": torch.from_numpy(self.attn_mask[idx].astype(bool)),
        }

# class SmilesDataset(Dataset):
#     def __init__(self, smiles: pd.Series, tokenizer, max_len, bert_tokenizer=None):
#         self.smiles = smiles.values  # numpy array (much faster)
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.bert_tokenizer = bert_tokenizer

#     def __len__(self):
#         return len(self.smiles)

#     def __getitem__(self, idx):
#         inputs = self.tokenizer(
#             self.smiles[idx], 
#             return_tensors="pt", 
#             padding="max_length",
#             truncation=True, 
#             max_length=self.max_len
#         )
#         # return inputs['input_ids'][0], inputs['attention_mask'][0].bool()

#         smiles_ids = inputs["input_ids"][0]
#         smiles_mask = inputs["attention_mask"][0].bool()
#         smiles_ids_bert = None
#         smiles_mask_bert = None

#         if self.bert_tokenizer:
#             bert_inputs = self.bert_tokenizer(
#                 self.smiles[idx], 
#                 return_tensors="pt", 
#                 padding="max_length",
#                 truncation=True, 
#                 max_length=self.max_len
#             )
#             smiles_ids_bert = inputs["input_ids"][0]
#             smiles_mask_bert = inputs["attention_mask"][0].bool()
#         return {
#             "smiles_ids": smiles_ids,
#             "smiles_mask": smiles_mask,
#             "smiles_ids_bert": smiles_ids_bert,
#             "smiles_mask_bert": smiles_mask_bert
#         }





# class SmilesDatasetWithCondition(Dataset):
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         smiles_tokenizer,
#         smiles_max_len: int,
#         text_emb_path: str,
#         text_mask_path: str,
#         text_shape: tuple
#     ):
#         """
#         text_shape = (N, L_text, 768)
#         """

#         self.smiles = df["SMILES"].values
#         self.sample_ids = df["sample_id"].values

#         self.smiles_tokenizer = smiles_tokenizer
#         self.smiles_max_len = smiles_max_len

#         # memmap-backed text embeddings
#         self.text_emb = np.memmap(
#             text_emb_path,
#             dtype="float16",
#             mode="r",
#             shape=text_shape
#         )

#         self.text_mask = np.memmap(
#             text_mask_path,
#             dtype="uint8",
#             mode="r",
#             shape=text_shape[:2]
#         )

#     def __len__(self):
#         return len(self.smiles)

#     def __getitem__(self, idx):
#         # --- SMILES ---
#         smiles_inputs = self.smiles_tokenizer(
#             self.smiles[idx],
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.smiles_max_len
#         )

#         smiles_ids = smiles_inputs["input_ids"][0]
#         smiles_mask = smiles_inputs["attention_mask"][0].bool()

#         # --- TEXT (by sample_id) ---
#         sid = self.sample_ids[idx]

#         text_emb = torch.from_numpy(self.text_emb[sid]).cpu() # [L_text, 768]
#         text_mask = torch.from_numpy(self.text_mask[sid]).cpu().bool()
        
#         return {
#             "smiles_ids": smiles_ids,
#             "smiles_mask": smiles_mask,
#             "text_emb": text_emb,
#             "text_mask": text_mask
#         }


# class TextDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len):
#         self.texts = df["description"].tolist()
#         self.ids = df["sample_id"].tolist()
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.texts[idx],
#             max_length=self.max_len,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#             "sample_id": self.ids[idx]
#         }


# def sample_valid_index_per_row(valid_mask):
#     B, L = valid_mask.shape
#     device = valid_mask.device

#     scores = torch.rand(B, L, device=device)
#     scores = scores.masked_fill(~valid_mask, -1e9)
#     return scores.argmax(dim=1)

# def corrupt_bond_order(tokens, valid_mask, bond_ids):
#     B, L = tokens.shape
#     device = tokens.device

#     idx = sample_valid_index_per_row(valid_mask)
#     idx = torch.clamp(idx, max=L-2)

#     b1 = bond_ids[torch.randint(len(bond_ids), (B,), device=device)]
#     b2 = bond_ids[torch.randint(len(bond_ids), (B,), device=device)]

#     out = tokens.clone()
#     out[torch.arange(B), idx] = b1
#     out[torch.arange(B), idx + 1] = b2
#     return out

# def corrupt_valency(tokens, valid_mask, atom_ids):
#     B, L = tokens.shape
#     device = tokens.device

#     idx = sample_valid_index_per_row(valid_mask)
#     idx = torch.clamp(idx, max=L-2)

#     atom = atom_ids[torch.randint(len(atom_ids), (B,), device=device)]

#     out = tokens.clone()
#     out[torch.arange(B), idx] = atom
#     out[torch.arange(B), idx + 1] = atom
#     return out

# def corrupt_paren_flip(tokens, valid_mask, open_id, close_id):
#     out = tokens.clone()
#     mask = (tokens == close_id) & valid_mask
#     out[mask] = open_id
#     return out

# def corrupt_paren_insert(tokens, pad_mask, open_id):
#     idx = sample_valid_index_per_row(pad_mask)
#     out = tokens.clone()
#     out[torch.arange(tokens.size(0)), idx] = open_id
#     return out

# def corrupt_ring(tokens, valid_mask, ring_ids):
#     B, L = tokens.shape
#     device = tokens.device

#     idx = sample_valid_index_per_row(valid_mask)
#     bad_ring = ring_ids[torch.randint(len(ring_ids), (B,), device=device)]

#     out = tokens.clone()
#     out[torch.arange(B), idx] = bad_ring
#     out[torch.arange(B), 0] = bad_ring   # unmatched ring
#     return out

# def corrupt_transition(tokens, pad_mask, vocab_ids):
#     B, L = tokens.shape
#     device = tokens.device

#     # valid positions except last
#     valid = pad_mask.clone()
#     valid[:, -1] = False

#     scores = torch.rand(B, L, device=device)
#     scores = scores.masked_fill(~valid, -1e9)
#     idx = scores.argmax(dim=1)

#     bad = vocab_ids[torch.randint(len(vocab_ids), (B,), device=device)]

#     out = tokens.clone()
#     out[torch.arange(B), idx + 1] = bad
#     return out

# def corrupt_truncate(tokens, pad_mask, pad_id):
#     idx = sample_valid_indices(pad_mask)
#     out = tokens.clone()
#     out[torch.arange(tokens.size(0)), idx] = pad_id
#     return out

# def corrupt_unbalanced(tokens, pad_mask, open_id, close_id):
#     """
#     tokens:   (B, L)
#     pad_mask: (B, L)  True = padding
#     """
#     B, L = tokens.shape
#     device = tokens.device

#     out = tokens.clone()

#     # valid positions = not padding
#     valid = ~pad_mask                                # (B, L)
#     has_valid = valid.any(dim=1)                     # (B,)

#     # index of last valid token per row
#     idx = valid.long().argmax(dim=1)                 # first True
#     idx = (valid.sum(dim=1) - 1).clamp(min=0)        # last True index

#     batch_idx = torch.arange(B, device=device)

#     out[batch_idx[has_valid], idx[has_valid]] = close_id
#     return out

# def make_invalid(tokens, valid_mask, vocab, mix_prob=0.3):
#     device = tokens.device
#     B = tokens.size(0)

#     open_id  = vocab["("]
#     close_id = vocab[")"]

#     ring_ids = torch.tensor([vocab[str(i)] for i in range(1, 10)], device=device)
#     bond_ids = torch.tensor([vocab[b] for b in ["=", "#"] if b in vocab], device=device)
#     atom_ids = torch.tensor([vocab[a] for a in ["C", "N", "O", "S"] if a in vocab], device=device)

#     OPS = [
#         # Syntax (dominant)
#         (0.35, lambda t, m: corrupt_unbalanced(t, m, open_id, close_id)),
#         (0.35, lambda t, m: corrupt_paren_insert(t, m, open_id)),

#         # Ring
#         (0.16, lambda t, m: corrupt_ring(t, m, ring_ids)),

#         # Bond
#         (0.06, lambda t, m: corrupt_bond_order(t, m, bond_ids)),
#     ]

#     probs = torch.tensor([p for p, _ in OPS], device=device)
#     probs /= probs.sum()

#     out = tokens.clone()

#     for i in range(B):
#         # always at least one strong corruption
#         idx = torch.multinomial(probs, 1).item()
#         out[i:i+1] = OPS[idx][1](out[i:i+1], valid_mask[i:i+1])

#         # often mix
#         if torch.rand(1).item() < mix_prob:
#             idx = torch.multinomial(probs, 1).item()
#             out[i:i+1] = OPS[idx][1](out[i:i+1], valid_mask[i:i+1])

#     return out