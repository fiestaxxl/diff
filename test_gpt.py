import torch
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
from dimol.models.gpt import SmilesAR
from dimol.tokenizer.smiles_tokenizer import SmilesTokenizer

device = "cuda"
ar = SmilesAR.from_pretrained("checkpoints_ar/ar_v2/1000", map_location=device).eval()
tok = SmilesTokenizer.load("data/smiles_bpe.json")

N = 1000
bos = torch.full((N, 1), tok.bos_id, dtype=torch.long, device=device)
out = ar.generate(bos, max_new_tokens=200, temperature=1.0, eos_id=tok.eos_id)

smiles = tok.decode_batch(out, special_decode=True)

print("\n".join(smiles[:25]))
valid = sum(1 for s in smiles if s and Chem.MolFromSmiles(s) is not None)
print(f"AR validity: {valid}/{N} = {valid/N:.2%}")