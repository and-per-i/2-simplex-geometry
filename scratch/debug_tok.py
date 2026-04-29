import sys
import os
sys.path.append(".")
from tokenizer.hf_tokenizer import load_tokenizer
import torch

TOKENIZER_PATH = "tokenizer/weights/geometry.757.model"
prompt = "a b c = triangle a b c ; m = midpoint a b ? cong c m m a"

try:
    tok = load_tokenizer(TOKENIZER_PATH, vocab_size=1024)
    print(f"Tokenizer loaded. Pad token: {tok.pad_token}, Pad ID: {tok.pad_token_id}")
    
    # Test encode
    print("Testing encode(add_special_tokens=False)")
    ids = tok.encode(prompt, add_special_tokens=False)
    print(f"Encode result: {ids}")
    
    # Test __call__ without any extras
    print("Testing __call__ with return_tensors=None")
    res = tok(prompt, return_tensors=None, add_special_tokens=False)
    print(f"Tokenization result (dict): {res}")
    
    # Test with return_tensors="pt"
    print("Testing __call__ with return_tensors='pt'")
    res_pt = tok(prompt, return_tensors="pt")
    print(f"Tokenization result (PT): {res_pt}")

except Exception as e:
    import traceback
    traceback.print_exc()
