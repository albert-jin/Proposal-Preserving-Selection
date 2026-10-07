"""Tiny tokenizer fixture from the server structural checks."""
import torch
from transformers import BatchEncoding


class Tokenizer:
    eos_token_id=None
    unk_token_id=None
    def __call__(self,text,return_tensors=None):
        return BatchEncoding(dict(input_ids=torch.tensor([[2,3,4]]),attention_mask=torch.ones(1,3,dtype=torch.long)))
    def decode(self,ids,**kwargs):return ' '.join(str(i) for i in ids)
    def convert_tokens_to_ids(self,token):return None
    def encode(self,*args,**kwargs):return []
