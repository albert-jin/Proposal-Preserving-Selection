"""CPU check that proposal logits are base and candidate scores retain LoRA."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from tiny_fixture import Tokenizer
from utils import add_lora_to_student
from evaluate_s2t import EvaluationConfig,S2TDecoder
from split_adapter import SplitAdapterDecoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261004)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=96,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,tie_word_embeddings=True)).eval()
    original=copy.deepcopy(base)
    model=add_lora_to_student(base,16,32,0.0).eval()
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'lora_B' in name:value.normal_(0,0.04)
        model.get_output_embeddings().weight.add_(torch.randn_like(model.get_output_embeddings().weight)*0.01)
    def bundle():return SimpleNamespace(slm=model,slm_tokenizer=Tokenizer(),slm_formatter=SimpleNamespace(build_prompt=lambda q,d:q),
        s2t_token_ids=tuple(range(80,96)),llm=None,llm_tokenizer=None,llm_formatter=None,same_vocab=False)
    config=EvaluationConfig(max_new_tokens=8,candidate_k=16,entropy_threshold=0.0,max_context_len=64)
    native=S2TDecoder(config,bundle());split=SplitAdapterDecoder(config,bundle())
    ids=torch.tensor([[2,3,4]])
    with torch.no_grad():
        base_logits=original(ids,use_cache=False).logits
        proposal_logits=split.slm(ids,use_cache=False).logits
        assert torch.equal(base_logits,proposal_logits),'Proposal path is not the original base model'
        active=model(ids,use_cache=False).logits
        assert not torch.equal(active,proposal_logits),'Fixture must exercise nonzero adapter changes'
        candidates=torch.tensor([10,11,12,13])
        ns,_=native.score_s2t_local_candidates([2,3,4],candidates)
        ss,_=split.score_s2t_local_candidates([2,3,4],candidates)
        assert torch.equal(ns,ss),'Candidate scoring changed'
        assert not split.slm.scoring
        assert torch.equal(split.slm(ids,use_cache=False).logits,base_logits),'Scoring leaked adapter state into next proposal'
    result=dict(status='tiny_cpu_adapter_separation_passed',base_proposal_max_error=float((proposal_logits-base_logits).abs().max()),
        native_selector_max_error=float((ns-ss).abs().max()),adapter_state_restored=True,
        note='Tiny FP32 structural test with nonzero LoRA/head perturbations; not actual checkpoint accuracy.')
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
