"""Tiny random Qwen/PEFT CPU integration, no pretrained model or benchmark use."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
os.environ['HF_HUB_OFFLINE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from peft import PeftModel
from train_lowmem import group_loss,S2TLocalTrainingConfig,listwise_loss,register_reserved_row_gradient_mask,collate_s2t_local_groups,add_lora_to_student,score_candidates_from_reserved_logits
from train_s2t_local import language_model_kl


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--tied-embeddings',action='store_true'); a=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(20261004)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=96,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,tie_word_embeddings=a.tied_embeddings))
    reference=copy.deepcopy(base).eval().requires_grad_(False)
    model=add_lora_to_student(copy.deepcopy(base),16,32,0.0)
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_B' in name:param.normal_(std=0.02)
    full=copy.deepcopy(model).train(); micro=copy.deepcopy(model).train()
    micro.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    micro.enable_input_require_grads()
    reserved=torch.arange(80,96)
    hooks=[register_reserved_row_gradient_mask(m,reserved) for m in [full,micro]]
    rows=[dict(context_ids=list(range(1,n+1)),cand_ids=list(range(20,36)),
        teacher_prob_abs=torch.softmax(torch.arange(16,dtype=torch.float32)*(1 if n==5 else -1),0).tolist()) for n in [5,9]]
    batch=collate_s2t_local_groups(rows,0)
    cfg=S2TLocalTrainingConfig()
    output=full(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
    pos=batch['attention_mask'].sum(-1)-1
    bins=output[torch.arange(len(pos)),pos][:,reserved]
    scores,_=score_candidates_from_reserved_logits(bins,cfg.label_mode,cfg.score_mode,cfg.bin_softmax_temperature)
    selection,_=listwise_loss(scores.view(2,16),batch['teacher_abs'],batch['candidate_mask'],cfg)
    with torch.no_grad():
        ref=reference(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
    original=selection+cfg.lm_kl_weight*language_model_kl(output,ref,batch['attention_mask'],pos)
    original.backward()
    accumulated=0.0
    for row in rows:
        b=collate_s2t_local_groups([row],0)
        loss,_=group_loss(micro,reference,b,reserved,cfg,chunk_tokens=7,
            selection_scale=0.5,prefix_scale=len(row['context_ids'])/14)
        accumulated+=float(loss.detach()); loss.backward()
    errors={}
    for (name,pf),(other,pm) in zip(full.named_parameters(),micro.named_parameters()):
        assert name==other
        if pf.requires_grad:
            assert pf.grad is not None and pm.grad is not None,name
            assert torch.allclose(pf.grad,pm.grad,rtol=2e-4,atol=2e-6),name
            errors[name]=float((pf.grad-pm.grad).abs().max())
    assert abs(float(original.detach())-accumulated)<2e-6
    assert all(p.grad is None for p in reference.parameters())
    assert micro.get_output_embeddings().weight.grad[:80].count_nonzero()==0
    optimizer=torch.optim.AdamW([p for p in micro.parameters() if p.requires_grad],lr=5e-5,weight_decay=0.01)
    optimizer.step(); micro.eval()
    with torch.no_grad():expected=micro(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
    with tempfile.TemporaryDirectory(prefix='tiny_qwen_fixture_') as td:
        micro.save_pretrained(td)
        restored=PeftModel.from_pretrained(copy.deepcopy(base),td).eval()
        with torch.no_grad():actual=restored(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
        assert torch.allclose(expected,actual,rtol=1e-5,atol=1e-6)
        restore_error=float((expected-actual).abs().max())
    result=dict(status='tiny_random_qwen_cpu_integration_passed',original_loss=float(original.detach()),
        accumulated_loss=accumulated,max_parameter_gradient_error=max(errors.values()),
        checked_trainable_tensors=len(errors),reserved_head_gradient_mask_passed=True,reference_gradients_absent=True,
        save_reload_max_logit_error=restore_error,backbone_activation_checkpointing=True,tie_word_embeddings=a.tied_embeddings,
        scope='Random two-layer32-hidden Qwen on CPU with LoRA/dropout0 and unequal5/9-token prefixes. Real1.5B BF16 GPU memory, throughput and training quality remain untested.')
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    for hook in hooks:hook.remove()


if __name__=='__main__':main()
