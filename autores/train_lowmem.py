"""Single-GPU adaptation of the pinned repository objective, final checkpoint only."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_s2t_local import S2TLocalTrainingConfig,listwise_loss,register_reserved_row_gradient_mask
from utils import S2TLocalDataset,collate_s2t_local_groups,add_lora_to_student,get_s2t_reserved_token_ids,score_candidates_from_reserved_logits,set_seed
from memory_objective import prefix_kl_from_hidden


def group_loss(student,reference,batch,reserved,config,chunk_tokens=64,selection_scale=1.0,prefix_scale=1.0):
    ids,mask=batch['input_ids'],batch['attention_mask']
    hidden=student.get_base_model().model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    with torch.no_grad():
        frozen=reference.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    positions=mask.sum(-1)-1
    last=hidden[torch.arange(hidden.shape[0],device=hidden.device),positions]
    weight=student.get_output_embeddings().weight
    bins=F.linear(last,weight).index_select(-1,reserved)
    scores,_=score_candidates_from_reserved_logits(bins,config.label_mode,config.score_mode,config.bin_softmax_temperature)
    scores=scores.view(int(batch['batch_size']),int(batch['num_candidates']))
    selection,metrics=listwise_loss(scores,batch['teacher_abs'],batch['candidate_mask'],config)
    lm=prefix_kl_from_hidden(hidden,frozen,weight,reference.get_output_embeddings().weight,
        mask,positions,chunk_tokens=chunk_tokens)
    return selection_scale*selection+prefix_scale*config.lm_kl_weight*lm,dict(**metrics,selection=float(selection.detach()),lm_kl=float(lm.detach()))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261004)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--kl-chunk',type=int,default=64)
    a=p.parse_args()
    assert min(a.epochs,a.microbatch,a.accumulation,a.kl_chunk)>0 and a.lr>0
    root=Path(os.environ.get('AUTO3_ROOT',str(Path(__file__).resolve().parents[1] / 'work')))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(out/'.training.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'summary.json').exists():
        manifest=json.loads((out/'final/checkpoint_manifest.json').read_text())
        assert all(digest(out/'final'/name)==info['sha256'] for name,info in manifest['files'].items())
        print('Completed checkpoint verified; no repeated training.')
        return
    if (out/'training.jsonl').exists() or (out/'final').exists():
        raise RuntimeError('Incomplete previous training evidence: inspect it and use a new run for a justified repair.')
    source=Path(a.input)
    records=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    train={r['id']:r for r in [json.loads(s) for s in (root/'data/math_train.jsonl').read_text().splitlines()]}
    assert records and len({r['group_id'] for r in records})==len(records)
    for r in records:
        assert r['problem_id'] in train and r['question_sha256']==train[r['problem_id']]['question_sha256']
        assert len(r['cand_ids'])==16 and len(r['teacher_prob_abs'])==16
        assert 'teacher' in r['teacher_label_source']
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy: raise RuntimeError('GPU occupied: wait for the existing experiment and smoke queue.')
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from huggingface_hub import snapshot_download
    base=json.loads((root/'data/manifest.json').read_text())
    model_path=snapshot_download(base['model'],revision=base['model_revision'],local_files_only=True)
    cfg=S2TLocalTrainingConfig(seed=a.seed,slm_model=model_path,slm_device='cuda:0',reference_device='cuda:0',
        output_dir=str(out),data_glob=str(source),batch_size=a.microbatch,grad_accum_steps=a.accumulation,
        num_epochs=a.epochs,lr=a.lr,num_workers=0,save_steps=0)
    provenance=dict(arguments=vars(a),upstream_config=asdict(cfg),input_sha256=digest(source),groups=len(records),
        model=base['model'],model_revision=base['model_revision'],
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        effective_group_batch=a.microbatch*a.accumulation,
        adaptation='Pinned repository objective and head gradient mask; token-wise projection recomputation, backbone activation checkpointing, group-weighted selection and token-weighted LM microbatch accumulation, final checkpoint only.',
        caveat='This is not the different paper-appendix recipe. Microbatching and mixed-precision accumulation need not be bitwise identical to a full batch. Nonreserved lm_head decay follows the released AdamW implementation.')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==provenance
    else:save(out/'config.json',provenance)
    set_seed(a.seed); torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    student=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    reference=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    reference.requires_grad_(False)
    student=add_lora_to_student(student,cfg.lora_rank,cfg.lora_alpha,cfg.lora_dropout)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    student.enable_input_require_grads()
    student.config.use_cache=False
    reserved=torch.tensor(get_s2t_reserved_token_ids(tokenizer,student.get_input_embeddings().weight.shape[0],16),device='cuda')
    hook=register_reserved_row_gradient_mask(student,reserved)
    dataset=S2TLocalDataset(str(source),16,512,16)
    assert len(dataset)==len(records)
    loader=DataLoader(dataset,batch_size=a.microbatch,shuffle=True,num_workers=0,
        collate_fn=lambda rows:collate_s2t_local_groups(rows,pad))
    params=[v for v in student.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.01)
    start=time.time(); steps=0
    for epoch in range(a.epochs):
        student.train(); optimizer.zero_grad(set_to_none=True); losses=[]; agreements=[]
        iterator=iter(loader); microsteps=0
        for _ in range(0,len(loader),a.accumulation):
            window=list(itertools.islice(iterator,a.accumulation))
            group_counts=[int(b['batch_size']) for b in window]
            prefix_counts=[int(b['attention_mask'].sum())-b['input_ids'].shape[0] for b in window]
            for batch,ng,nt in zip(window,group_counts,prefix_counts):
                batch={k:(v.to('cuda') if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
                loss,metrics=group_loss(student,reference,batch,reserved,cfg,a.kl_chunk,
                    selection_scale=ng/sum(group_counts),prefix_scale=nt/max(1,sum(prefix_counts)))
                assert torch.isfinite(loss),'Nonfinite objective'
                loss.backward()
                losses.append(metrics['selection']+cfg.lm_kl_weight*metrics['lm_kl'])
                agreements.append(metrics['candidate_hit1']); microsteps+=1
                if microsteps%25==0:print('train',epoch+1,microsteps,'/',len(loader),'loss',losses[-1],flush=True)
            torch.nn.utils.clip_grad_norm_(params,cfg.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); steps+=1
        record=dict(epoch=epoch+1,mean_loss=sum(losses)/len(losses),teacher_agree_at_1=sum(agreements)/len(agreements),
            updates=steps,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    final=out/'final'; student.save_pretrained(final)
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in final.iterdir() if p.is_file()}
    save(final/'checkpoint_manifest.json',dict(complete=True,files=files))
    save(out/'summary.json',dict(status='complete',epochs=a.epochs,groups=len(records),updates=steps,
        trainable_parameters=sum(v.numel() for v in params),elapsed_seconds=time.time()-start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint=str(final),generation_adapter='enabled_as_in_upstream'))
    hook.remove()


if __name__=='__main__':main()
