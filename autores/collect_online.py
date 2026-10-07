"""Training-only wrapper around the unchanged upstream online teacher decoder.

No benchmark answers are read by generation. Quantization and dataset routing
are explicit adaptations; the paper and README configurations differ.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value)+'\n')
        f.flush()


class BudgetReached(Exception):
    pass


def pack_group(row,step,prefix,candidates,probabilities):
    assert len(candidates)==16 and len(set(candidates))==16
    assert len(probabilities)==16 and all(math.isfinite(x) and 0<=x<=1 for x in probabilities)
    assert 0<sum(probabilities)<=1+1e-6
    assert prefix
    return dict(group_id=row['id']+':step'+str(step),problem_id=row['id'],
        question_sha256=row['question_sha256'],step=step,context_ids=list(prefix[-511:]),
        cand_ids=candidates,teacher_prob_abs=probabilities,teacher_sum_on_K=sum(probabilities),
        k_eff=len(candidates),candidate_source='slm_topk',
        teacher_label_source='quantized_teacher_full_distribution_on_candidate_ids',
        full_context_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--max-questions',type=int,default=512)
    p.add_argument('--max-groups',type=int,default=2000)
    p.add_argument('--quantile',type=float,default=0.90,help='Paper D.1 top10%%; README recipe instead uses0.99. Explicitly record choice.')
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--seed',type=int,default=20261004)
    a=p.parse_args()
    assert a.max_questions>0 and a.max_groups>0 and 0<a.quantile<1
    root=Path(os.environ.get('AUTO3_ROOT',str(Path(__file__).resolve().parents[1] / 'work')))
    repo=Path(__file__).resolve().parents[1]
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    train=read(root/'data/math_train.jsonl')
    cal=read(root/'data/math_calibration.jsonl')
    dev=read(root/'data/math_dev.jsonl')
    assert len(train)==7340 and len(cal)==32 and len(dev)==128
    assert len({r['id'] for r in train+cal+dev})==7500
    assert all('/train:' in r['id'] for r in train+cal+dev)
    assert a.max_questions<=len(train)
    rows=train[:a.max_questions]
    teacher=json.loads((root/'docs/quantized_teacher_download.json').read_text())
    joint=json.loads((root/'results/teacher_joint_smoke/summary.json').read_text())
    assert joint['status']=='joint_online_teacher_rerank_smoke_passed'
    assert joint['teacher_revision']==teacher['revision']
    teacher_path=joint['runtime_teacher_path']
    source={}
    for name in ['scripts/evaluate_s2t.py','scripts/utils.py']:
        expected_hashes = json.loads((repo/'configs/upstream_source_hashes.json').read_text())
        normalized = (repo/name).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(normalized).hexdigest() == expected_hashes[name]
        source[name]=digest(repo/name)
    cfg=dict(arguments=vars(a),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        teacher_repo=teacher['repo'],teacher_revision=teacher['revision'],teacher_precision='prequantized_bnb4bit',
        runtime_teacher_path=teacher_path,tokenizer_adaptation=joint['tokenizer_adaptation'],
        train_sha256=digest(root/'data/math_train.jsonl'),calibration_sha256=digest(root/'data/math_calibration.jsonl'),
        question_ids=[r['id'] for r in rows],calibration_ids=[r['id'] for r in cal],upstream_source_sha256=source,
        protocol='Upstream online KL-triggered teacher-rerank, K16, top64 trigger support, warmup3. Calibration and collection only on safe original train partition.',
        budget_note='First max_groups valid teacher states; final question may stop immediately after its final required group. This is not a benchmark evaluation.',
        source_note='Paper section5 says2k trajectories while D.1 says2k instances. This run treats the budget as candidate groups and reports question count separately.')
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==cfg, 'Do not change a resumed collection protocol'
    else:
        save(out/'config.json',cfg)
    groups=read(out/'groups.jsonl')
    known={r['group_id']:r for r in groups}
    assert len(groups)==len(known) and len(groups)<=a.max_groups
    valid={r['id']:r for r in rows}
    for r in groups:
        assert r['problem_id'] in valid and r['question_sha256']==valid[r['problem_id']]['question_sha256']
        assert len(r['cand_ids'])==16 and len(r['teacher_prob_abs'])==16
    if len(groups)==a.max_groups:
        save(out/'summary.json',dict(complete=True,groups=len(groups),label_sha256=digest(out/'groups.jsonl'),
            questions_represented=len({r['problem_id'] for r in groups})))
        print('Exact requested label budget already complete; no repeated teacher inference.')
        return
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:
        raise RuntimeError('GPU occupied: do not race the current pipeline or smoke queue.')
    import numpy as np
    import torch
    sys.path.insert(0,str(repo/'scripts'))
    from evaluate_s2t import EvaluationConfig,S2TDecoder,DecodingPolicy,build_decoding_policies
    from utils import load_model_bundle,set_seed
    from huggingface_hub import snapshot_download
    base=json.loads((root/'data/manifest.json').read_text())
    student=snapshot_download(base['model'],revision=base['model_revision'],local_files_only=True)
    runtime=EvaluationConfig(seed=a.seed,slm_model=student,llm_model=teacher_path,slm_device='cuda:0',llm_device='cuda:0',
        dataset='math',candidate_k=16,trigger_quantile=a.quantile,max_new_tokens=a.max_tokens,
        output_dir=str(out/'trace'),collect_s2t_data=False,max_context_len=512)
    (out/'trace').mkdir(exist_ok=True)
    save(out/'decoder_config.json',asdict(runtime))
    torch.set_num_threads(8)
    set_seed(a.seed)
    bundle=load_model_bundle(student,teacher_path,slm_device='cuda:0',llm_device='cuda:0',local_files_only=True)
    assert bundle.same_vocab and getattr(bundle.llm,'is_loaded_in_4bit',False)

    class Collector(S2TDecoder):
        def collect_training_group(self,policy,problem_id,step,prefix_ids,candidate_ids,teacher_scores):
            if not self.config.collect_s2t_data or policy.guidance_type!='teacher_rerank' or teacher_scores.sum().item()<=0:
                return
            row=rows[problem_id]
            record=pack_group(row,step,prefix_ids,candidate_ids.detach().cpu().tolist(),teacher_scores.detach().float().cpu().tolist())
            gid=record['group_id']
            if gid in known:
                old=known[gid]
                assert old['cand_ids']==record['cand_ids'] and old['full_context_sha256']==record['full_context_sha256'], 'Interrupted question replay diverged'
                assert np.allclose(old['teacher_prob_abs'],record['teacher_prob_abs'],rtol=1e-5,atol=1e-7)
                return
            append(out/'groups.jsonl',record)
            known[gid]=record
            if len(known)>=a.max_groups:
                raise BudgetReached()

    decoder=Collector(runtime,bundle)
    observed=read(out/'calibration_rollouts.jsonl')
    cal_done={r['id']:r for r in observed}
    assert len(observed)==len(cal_done) and set(cal_done)<={r['id'] for r in cal}
    policy=DecodingPolicy(name='Calibration',trigger_type='kl',guidance_type='student_greedy')
    for i,row in enumerate(cal):
        if row['id'] in cal_done:
            assert cal_done[row['id']]['question_sha256']==row['question_sha256']
            continue
        set_seed(a.seed+i)
        result=decoder.generate_hybrid(row['question'],'math',policy,i,str(out/'trace'))
        record=dict(id=row['id'],question_sha256=row['question_sha256'],result=result)
        append(out/'calibration_rollouts.jsonl',record)
        cal_done[row['id']]=record
        print('teacher_calibration',i+1,'/',len(cal),flush=True)
    values=[v for row in cal for v in cal_done[row['id']]['result']['d_kl_history']]
    assert values and np.isfinite(values).all()
    runtime.kl_threshold=float(np.quantile(values,a.quantile))
    runtime.collect_s2t_data=True
    save(out/'calibration.json',dict(kl_threshold=runtime.kl_threshold,quantile=a.quantile,n_steps=len(values),ids=[r['id'] for r in cal]))
    completed=read(out/'completed.jsonl')
    done={r['id'] for r in completed}
    assert len(done)==len(completed) and done<=set(valid)
    policy=build_decoding_policies('teacher_rerank',16,1)[0]
    for i,row in enumerate(rows):
        if row['id'] in done:
            continue
        set_seed(a.seed+i)
        started=time.time()
        try:
            result=decoder.generate_hybrid(row['question'],'math',policy,i,str(out/'trace'))
        except BudgetReached:
            append(out/'budget_stop.jsonl',dict(id=row['id'],groups=len(known),seconds=time.time()-started))
            break
        append(out/'completed.jsonl',dict(id=row['id'],seconds=time.time()-started,result=result))
        print('teacher_collect',i+1,'/',len(rows),'groups',len(known),'/',a.max_groups,flush=True)
    save(out/'summary.json',dict(complete=len(known)==a.max_groups,groups=len(known),
        questions_represented=len({r['problem_id'] for r in known.values()}),
        label_sha256=digest(out/'groups.jsonl') if known else None,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    if len(known)<a.max_groups:
        raise RuntimeError('Question cap reached before requested group budget; review evidence before changing protocol.')


if __name__=='__main__':
    main()
