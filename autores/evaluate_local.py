"""Single-seed complete-split evaluation using the released entropy S2T policy.

Calibration uses existing train-only traces, frozen identically for bootstrap and
self-evolved checkpoints. This does not reproduce the paper's learned router.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ.get('AUTO3_ROOT',str(Path(__file__).resolve().parents[1] / 'work')))
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def freeze_entropy():
    import numpy as np
    source=ROOT/'results/teacher_bootstrap_v1/collection/calibration_rollouts.jsonl'
    cal=read(ROOT/'data/math_calibration.jsonl');rows=read(source)
    assert len(cal)==len(rows)==32
    assert [r['id'] for r in rows]==[r['id'] for r in cal]
    assert all(r['question_sha256']==c['question_sha256'] for r,c in zip(rows,cal))
    assert all('/train:' in r['id'] for r in rows)
    values=[v for r in rows for v in r['result']['d_entropy_history']]
    assert values and np.isfinite(values).all()
    result=dict(quantile=0.99,threshold=float(np.quantile(values,0.99)),steps=len(values),
        source=str(source),source_sha256=digest(source),ids=[r['id'] for r in cal],
        meaning='Top1% entropy on base-student training-calibration greedy traces; fixed for bootstrap and refined checkpoints. Uses released entropy_s2t_local, not paper router or KL policy.')
    path=ROOT/'results/teacher_bootstrap_v1/entropy_calibration.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:save(path,result)
    return path,result


def grade(text,gold,dataset,explicit=False):
    from parser import extract_answer,math_equal
    def expired(*_):raise TimeoutError('Symbolic grading exceeded3seconds')
    old=signal.signal(signal.SIGALRM,expired);signal.alarm(3)
    try:
        pred=extract_answer(text,dataset,use_last_number=not explicit)
        return bool(pred and math_equal(pred,gold)),pred,None
    except Exception as exc:return False,None,repr(exc)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--generation-adapter-mode',choices=['native','base_proposals'],default='native')
    a=p.parse_args();source=Path(a.input).resolve();checkpoint=Path(a.checkpoint).resolve();out=Path(a.output)
    allowed={'math_dev.jsonl':(128,1024),'math500_test.jsonl':(500,1024),'gsm8k_test.jsonl':(1319,1024),'aime25_test.jsonl':(30,4096),
        'math_dev_quick64.jsonl':(64,1024),'math500_quick128.jsonl':(128,1024),'gsm8k_quick128.jsonl':(128,1024)}
    assert source.parent==(ROOT/'data').resolve() and source.name in allowed
    expected,cap=allowed[source.name];rows=read(source);assert len(rows)==expected
    assert len({r['id'] for r in rows})==expected
    sampling=None
    if '_quick' in source.name:
        subset_manifest=ROOT/'data/quick_subset_manifest.json'
        sampling=dict(json.loads(subset_manifest.read_text())['subsets'][source.name],manifest_sha256=digest(subset_manifest))
        assert sampling['subset_sha256']==digest(source) and sampling['subset_n']==expected
        assert [r['id'] for r in rows]==sampling['ids']
        assert digest(ROOT/'data'/sampling['source'])==sampling['source_sha256']
    manifest_path=checkpoint/'checkpoint_manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['complete']
    for name,record in manifest['files'].items():assert digest(checkpoint/name)==record['sha256']
    calibration_path,calibration=freeze_entropy()
    out.mkdir(parents=True,exist_ok=True);lock=open(out/'.evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from huggingface_hub import snapshot_download
    from evaluate_s2t import EvaluationConfig,S2TDecoder,build_decoding_policies
    from utils import load_model_bundle,set_seed
    base=json.loads((ROOT/'data/manifest.json').read_text())
    student=snapshot_download(base['model'],revision=base['model_revision'],local_files_only=True)
    dataset=rows[0]['dataset'];assert all(r['dataset']==dataset for r in rows)
    config=EvaluationConfig(seed=20261004,dataset=dataset,slm_model=student,llm_model='',slm_device='cuda:0',
        candidate_k=16,max_new_tokens=cap,entropy_threshold=calibration['threshold'],trigger_quantile=0.99,
        use_s2t_lora=True,s2t_lora_ckpt=str(checkpoint),policy_spec='entropy_s2t_local',output_dir=str(out))
    frozen=dict(arguments=vars(a),input_sha256=digest(source),checkpoint_manifest_sha256=digest(manifest_path),
        calibration_sha256=digest(calibration_path),runtime=asdict(config),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        comparator_identity='Released entropy-triggered S2T-local with quantized-teacher training adaptation; not KL S2T or paper learned router.',
        sampling=sampling,
        generation_adapter_mode=a.generation_adapter_mode,
        method_note='Base model generates proposals and KV states; bootstrap LoRA/head used only for independent candidate scoring.' if a.generation_adapter_mode=='base_proposals' else 'Released shared adapter for generation and scoring.',
        evidence_level='reused_development' if source.name.startswith('math_dev') else ('fixed_benchmark_subset' if sampling else 'complete_benchmark'))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==frozen
    else:save(out/'config.json',frozen)
    predictions=read(out/'predictions.jsonl');known={r['id']:r for r in predictions};valid={r['id']:r for r in rows}
    assert len(known)==len(predictions) and set(known)<=set(valid)
    assert all(r['question_sha256']==valid[r['id']]['question_sha256'] for r in predictions)
    if len(predictions)<expected:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
        torch.set_num_threads(8);set_seed(20261004)
        bundle=load_model_bundle(student,load_llm=False,slm_device='cuda:0',s2t_lora_ckpt=str(checkpoint),use_s2t_lora=True)
        from split_adapter import SplitAdapterDecoder
        Decoder=SplitAdapterDecoder if a.generation_adapter_mode=='base_proposals' else S2TDecoder
        class ObservedDecoder(Decoder):
            def initialize_state(self,*args,**kwargs):
                self.observed_state=super().initialize_state(*args,**kwargs)
                return self.observed_state
        decoder=ObservedDecoder(config,bundle);policy=build_decoding_policies('entropy_s2t_local',16,1)[0]
        (out/'trace').mkdir(exist_ok=True)
        for i,row in enumerate(rows):
            if row['id'] in known:continue
            set_seed(20261004)
            result=decoder.generate_hybrid(row['question'],dataset,policy,i,str(out/'trace'))
            state=decoder.observed_state;ids=state.generated_ids[state.prompt_len:]
            complete=bool(ids and ids[-1] in decoder.stop_ids)
            ok,pred,error=grade(result['generated_text'],row['answer'],dataset)
            explicit,explicit_pred,explicit_error=grade(result['generated_text'],row['answer'],dataset,True)
            record=dict(id=row['id'],question_sha256=row['question_sha256'],gold=row['answer'],text=result['generated_text'],
                correct=ok,prediction=pred,grading_error=error,explicit_correct=explicit,explicit_prediction=explicit_pred,
                explicit_error=explicit_error,completed=complete,truncated=not complete,generated_ids=ids,
                tokens=result['total_steps'],seconds=result['generation_time_sec'],triggers=result['trigger_count'],flips=result['flipped_num'])
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
            predictions.append(record);known[row['id']]=record
            print('evaluate',len(predictions),'/',expected,flush=True)
    assert len(predictions)==expected and set(known)==set(valid)
    save(out/'summary.json',dict(complete=True,n=expected,correct=sum(r['correct'] for r in predictions),
        accuracy_percent=100*sum(r['correct'] for r in predictions)/expected,
        completed_explicit_correct=sum(r['explicit_correct'] and r['completed'] for r in predictions),
        truncated=sum(r['truncated'] for r in predictions),tokens=sum(r['tokens'] for r in predictions),
        seconds=sum(r['seconds'] for r in predictions),triggers=sum(r['triggers'] for r in predictions),
        grading_errors=sum(bool(r['grading_error'] or r['explicit_error']) for r in predictions),
        prediction_sha256=digest(out/'predictions.jsonl'),evidence_level=frozen['evidence_level'],
        generation_adapter_mode=a.generation_adapter_mode,
        sampling=frozen['sampling'],comparison_caveat='Fixed-subset percentages cannot establish full-dataset superiority over published scores.' if sampling else None))
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':main()
