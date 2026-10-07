"""Check upstream online teacher-rerank on one card, only when GPU is idle.

This is a training-calibration hardware smoke test, never a benchmark result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--max-tokens',type=int,default=128)
    a = p.parse_args()
    root = Path(os.environ.get('AUTO3_ROOT',str(Path(__file__).resolve().parents[1] / 'work')))
    out = Path(a.output)
    if (out/'summary.json').exists():
        raise RuntimeError('Existing hardware evidence: inspect it instead of overwriting or repeating.')
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:
        raise RuntimeError('GPU occupied; leave the running student experiment untouched.')
    teacher = json.loads((root/'docs/quantized_teacher_download.json').read_text())
    teacher_smoke = json.loads((root/'docs/quantized_teacher_smoke.json').read_text())
    assert teacher_smoke['status'] == 'single_train_context_gpu_smoke_passed_not_baseline_reproduction'
    assert teacher_smoke['revision'] == teacher['revision']
    teacher_path = teacher_smoke['runtime_model_path']
    out.mkdir(parents=True,exist_ok=True)
    import torch
    from huggingface_hub import snapshot_download
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'scripts'))
    from evaluate_s2t import EvaluationConfig, S2TDecoder, build_decoding_policies
    from utils import load_model_bundle, set_seed
    source_files = ['scripts/evaluate_s2t.py','scripts/utils.py']
    for name in source_files:
        expected_hashes = json.loads((repo/'configs/upstream_source_hashes.json').read_text())
        normalized = (repo/name).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(normalized).hexdigest() == expected_hashes[name]
    manifest = json.loads((root/'data/manifest.json').read_text())
    student = snapshot_download(manifest['model'],revision=manifest['model_revision'],local_files_only=True)
    cal = [json.loads(s) for s in (root/'data/math_calibration.jsonl').read_text().splitlines()]
    assert len(cal) == 32 and all('/train:' in r['id'] for r in cal)
    row = max(cal,key=lambda r:len(r['question']))
    config = EvaluationConfig(seed=20261004,slm_model=student,llm_model=teacher_path,
        slm_device='cuda:0',llm_device='cuda:0',dataset='math',candidate_k=16,
        max_new_tokens=a.max_tokens,kl_threshold=0.0,collect_s2t_data=True,
        s2t_data_dir=str(out/'smoke_groups'),output_dir=str(out/'logs'))
    Path(config.s2t_data_dir).mkdir(exist_ok=True)
    Path(config.output_dir).mkdir(exist_ok=True)
    record=dict(evidence_level='hardware_smoke_train_calibration_only',question_id=row['id'],
        question_sha256=row['question_sha256'],max_tokens=a.max_tokens,seed=20261004,
        teacher_revision=teacher['revision'],student_revision=manifest['model_revision'],
        runtime_teacher_path=teacher_path,tokenizer_adaptation=teacher_smoke['tokenizer_compatibility'],
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        trigger='KL threshold0 solely to exercise teacher candidate selection, not a calibrated experiment',
        source_hashes={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in source_files},
        caveat='One bounded question does not prove every longer-context evaluation will fit. Smoke groups must not enter model training.')
    (out/'config.json').write_text(json.dumps(record,indent=2))
    torch.set_num_threads(8)
    set_seed(config.seed)
    started=time.time()
    try:
        bundle=load_model_bundle(slm_model=student,llm_model=teacher_path,
            slm_device='cuda:0',llm_device='cuda:0',local_files_only=True)
        assert bundle.same_vocab and getattr(bundle.llm,'is_loaded_in_4bit',False)
        record['load_seconds']=time.time()-started
        record['memory_after_loading']=torch.cuda.memory_allocated()
        decoder=S2TDecoder(config,bundle)
        result=decoder.generate_hybrid(row['question'],'math',build_decoding_policies('teacher_rerank',16,1)[0],0,str(out/'logs'))
        (out/'generated.json').write_text(json.dumps(result,indent=2))
        assert result['total_steps']>0 and result['trigger_count']>0
        record.update(status='joint_online_teacher_rerank_smoke_passed',
            generated_steps=result['total_steps'],trigger_count=result['trigger_count'],
            generation_seconds=result['generation_time_sec'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except torch.OutOfMemoryError as exc:
        record.update(status='joint_online_out_of_memory',error=str(exc),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (out/'summary.json').write_text(json.dumps(record,indent=2))
        raise
    (out/'summary.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
