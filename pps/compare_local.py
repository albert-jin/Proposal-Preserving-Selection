"""Compare complete paired predictions without changing the frozen metric."""
import argparse
import json
from pathlib import Path
from evaluate_local import read,save,digest


def validate_pair(baseline, method):
    """Compare deployment modes only when all experimental inputs match."""
    for key in ('input_sha256', 'calibration_sha256', 'checkpoint_manifest_sha256',
                'comparator_identity', 'evidence_level'):
        assert baseline[key] == method[key], 'Paired runs differ: ' + key
    assert baseline.get('sampling') == method.get('sampling'), 'Paired subsets differ'
    assert baseline.get('generation_adapter_mode') == 'native', 'Baseline must use native deployment'
    assert method.get('generation_adapter_mode') == 'base_proposals', 'Method must use PPS deployment'
    sources = baseline.get('evaluation_source_sha256')
    assert sources and sources == method.get('evaluation_source_sha256'), \
        'Paired runs require matching source fingerprints; rerun older evaluations'
    br, mr = dict(baseline['runtime']), dict(method['runtime'])
    for key in ('s2t_lora_ckpt', 'output_dir'):
        br.pop(key, None)
        mr.pop(key, None)
    assert br == mr, 'Incompatible inference protocols'


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--method',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();bdir=Path(a.baseline);mdir=Path(a.method)
    bs=json.loads((bdir/'summary.json').read_text());ms=json.loads((mdir/'summary.json').read_text())
    assert bs['complete'] and ms['complete'] and bs['n']==ms['n']
    assert bs['prediction_sha256']==digest(bdir/'predictions.jsonl') and ms['prediction_sha256']==digest(mdir/'predictions.jsonl')
    bc=json.loads((bdir/'config.json').read_text());mc=json.loads((mdir/'config.json').read_text())
    validate_pair(bc,mc)
    assert bs['evidence_level']==bc['evidence_level'] and ms['evidence_level']==mc['evidence_level']
    b=read(bdir/'predictions.jsonl');m=read(mdir/'predictions.jsonl');bd={r['id']:r for r in b};md={r['id']:r for r in m}
    assert len(b)==len(m)==len(bd)==len(md)==bs['n'] and set(bd)==set(md)
    assert all(bd[k]['gold']==md[k]['gold'] and bd[k]['question_sha256']==md[k]['question_sha256'] for k in bd)
    wins=[k for k in bd if md[k]['correct'] and not bd[k]['correct']];losses=[k for k in bd if bd[k]['correct'] and not md[k]['correct']]
    base=sum(r['correct'] for r in b);method=sum(r['correct'] for r in m)
    be=sum(r['explicit_correct'] and r['completed'] for r in b);me=sum(r['explicit_correct'] and r['completed'] for r in m)
    errors=sum(bool(r['grading_error'] or r['explicit_error']) for r in b+m)
    development=bs['evidence_level']==ms['evidence_level']=='reused_development'
    result=dict(n=len(b),baseline_correct=base,method_correct=method,delta_percentage_points=100*(method-base)/len(b),
        wins=len(wins),losses=len(losses),win_ids=wins,loss_ids=losses,baseline_completed_explicit=be,method_completed_explicit=me,
        grading_errors=errors,baseline_identity=bc['comparator_identity'],development_only=development,
        evaluation_scope=ms['evidence_level'],sampling=mc.get('sampling'),
        baseline_generation_mode=bc.get('generation_adapter_mode','native'),method_generation_mode=mc.get('generation_adapter_mode','native'),
        baseline_source=bc['git_sha'],method_source=mc['git_sha'],
        interpretation='Single-seed comparison with the same checkpoint and evaluation inputs. Subset and development results do not establish full-benchmark performance or statistical significance.')
    save(Path(a.output),result);print(json.dumps({k:v for k,v in result.items() if not k.endswith('_ids')},indent=2))


if __name__=='__main__':main()
