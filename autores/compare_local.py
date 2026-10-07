"""Compare complete paired predictions without changing the frozen metric."""
import argparse
import json
import ast
import subprocess
from pathlib import Path
from evaluate_local import read,save,digest,REPO


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--method',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();bdir=Path(a.baseline);mdir=Path(a.method)
    bs=json.loads((bdir/'summary.json').read_text());ms=json.loads((mdir/'summary.json').read_text())
    assert bs['complete'] and ms['complete'] and bs['n']==ms['n']
    assert bs['prediction_sha256']==digest(bdir/'predictions.jsonl') and ms['prediction_sha256']==digest(mdir/'predictions.jsonl')
    bc=json.loads((bdir/'config.json').read_text());mc=json.loads((mdir/'config.json').read_text())
    assert bc['input_sha256']==mc['input_sha256'] and bc['calibration_sha256']==mc['calibration_sha256']
    assert bc.get('sampling')==mc.get('sampling')
    br=dict(bc['runtime']);mr=dict(mc['runtime'])
    for key in ['s2t_lora_ckpt','output_dir']:br.pop(key,None);mr.pop(key,None)
    assert br==mr,'Incompatible inference protocols'
    assert bc['comparator_identity']==mc['comparator_identity']
    if bc['git_sha']!=mc['git_sha']:
        # A method change may use another commit; still require exact shared
        # native decoder, prompt, grader, and calibration implementation.
        for name in ['scripts/evaluate_s2t.py','scripts/utils.py','scripts/parser.py']:
            before=subprocess.check_output(['git','show',bc['git_sha']+':'+name],cwd=REPO)
            after=subprocess.check_output(['git','show',mc['git_sha']+':'+name],cwd=REPO)
            assert before==after,'Shared evaluation implementation changed: '+name
        def functions_at(sha):
            code=subprocess.check_output(['git','show',sha+':autores/evaluate_local.py'],cwd=REPO,text=True)
            return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(code).body if isinstance(node,ast.FunctionDef)}
        before,after=functions_at(bc['git_sha']),functions_at(mc['git_sha'])
        assert all(before[k]==after[k] for k in ['grade','freeze_entropy'])
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
        advance_to_main=bool(development and method>base and me>be and errors==0),
        raw_main_improvement=bool(not development and method>base and errors==0),
        evaluation_scope=ms['evidence_level'],sampling=mc.get('sampling'),
        baseline_generation_mode=bc.get('generation_adapter_mode','native'),method_generation_mode=mc.get('generation_adapter_mode','native'),
        baseline_source=bc['git_sha'],method_source=mc['git_sha'],
        interpretation='Single-seed paired comparison on every prespecified evaluation item. Fixed benchmark subsets are rapid exploratory evidence, not full-set superiority over published scores. Development is screening only; retain all outcomes and do not claim statistical significance.')
    save(Path(a.output),result);print(json.dumps({k:v for k,v in result.items() if not k.endswith('_ids')},indent=2))


if __name__=='__main__':main()
