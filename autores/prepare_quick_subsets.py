"""Freeze the user's rapid-evaluation subsets before any new model scores."""
import datetime
import json
import random
from evaluate_local import ROOT,read,digest,save


def main():
    choices=[('math_dev.jsonl','math_dev_quick64.jsonl',64),
             ('math500_test.jsonl','math500_quick128.jsonl',128),
             ('gsm8k_test.jsonl','gsm8k_quick128.jsonl',128)]
    records={}
    for source_name,target_name,count in choices:
        source=ROOT/'data'/source_name;target=ROOT/'data'/target_name;rows=read(source)
        assert len({r['id'] for r in rows})==len(rows)>count
        indices=sorted(random.Random(20261004).sample(range(len(rows)),count))
        selected=[rows[i] for i in indices]
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
        if target.exists():assert target.read_text()==content,'Do not change an existing subset'
        else:target.write_text(content)
        records[target_name]=dict(source=source_name,source_sha256=digest(source),subset_sha256=digest(target),
            source_n=len(rows),subset_n=count,indices=indices,ids=[r['id'] for r in selected],fraction=count/len(rows),seed=20261004,
            sampling='Uniform without replacement, then original source order. No scores or labels used to select indices.')
    path=ROOT/'data/quick_subset_manifest.json'
    if path.exists():assert json.loads(path.read_text())['subsets']==records
    else:save(path,dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='User explicitly prioritized fixed-subset evaluation speed; previous full-evaluation queue had not started.',
        subsets=records,small_dataset='AIME25 retains all30 questions because already small.',
        comparison='Use identical subset for bootstrap and refinement. Published full-set values are separate references, not paired controls.'))
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ['indices','ids']} for k,v in records.items()},indent=2))


if __name__=='__main__':main()
