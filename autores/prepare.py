"""Download pinned inputs; split only the official MATH train partition."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(os.environ.get('AUTO3_ROOT', str(Path(__file__).resolve().parents[1] / 'work')))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from parser import parse_ground_truth, parse_question


def save(name, rows):
    path = ROOT / 'data' / (name + '.jsonl')
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'file': path.name, 'n': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def normalize(ds, kind, source):
    rows = []
    for i, ex in enumerate(ds):
        question = parse_question(ex, kind)
        _, answer = parse_ground_truth(ex, kind)
        rows.append(dict(id=f'{source}:{i}', question=question, answer=str(answer),
                         dataset=kind, question_sha256=hashlib.sha256(question.encode()).hexdigest()))
    return rows


if __name__ == '__main__':
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model = 'Qwen/Qwen2.5-1.5B-Instruct'
    pins = json.loads((Path(__file__).resolve().parents[1] / 'configs/server_data_manifest.json').read_text())
    revision = pins['model_revision']
    print('Downloading model', model, revision, flush=True)
    snapshot_download(model, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    manifest = {'model': model, 'model_revision': revision, 'seed': 20261004, 'sources': {}, 'files': []}
    specs = [('DigitalLearningGmbH/MATH-lighteval', 'default', 'train', 'math', 'math_train'),
             ('HuggingFaceH4/MATH-500', None, 'test', 'math', 'math500_test'),
             ('openai/gsm8k', 'main', 'test', 'gsm8k', 'gsm8k_test'),
             ('math-ai/aime25', None, 'test', 'aime25', 'aime25_test')]
    for repo, config, split, kind, name in specs:
        rev = pins['sources'][repo]['revision']
        print('Downloading dataset', repo, rev, flush=True)
        ds = load_dataset(repo, config, split=split, revision=rev)
        if name == 'math_train' and len(ds) != 7500:
            raise RuntimeError('Expected the original 7500-question MATH train split.')
        rows = normalize(ds, kind, f'{repo}/{split}')
        manifest['sources'][repo] = {'revision': rev, 'fingerprint': ds._fingerprint}
        if name == 'math_train':
            random.Random(20261004).shuffle(rows)
            parts = {'calibration': rows[:32], 'dev': rows[32:160], 'train': rows[160:]}
            for part, values in parts.items():
                manifest['files'].append(save('math_' + part, values))
        else:
            manifest['files'].append(save(name, rows))
    for actual in manifest['files']:
        expected = next(item for item in pins['files'] if item['file'] == actual['file'])
        assert actual == expected, f"Prepared dataset differs from frozen server split: {actual['file']}"
    # Hash-based overlap check is independent of numeric dataset identifiers.
    seen = {}
    for name in ['math_calibration', 'math_dev', 'math_train', 'math500_test']:
        rows = [json.loads(x) for x in (ROOT / 'data' / (name + '.jsonl')).read_text().splitlines()]
        hashes = {r['question_sha256'] for r in rows}
        for other, values in seen.items():
            if hashes & values:
                raise RuntimeError(f'Question overlap: {name} / {other}')
        seen[name] = hashes
    (ROOT / 'data' / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('PREPARE_COMPLETE', flush=True)
