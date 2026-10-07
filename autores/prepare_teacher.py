"""Download one pinned quantized teacher, without loading it or touching GPU."""
import hashlib
import json
import os
from pathlib import Path
import time
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ.get('AUTO3_ROOT', str(Path(__file__).resolve().parents[1] / 'work')))
(root/'docs').mkdir(parents=True, exist_ok=True)
repo = 'unsloth/Qwen2.5-32B-Instruct-bnb-4bit'
revision = 'aa79e3472818bdec779075d80928602591d9f2a0'
info = HfApi(endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'), token=False).model_info(repo,revision=revision,files_metadata=True,timeout=30)
assert info.sha == revision
files = [s for s in info.siblings if s.rfilename.endswith('.safetensors')]
assert len(files) == 4 and sum(s.size for s in files) == 19215448557
path = Path(snapshot_download(repo,revision=revision,token=False,max_workers=2,
    allow_patterns=['*.json','*.safetensors','*.model','merges.txt','vocab.json','LICENSE','README.md']))
records = []
for f in files:
    p = path/f.rfilename
    assert p.stat().st_size == f.size
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    expected = f.lfs.sha256 if f.lfs else None
    assert expected and h.hexdigest() == expected, f.rfilename
    records.append(dict(file=f.rfilename,bytes=p.stat().st_size,sha256=h.hexdigest()))
config=json.loads((path/'config.json').read_text())
result=dict(time=time.time(),repo=repo,revision=revision,path=str(path),files=records,
    quantization_config=config.get('quantization_config'),status='downloaded_hash_verified_not_gpu_tested',
    purpose='Teacher-supervised resource-adapted S2T control and potential Self-Evolving bootstrap. Quantization is a disclosed departure from full-precision teacher.')
(root/'docs/quantized_teacher_download.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
