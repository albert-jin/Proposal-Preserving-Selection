"""Run manually only after the active student GPU pipeline ends."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ.get('AUTO3_ROOT', str(Path(__file__).resolve().parents[1] / 'work')))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if processes:
    raise RuntimeError('Another GPU process is active; leave it untouched and run this smoke test later.')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

download = json.loads((root/'docs/quantized_teacher_download.json').read_text())
assert download['status'] == 'downloaded_hash_verified_not_gpu_tested'
teacher_path = download['path']
data_manifest = json.loads((root/'data/manifest.json').read_text())
student_path = snapshot_download(data_manifest['model'],revision=data_manifest['model_revision'],local_files_only=True)
st = AutoTokenizer.from_pretrained(student_path,local_files_only=True)
tt = AutoTokenizer.from_pretrained(teacher_path,local_files_only=True)
sv, tv = st.get_vocab(), tt.get_vocab()
assert all(tv.get(token) == index for token,index in sv.items()), 'Shared candidate IDs differ'
extra = {token:index for token,index in tv.items() if token not in sv}
assert extra == {'<|PAD_TOKEN|>': 151665}, 'Unexpected tokenizer difference; investigate before loading'
# The quantized export added one padding token. Use an explicit, separate runtime
# view with the original student tokenizer; never modify the pinned HF snapshot.
runtime = root/'models/teacher_student_tokenizer'
runtime.mkdir(parents=True,exist_ok=True)
tokenizer_files = {'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','chat_template.jinja'}
for source in Path(teacher_path).iterdir():
    if source.is_file() and source.name not in tokenizer_files:
        target = runtime/source.name
        if not target.exists(): target.symlink_to(source.resolve())
for source in Path(student_path).iterdir():
    if source.is_file() and source.name in tokenizer_files:
        target = runtime/source.name
        if not target.exists(): target.symlink_to(source.resolve())
assert AutoTokenizer.from_pretrained(runtime,local_files_only=True).get_vocab() == sv
runtime_record = dict(original_teacher_path=teacher_path,runtime_model_path=str(runtime),
    original_vocabulary_identical=False,shared_token_ids_identical=True,excluded_teacher_only_tokens=extra,
    adaptation='Quantized export padding token removed by using the pinned student tokenizer in a separate symlink view; model weights/config are unchanged.',
    tokenizer_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.iterdir() if p.name in tokenizer_files})
(root/'docs/teacher_tokenizer_compatibility.json').write_text(json.dumps(runtime_record,indent=2))
row = json.loads((root/'data/math_calibration.jsonl').read_text().splitlines()[0])
assert '/train:' in row['id']
ids = st.encode(row['question'], add_special_tokens=False)[-512:]
assert ids
group = dict(group_id=row['id'], cand_ids=list(dict.fromkeys(ids))[:16])
torch.set_num_threads(8)
torch.manual_seed(20261004)
start = time.time()
model = AutoModelForCausalLM.from_pretrained(runtime,local_files_only=True,
    torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
assert getattr(model,'is_loaded_in_4bit',False)
load_seconds = time.time()-start
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.time()
    logits = model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1).logits[0,-1].float()
    assert torch.isfinite(logits).all()
    probs = logits.softmax(-1)
    scores = probs[group['cand_ids']].cpu().tolist()
    torch.cuda.synchronize()
    forward_seconds = time.time()-start
result = dict(status='single_train_context_gpu_smoke_passed_not_baseline_reproduction',
    teacher_repo=download['repo'],revision=download['revision'],vocabulary_identical=True,
    runtime_model_path=str(runtime),tokenizer_compatibility=runtime_record,
    group_id=group['group_id'],context_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    input_tokens=len(ids),candidate_ids=group['cand_ids'],candidate_probabilities=scores,
    load_seconds=load_seconds,forward_seconds=forward_seconds,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__)
(root/'docs/quantized_teacher_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
