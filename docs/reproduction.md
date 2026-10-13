# Reproduction details

## Implementation

`pps/split_adapter.py` implements deployment separation: the base model generates
proposals and generation KV states, while the trained adapter scores candidates
through independent forward passes. The shared decoder, training objective,
dataset utilities, and mathematical answer grader are in `scripts/`.

The executable workflow downloads pinned pretrained models and public datasets,
collects candidate probabilities from the teacher, trains a scoring adapter, and
evaluates native and PPS deployment using the same checkpoint. Model weights,
checkpoints, training examples, predictions, and execution logs are created under
`PPS_ROOT` (default: `<repository>/work`) and are excluded from version control.

## Paper and executable protocol

| Item | Current manuscript | Packaged executable workflow |
|---|---|---|
| Students | Qwen2.5-Instruct 0.5B, 1.5B, 7B | Qwen2.5-1.5B-Instruct |
| Seeds | 42, 1337, 2024 | 20261004 |
| Training teacher | Qwen2.5-32B-Instruct, BF16 | Pinned 4-bit export of Qwen2.5-32B-Instruct |
| Benchmark scope | Eight benchmarks | MATH500, GSM8K, AIME25 |
| Default evaluation | Paper-specific protocol | Complete splits: 500, 1,319, and 30 examples |
| Optional subset evaluation | Separate from paper tables | 128 MATH500, 128 GSM8K, and all 30 AIME25 examples |
| Candidate scoring | Expected reserved-bin score | 16 candidates and 16 bins |
| Scoring context | 512 tokens | 512 tokens |
| Evaluation trigger | 99th-percentile entropy | 99th-percentile entropy from training-calibration traces |

`configs/paper_settings.json` documents the manuscript settings; it does not
configure the executable workflow. `configs/server_data_manifest.json` pins its
student model, dataset revisions, split sizes, and data hashes.

`bash run_paired_eval.sh` evaluates complete supported benchmark splits by default.
Pass `quick` as its second argument to use fixed subsets. Subset sampling uses a
fixed seed without consulting model outputs or scores. Development data remains
separate from benchmark evaluation. Results retain their evaluation scope.

The collection quantile of 0.90 controls the KL trigger for teacher-label
collection. The evaluation quantile of 0.99 controls the entropy trigger. They
are distinct parameters. The collection budget counts candidate groups; it also
records the number of represented questions.

## Input and comparison checks

- Preparation verifies official split hashes and checks question overlap among
  MATH training, calibration, development, and MATH500 partitions.
- Evaluation verifies complete dataset hashes against the pinned manifest. For
  subsets, it also verifies source hashes, indices, IDs, and exact selected rows.
- Checkpoint files must match `checkpoint_manifest.json`. Calibration uses only
  the 32 training-calibration questions and the corresponding generation traces.
- Paired comparison requires the same checkpoint manifest, data, calibration,
  runtime settings, evaluation scope, and source fingerprints. It requires native
  deployment as the baseline and base-proposal deployment as the PPS method.
- Source fingerprints cover the actual decoder, grader, utilities, evaluation
  wrapper, and PPS routing wrapper. Older outputs without these fingerprints
  must be regenerated before comparison.
- Completed and resumed runs retain configuration and output checks. Use a fresh
  output directory when changing a setup.

## Real-model GPU validation

`pps/validate_teacher.py` checks the teacher's token IDs against the student
and performs a forward pass on a training-calibration question. The quantized
export includes an extra padding token; a separate runtime directory uses the
pinned student tokenizer without modifying the downloaded model files.

`pps/validate_joint.py` loads the real student and teacher together and runs
teacher-guided decoding on one training-calibration question. Its candidate groups
are written to a separate validation directory and must not enter training.
Collection requires the successful validation record and matching teacher revision.
These checks establish compatibility and bounded GPU memory use, not benchmark
accuracy or memory requirements for every sequence length.

## Environment and validation scope

Use Linux, Python 3.12, and a CUDA-enabled PyTorch installation. File locking,
grading timeouts, GPU-idle checks, and the teacher's tokenizer symlinks are part of
the Linux workflow. The original adaptation used a 24 GB GPU; context length and
runtime versions affect memory requirements.

The core requirements pin Transformers 4.57.3, PEFT 0.17.1, and Datasets 3.6.0.
Teacher-label collection additionally uses bitsandbytes 0.48.2. The requirements
are not a complete lockfile of the original server environment.

This source cleanup does not rerun GPU training or benchmark evaluation. README
scores are transcribed manuscript results, not measurements produced by this
cleanup. The packaged single-model protocol does not reproduce all manuscript
tables, model sizes, seeds, or ablations.

## Figure provenance

The figures are exported from the PDFs referenced by the manuscript.
`assets/paper/sources.json` records source filenames and SHA-256 hashes;
`assets/paper/manuscript_sources.json` records manuscript source hashes. README
numerical results are from Table 2 and are not inferred from the plots.
