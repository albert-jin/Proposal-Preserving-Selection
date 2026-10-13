# Source attribution

The shared decoder, training objective, dataset utilities, and answer parser in
`scripts/` are retained from the server checkout of Select-to-Think.
The upstream paper is
[Select to Think: Unlocking SLM Potential with Local Sufficiency](https://proceedings.mlr.press/v306/ye26r.html).
The shared decoder and training objective retain their original behavior.
The answer parser removes an unused dataset alias and obsolete comments;
answer extraction and mathematical equivalence for the supported benchmarks
retain their original behavior.

The deployment separation in `pps/split_adapter.py` and the accompanying
training and evaluation wrappers implement PPS. The release pins model and
dataset revisions, validates source and checkpoint hashes, and uses real
training-calibration questions for GPU compatibility checks. The candidate
scoring rule, generation route, and training objective are preserved.

The supplied source snapshot did not contain a standalone license file. This
package does not assign a new license to third-party code. Model and dataset
licenses remain those of their respective providers.

Figures in `assets/paper/` are exported from the current anonymous manuscript.
They are manuscript figures, not newly generated outputs of a release-time run.
