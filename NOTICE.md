# Source attribution

The shared decoder, training objective, dataset utilities, and answer parser in
`scripts/` are retained from the server checkout of Select-to-Think.
The upstream paper is
[Select to Think: Unlocking SLM Potential with Local Sufficiency](https://proceedings.mlr.press/v306/ye26r.html).
These components remain unchanged apart from line-ending normalization.

The deployment separation in `autores/split_adapter.py` and the accompanying
experiment wrappers are provided for the anonymous PPS submission. Packaging
changes make paths portable, pin archived input revisions, replace checks against
private Git history with file hashes, and make a hardware smoke test independent
of discarded pilot outputs. They do not change the candidate scoring rule,
generation route, or training loss.

The supplied source snapshot did not contain a standalone license file. This
package does not assign a new license to third-party code. Model and dataset
licenses remain those of their respective providers.

Figures in `assets/paper/` are exported from the current anonymous manuscript.
They are manuscript figures, not newly generated outputs of a release-time run.
