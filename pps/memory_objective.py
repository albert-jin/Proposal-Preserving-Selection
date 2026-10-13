"""Token-chunked full-vocabulary LM KL, preserving the upstream mask/objective."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def prefix_kl_from_hidden(student_hidden, reference_hidden, student_weight,
                          reference_weight, attention_mask, action_positions,
                          chunk_tokens=64, recompute=True):
    if chunk_tokens<1:
        raise ValueError('chunk_tokens must be positive')
    if student_hidden.shape!=reference_hidden.shape:
        raise ValueError('Student and frozen-reference hidden shapes differ')
    mask=attention_mask.bool().clone()
    mask[torch.arange(mask.size(0),device=mask.device),action_positions]=False
    indices=mask.reshape(-1).nonzero(as_tuple=True)[0]
    if not indices.numel():
        return student_hidden.sum()*0+student_weight.sum()*0
    h=student_hidden.reshape(-1,student_hidden.shape[-1])
    reference=reference_hidden.detach().reshape_as(h)
    frozen_weight=reference_weight.detach()

    def contribution(x,r,w,wr):
        # Keep every vocabulary token and the original log-softmax precision.
        # Recompute this projection during backward instead of keeping all B*K*L*V logits.
        log_ref=F.log_softmax(F.linear(r,wr),dim=-1)
        log_student=F.log_softmax(F.linear(x,w),dim=-1)
        token_kl=(log_ref.exp()*(log_ref-log_student)).sum(dim=-1)
        # Upstream multiplies by an FP32 mask before its final sum.
        return token_kl.to(torch.promote_types(token_kl.dtype,torch.float32)).sum()

    terms=[]
    for selected in indices.split(chunk_tokens):
        args=(h.index_select(0,selected),reference.index_select(0,selected),student_weight,frozen_weight)
        if recompute and torch.is_grad_enabled():
            terms.append(checkpoint(contribution,*args,use_reentrant=False,preserve_rng_state=False))
        else:
            terms.append(contribution(*args))
    return torch.stack(terms).sum()/mask.sum().float().clamp_min(1)
