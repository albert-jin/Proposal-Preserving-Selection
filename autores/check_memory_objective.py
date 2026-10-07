"""CPU loss/gradient comparison with the actual pinned upstream KL function."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_s2t_local import language_model_kl
from memory_objective import prefix_kl_from_hidden


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    results=[]
    for dtype in [torch.float64,torch.float32,torch.bfloat16]:
        for chunk in [1,5,64]:
            torch.manual_seed(1234)
            h=torch.randn(3,9,7,dtype=dtype,requires_grad=True)
            w=torch.randn(29,7,dtype=dtype,requires_grad=True)
            rh=torch.randn_like(h)
            rw=torch.randn_like(w)
            mask=torch.tensor([[1]*9,[1]*6+[0]*3,[1]*4+[0]*5])
            pos=mask.sum(-1)-1
            full=language_model_kl(F.linear(h,w),F.linear(rh,rw),mask,pos)
            full_grad=torch.autograd.grad(full,(h,w))
            hs=h.detach().clone().requires_grad_()
            ws=w.detach().clone().requires_grad_()
            refh=rh.clone().requires_grad_()
            refw=rw.clone().requires_grad_()
            small=prefix_kl_from_hidden(hs,refh,ws,refw,mask,pos,chunk_tokens=chunk)
            small.backward()
            assert refh.grad is None and refw.grad is None
            rtol,atol=(0.03,0.01) if dtype==torch.bfloat16 else ((1e-10,1e-10) if dtype==torch.float64 else (2e-5,2e-6))
            assert torch.allclose(full,small,rtol=rtol,atol=atol)
            for expected,actual in zip(full_grad,(hs.grad,ws.grad)):
                assert torch.allclose(expected,actual,rtol=rtol,atol=atol)
            excluded=~mask.bool()
            excluded[torch.arange(3),pos]=True
            assert hs.grad[excluded].count_nonzero()==0
            results.append(dict(dtype=str(dtype),chunk=chunk,loss_abs_error=float((full-small).detach().abs()),
                hidden_grad_max_error=float((full_grad[0]-hs.grad).abs().max()),
                weight_grad_max_error=float((full_grad[1]-ws.grad).abs().max()),
                reference_grad_absent=True,candidate_and_padding_grad_zero=True,rtol=rtol,atol=atol))
    result=dict(status='cpu_synthetic_equivalence_passed',cases=results,
        limitation='Loss and gradient checks on synthetic hidden states only; real-model GPU integration, memory and speed remain untested. No training protocol or benchmark score changed.')
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
