"""Keep base proposal/KV generation separate from LoRA candidate scoring."""
import torch
from evaluate_s2t import S2TDecoder


class SplitAdapterModel(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.scoring=False
        assert hasattr(model,'disable_adapter')
    @property
    def device(self):return self.model.device
    def forward(self,*args,**kwargs):
        if self.scoring:return self.model(*args,**kwargs)
        with self.model.disable_adapter():return self.model(*args,**kwargs)


class SplitAdapterDecoder(S2TDecoder):
    def __init__(self,config,bundle):
        bundle.slm=SplitAdapterModel(bundle.slm).eval()
        super().__init__(config,bundle)
    def score_s2t_local_candidates(self,*args,**kwargs):
        assert not self.slm.scoring
        self.slm.scoring=True
        try:return super().score_s2t_local_candidates(*args,**kwargs)
        finally:self.slm.scoring=False
