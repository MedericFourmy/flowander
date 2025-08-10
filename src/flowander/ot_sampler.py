# inspired by https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

import ot
from typing import Union
from functools import partial


bs = 50
x0 = self.p_simple.sample(z.shape[0])
dim = x0.shape[-1]

assert z.shape[0] % bs == 0
nb_batches = z.shape[0] // bs
zz = z.reshape(nb_batches, bs, dim)
xx0 = x0.reshape(nb_batches, bs, dim)
Cc = torch.cdist(xx0, zz)

class JointSampler:

    """
    Sample from joint multisample distribution q_k(x0, x1)
    """

    def __init__(
            self, 
            method: str,
            bs_ot=-1,
            reg: float = 0.05,
            reg_m: float = 1.0,
            normalize_cost: bool = False,
            num_threads: Union[int, str] = 1,
            ):
        if method == "exact":
            self.ot_fn = partial(ot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(ot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(ot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(ot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        
    def sample(x1,x2,C):
        a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
        G = ot.emd(a, b, C)
        
