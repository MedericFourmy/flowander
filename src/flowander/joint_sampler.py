# inspired by https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

import ot
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Union
from functools import partial


def compute_pi_condot(a, b, C):
    """
    https://arxiv.org/abs/2304.14772
    'CondOT is Uniform Coupling'
    """
    k = C.shape[0]
    return np.ones((k, k)) / k


class JointSampler:

    """
    Sample from joint multisample distribution q_k(x0, x1)
    """

    def __init__(
            self, 
            method: str,
            bs: Union[int,None] = None,
            reg: float = 0.05,
            reg_m: float = 1.0,
            normalize_cost: bool = False,
            num_threads: Union[int, str] = 1,
            ):
        if method == "linear_sum_assignment": 
            self.compute_pi = None
        elif method == "condot":
            self.compute_pi = compute_pi_condot 
        elif method == "exact_emd":
            self.compute_pi = partial(ot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.compute_pi = partial(ot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.compute_pi = partial(ot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.compute_pi = partial(ot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.bs = bs
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        
    def _get_map(self, C: np.ndarray):
        n, m = C.shape
        assert n == m
        a, b = ot.unif(n), ot.unif(m)

        if self.normalize_cost:
            C = C / C.max()  # should not be normalized when using minibatches
        pi = self.compute_pi(a, b, C)
        return pi
        
    def _sample_map(self, pi):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        bs = pi.shape[0]
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(bs**2, size=bs, p=p, replace=False)
        return np.divmod(choices, bs)
    
    def sample_plan(self, C):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target as indices of the source/target batches.$

        Parameters
        ----------
        C : Cost matrix of the batch

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        if self.method == "linear_sum_assignment":
            rows, cols = linear_sum_assignment(C)
        else:
            pi = self._get_map(C)
            rows, cols = self._sample_map(pi)
        return rows, cols