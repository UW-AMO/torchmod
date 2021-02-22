"""
Sparse Operators
"""
# pylint:disable=no-name-in-module
from scipy.optimize import bisect
from torch import Tensor, zeros
from torch.nn import Module


class Softcone(Module):
    """
    Add description of the operator computation.
    """

    def __init__(self, d: Tensor, lam: float = 0.5):
        super().__init__()
        self.d = d
        self.lam = lam

    def forward(self, x: Tensor) -> Tensor:
        def root_fun(alpha: float) -> float:
            return sum((self.d*x)**2/(alpha*self.d + self.lam)**2) - 1.0
        alpha = 0.0
        y = zeros(len(x))
        if root_fun(0.0) > 0.0:
            alpha = bisect(root_fun, 0.0, len(x)*max(abs(x)))
            if alpha > 0.0:
                y = (self.d*x)/(self.d + self.lam/alpha)
        return y
