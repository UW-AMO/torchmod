"""
Test soft cone module class
"""
import pytest
import numpy as np
from torch import Tensor
from torchmod import Softcone


@pytest.mark.parametrize("d", [Tensor(np.ones(5))])
@pytest.mark.parametrize("lam", [4.0])
@pytest.mark.parametrize("x", [Tensor(np.ones(5))])
def test_case_zero(d, lam, x):
    mod = Softcone(d, lam)
    y = mod.forward(x)
    assert np.allclose(y, 0)


@pytest.mark.parametrize("d", [Tensor(np.ones(5))])
@pytest.mark.parametrize("lam", [1.0])
@pytest.mark.parametrize("x", [Tensor(np.ones(5))])
def test_case_scale(d, lam, x):
    mod = Softcone(d, lam)
    y = mod.forward(x)

    norm_y = np.linalg.norm(x) - lam
    assert np.allclose(y, x/(1.0 + lam/norm_y))
