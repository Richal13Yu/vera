"""Regression for mixed-precision validation producing BF16 Jacobian fields."""

import numpy as np
import pytest
import torch

from vera.utils.jacobian_utils import visualize_jacobian


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_validation_jacobian_renders_to_uint8(dtype):
    jacobian = torch.linspace(-1, 1, 7 * 2 * 16 * 16).reshape(1, 7, 2, 16, 16)
    jacobian = jacobian.to(dtype).requires_grad_(True)
    image = visualize_jacobian(jacobian, "eef_gripper", target_hw=(32, 32))
    assert image.shape == (3, 32, 32)
    assert image.dtype == np.uint8
    assert image.max() > image.min()
    assert jacobian.dtype == dtype
    assert jacobian.grad is None
