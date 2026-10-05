"""Disabling W&B must preserve validation losses without generating media."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("lightning")
from vera.idm.jacobian.image_jacobian import ImageJacobian


def test_validation_without_logger_computes_loss_and_skips_media():
    total = torch.tensor(0.5)
    losses, metrics, diagnostics = {"flow": total}, {}, {"warp_valid_ratio": torch.tensor(1.)}
    algorithm = SimpleNamespace(
        logger=None,
        trainer=SimpleNamespace(is_global_zero=True),
        cfg=SimpleNamespace(logging=SimpleNamespace(max_validation_batches=1, max_num_videos=24)),
        _compute_loss=Mock(return_value=(losses, metrics, total, None, diagnostics, {})),
        _log_losses=Mock(),
    )
    # Media generation would need RGB tensors; scalar validation needs only the
    # loss computation contract, and must finish even without a media logger.
    batch = {}
    ImageJacobian.validation_step(algorithm, batch, 0, namespace="validation/test")
    algorithm._compute_loss.assert_called_once_with(batch, namespace="validation/test", collect_diagnostics=True)
    algorithm._log_losses.assert_called_once_with(losses, metrics, total, diagnostics, namespace="validation/test")
