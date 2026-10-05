"""Temporal and camera alignment for the optional warp-teacher RGB pairs."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from vera.datasets.base import DatasetConfig, UnifiedDataset
from vera.datasets.core.layout import apply_layout
from vera.datasets.core.sources import Episode, Source
from vera.datasets.core.view_loader import ViewLoader


class ToySource(Source):
    def __init__(self, data_root=".", *, name="toy", views=None):
        super().__init__(data_root)
        self.name = name
        self.views = list(views or ["front", "wrist"])

    def list_episodes(self):
        return [Episode("0", self.name, 5, self.views)]


class ToyViewLoader(ViewLoader):
    def __init__(self, height=2, per_view_w=3, **_kwargs):
        super().__init__(height, per_view_w)
        self.rgb_reads = []

    def load_rgb(self, episode, frame_indices):
        indices = np.asarray(frame_indices)
        assert np.all((indices >= 0) & (indices < episode.num_frames))
        self.rgb_reads.append(indices.copy())
        return [
            torch.as_tensor(indices, dtype=torch.float32)
            .add(view_idx * 10)
            .div(100)
            .view(-1, 1, 1, 1)
            .expand(-1, 3, self.height, self.per_view_w)
            .clone()
            for view_idx in range(len(episode.views))
        ]

    def load_trajectory(self, episode, key):
        assert key == "qpos"
        return np.square(np.arange(episode.num_frames, dtype=np.float64))[:, None]


def make_dataset(*, enabled=True, views=None, layout="separate", pad_views_to=None):
    source = ToySource(views=views)
    loader = ToyViewLoader()
    cfg = DatasetConfig(
        layout=layout,
        n_frames=3,
        height=2,
        per_view_w=3,
        derive_action=True,
        action_mode="qpos_delta",
        action_dim=1,
        linearize=2,
        overfit_idx=0,
        load_rgb_next=enabled,
        pad_views_to=pad_views_to,
        pad_position="left",
    )
    return UnifiedDataset(source, loader, cfg), loader


@pytest.mark.parametrize(
    "views,layout,pad_views_to",
    [
        (["front"], "separate", None),
        (["front", "wrist"], "separate", None),
        (["front", "wrist"], "separate", 3),
        (["front", "wrist"], "tiled", 3),
    ],
)
def test_rgb_next_matches_action_endpoint_and_camera_layout(views, layout, pad_views_to):
    dataset, loader = make_dataset(views=views, layout=layout, pad_views_to=pad_views_to)
    batch = dataset[0]
    np.testing.assert_array_equal(loader.rgb_reads[0], [0, 1, 2])
    np.testing.assert_array_equal(loader.rgb_reads[1], [2, 3, 4])
    # A nonconstant trajectory makes an off-by-one action target observable.
    torch.testing.assert_close(batch["du"], torch.tensor([[4.0], [8.0], [12.0]]))
    assert batch["rgb_next"].shape == batch["rgb"].shape
    for key, indices in [("rgb", [0, 1, 2]), ("rgb_next", [2, 3, 4])]:
        expected = apply_layout(
            ToyViewLoader().load_rgb(dataset._episodes[0], np.asarray(indices)),
            layout=layout,
            pad_views_to=pad_views_to,
            pad_position="left",
        )
        torch.testing.assert_close(batch[key], expected)


def test_rgb_next_default_preserves_batch_and_rgb_reads():
    dataset, loader = make_dataset(enabled=False)
    batch = dataset[0]
    assert set(batch) == {"rgb", "du"}
    assert len(loader.rgb_reads) == 1
    assert DatasetConfig("separate", 1, 2, 3).load_rgb_next is False


@pytest.mark.parametrize("enabled", [False, True])
def test_packed_factory_forwards_rgb_next_flag(monkeypatch, enabled):
    from vera.datasets import registry

    monkeypatch.setattr(registry, "PackedSource", ToySource)
    monkeypatch.setattr(registry, "PackedViewLoader", ToyViewLoader)
    cfg = {
        "name": "mimicgen",
        "action_mode": "qpos_delta",
        "camera": {"views": ["front", "wrist"], "image_size": [2, 3]},
        "sampling": {"num_frames": 3, "linearize_time_length": 2},
        "overfit_idx": 0,
    }
    if enabled:
        cfg["load_rgb_next"] = True
        cfg["load_flow"] = False
    dataset = registry.build_dataset(cfg, stage="training")
    assert dataset.cfg.load_rgb_next is enabled
    assert dataset.cfg.load_flow is (not enabled)
    assert ("rgb_next" in dataset[0]) is enabled


def test_time_aware_actions_cannot_silently_mismatch_rgb_next():
    dataset, loader = make_dataset()
    cfg = replace(dataset.cfg, use_time_aware_delta=True)
    with pytest.raises(ValueError, match="different target frame"):
        UnifiedDataset(dataset.source, loader, cfg)
    # The pre-existing action-only path keeps its previous behavior.
    UnifiedDataset(dataset.source, loader, replace(cfg, load_rgb_next=False))


def test_rgb_next_rejects_nonpositive_transition():
    dataset, loader = make_dataset()
    with pytest.raises(ValueError, match="linearize >= 1"):
        UnifiedDataset(dataset.source, loader, replace(dataset.cfg, linearize=0))
