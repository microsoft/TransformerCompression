# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import pickle
from typing import Any

import pytest
import torch

from slicegpt import hf_utils


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state_dict: dict[str, torch.Tensor] | None = None
        self.eval_called = False

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.loaded_state_dict = state_dict

    def eval(self) -> None:
        self.eval_called = True


class _FakeModelAdapter:
    def __init__(self) -> None:
        self.hidden_size = 2
        self.model = _FakeModel()
        self.parallel_blocks = True
        self.slicing_conf = hf_utils.SlicingConfig()

    def get_layers(self) -> list[Any]:
        return []


@pytest.fixture(autouse=True)
def _restore_torch_initializers():
    initializers = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )
    yield
    (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    ) = initializers


@pytest.fixture
def mocked_model_loading(monkeypatch: pytest.MonkeyPatch) -> _FakeModelAdapter:
    model_adapter = _FakeModelAdapter()

    monkeypatch.setattr(hf_utils, "get_model_and_tokenizer", lambda *args, **kwargs: (model_adapter, object()))
    monkeypatch.setattr(hf_utils, "replace_layers", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_utils, "fuse_modules", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_utils, "slice_rotated_model", lambda *args, **kwargs: None)

    return model_adapter


def test_load_sliced_model_loads_tensor_state_dict_with_restricted_cpu_torch_load(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_model_loading: _FakeModelAdapter,
) -> None:
    state_dict = {"weight": torch.ones(1)}
    torch.save(state_dict, tmp_path / "tiny_0.5.pt")

    load_kwargs: dict[str, Any] = {}
    original_torch_load = torch.load

    def recording_torch_load(*args, **kwargs):
        load_kwargs.update(kwargs)
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(hf_utils.torch, "load", recording_torch_load)

    hf_utils.load_sliced_model("tiny", str(tmp_path), sparsity=0.5)

    assert load_kwargs.get("map_location") == "cpu"
    assert load_kwargs.get("weights_only") is True
    assert mocked_model_loading.model.loaded_state_dict is not None
    assert torch.equal(mocked_model_loading.model.loaded_state_dict["weight"], state_dict["weight"])
    assert mocked_model_loading.model.eval_called


def _write_sentinel(path: str) -> None:
    pathlib.Path(path).write_text("executed")


class _MaliciousReducer:
    def __init__(self, sentinel_path: pathlib.Path) -> None:
        self.sentinel_path = sentinel_path

    def __reduce__(self):
        return _write_sentinel, (str(self.sentinel_path),)


def test_load_sliced_model_rejects_pickle_payload_without_executing_or_loading_state(
    tmp_path: pathlib.Path,
    mocked_model_loading: _FakeModelAdapter,
) -> None:
    sentinel_path = tmp_path / "sentinel"
    torch.save({"payload": _MaliciousReducer(sentinel_path)}, tmp_path / "tiny_0.5.pt")
    initializers = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        hf_utils.load_sliced_model("tiny", str(tmp_path), sparsity=0.5)

    assert not sentinel_path.exists()
    assert mocked_model_loading.model.loaded_state_dict is None
    assert (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    ) == initializers
