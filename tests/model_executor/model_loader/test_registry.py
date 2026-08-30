# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import ModelConfig, get_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.expert_cache import (
    get_streamed_expert_cache_load_token,
)
from vllm.model_executor.model_loader import base_loader as base_loader_module
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_base_loader_sets_config_during_post_load(monkeypatch):
    load_config = LoadConfig(load_format="custom_load_format")
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        load_config=load_config,
        offload_config=SimpleNamespace(expert_cache_enabled=False),
    )
    model_config = SimpleNamespace(dtype=torch.float32)
    model = nn.Linear(1, 1)
    seen_config = None

    monkeypatch.setattr(
        base_loader_module,
        "initialize_model",
        lambda **kwargs: model,
    )
    monkeypatch.setattr(
        base_loader_module,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: False, is_xpu=lambda: False),
    )

    def record_post_load(*args):
        nonlocal seen_config
        seen_config = get_current_vllm_config()

    monkeypatch.setattr(
        base_loader_module,
        "process_weights_after_loading",
        record_post_load,
    )

    loader = CustomModelLoader(load_config)
    loader.load_model(vllm_config, model_config)  # type: ignore[arg-type]

    assert seen_config is vllm_config


def test_base_loader_scopes_each_expert_cache_load(monkeypatch):
    load_config = LoadConfig(load_format="custom_load_format")
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        load_config=load_config,
        offload_config=SimpleNamespace(expert_cache_enabled=True),
    )
    model_config = SimpleNamespace(dtype=torch.float32)
    phases: list[tuple[str, object]] = []

    def record_phase(phase: str) -> None:
        phases.append((phase, get_streamed_expert_cache_load_token()))

    def initialize_model(**kwargs):
        del kwargs
        record_phase("initialize")
        return nn.Linear(1, 1)

    def load_weights(model, model_config):
        del model, model_config
        record_phase("load")

    def process_weights_after_loading(*args):
        del args
        record_phase("postprocess")

    monkeypatch.setattr(base_loader_module, "initialize_model", initialize_model)
    monkeypatch.setattr(
        base_loader_module,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: False, is_xpu=lambda: False),
    )
    monkeypatch.setattr(
        base_loader_module,
        "process_weights_after_loading",
        process_weights_after_loading,
    )

    loader = CustomModelLoader(load_config)
    monkeypatch.setattr(loader, "load_weights", load_weights)
    loader.load_model(vllm_config, model_config)  # type: ignore[arg-type]
    loader.load_model(vllm_config, model_config)  # type: ignore[arg-type]

    assert [phase for phase, _ in phases] == [
        "initialize",
        "load",
        "postprocess",
    ] * 2
    first_load_tokens = {id(token) for _, token in phases[:3]}
    second_load_tokens = {id(token) for _, token in phases[3:]}
    assert len(first_load_tokens) == 1
    assert len(second_load_tokens) == 1
    assert first_load_tokens != second_load_tokens
    with pytest.raises(RuntimeError, match="model-load context"):
        get_streamed_expert_cache_load_token()


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)
