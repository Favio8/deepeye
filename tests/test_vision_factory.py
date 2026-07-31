"""``deepeye.vision`` 工厂函数 ``create_vision_adapter`` 单元测试。"""

from __future__ import annotations

import pytest

from deepeye.config import settings
from deepeye.vision import create_vision_adapter
from deepeye.vision.openai_adapter import OpenAIVisionAdapter


def test_create_vision_adapter_openai_default(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "openai")
    adapter = create_vision_adapter()
    assert isinstance(adapter, OpenAIVisionAdapter)


def test_create_vision_adapter_openai_model_override(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "openai")
    adapter = create_vision_adapter(model_override="gpt-4o-mini")
    assert isinstance(adapter, OpenAIVisionAdapter)
    assert adapter.model == "gpt-4o-mini"


def test_create_vision_adapter_openai_model_none_uses_config(monkeypatch):
    """未指定 model_override 时，应回退到 ``settings.openai_model``。"""
    monkeypatch.setattr(settings, "vision_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o")
    adapter = create_vision_adapter()
    assert isinstance(adapter, OpenAIVisionAdapter)
    assert adapter.model == "gpt-4o"


def test_create_vision_adapter_not_implemented(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "gemini")
    with pytest.raises(NotImplementedError):
        create_vision_adapter()


def test_create_vision_adapter_provider_case_insensitive(monkeypatch):
    """provider 大小写应被规范化（lower()）。"""
    monkeypatch.setattr(settings, "vision_provider", "OpenAI")
    adapter = create_vision_adapter()
    assert isinstance(adapter, OpenAIVisionAdapter)
