"""``deepeye.tools`` 三个 MCP 工具函数单元测试。

通过 ``unittest.mock.patch`` 替换 ``deepeye.tools.create_vision_adapter``，
避免任何真实 API 调用；使用 data URI 作为图像源，避免本地文件/网络 IO。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import TextContent

from deepeye.cache import vision_cache
from deepeye.config import settings
from deepeye.tools import (
    _DEFAULT_DESCRIBE_PROMPT,
    _OCR_PROMPT,
    ask_about_image,
    describe_image,
    extract_text,
)

_DATA_URI = "data:image/png;base64,iVBOR"


def _build_mock_adapter(return_value: str = "mocked description") -> MagicMock:
    """构造一个 mock 适配器，其 ``describe`` 是 AsyncMock 返回固定字符串。"""
    adapter = MagicMock()
    adapter.describe = AsyncMock(return_value=return_value)
    return adapter


# ---------------------------------------------------------------------------
# describe_image
# ---------------------------------------------------------------------------


@patch("deepeye.tools.create_vision_adapter")
async def test_describe_image_default_prompt(mock_factory):
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    result = await describe_image(image_source=_DATA_URI)

    mock_adapter.describe.assert_awaited_once()
    call = mock_adapter.describe.await_args
    # describe(b64_data, mime_type, prompt)
    assert call.args[0] == "iVBOR"
    assert call.args[1] == "image/png"
    assert call.args[2] == _DEFAULT_DESCRIBE_PROMPT

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "mocked description" in result[0].text


@patch("deepeye.tools.create_vision_adapter")
async def test_describe_image_custom_prompt(mock_factory):
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    await describe_image(
        image_source=_DATA_URI,
        prompt="描述图表数据趋势",
    )

    call = mock_adapter.describe.await_args
    assert call.args[2] == "描述图表数据趋势"


@patch("deepeye.tools.create_vision_adapter")
async def test_describe_image_custom_model(mock_factory):
    """model 参数应透传给工厂函数。"""
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    await describe_image(image_source=_DATA_URI, model="gpt-4o-mini")

    mock_factory.assert_called_once_with("gpt-4o-mini")


@patch("deepeye.tools.create_vision_adapter")
async def test_describe_image_result_format(mock_factory):
    mock_adapter = _build_mock_adapter(return_value="hello world")
    mock_factory.return_value = mock_adapter

    result = await describe_image(image_source=_DATA_URI)

    assert result[0].text == "图片分析结果：\nhello world"


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


@patch("deepeye.tools.create_vision_adapter")
async def test_extract_text_default_no_language_hint(mock_factory):
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    result = await extract_text(image_source=_DATA_URI)

    call = mock_adapter.describe.await_args
    assert call.args[2] == _OCR_PROMPT  # 不附加语言提示

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "mocked description"


@patch("deepeye.tools.create_vision_adapter")
async def test_extract_text_with_language_zh(mock_factory):
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    await extract_text(image_source=_DATA_URI, language="zh")

    call = mock_adapter.describe.await_args
    assert call.args[2] == _OCR_PROMPT + " 优先识别语言：zh"


@patch("deepeye.tools.create_vision_adapter")
async def test_extract_text_with_language_en(mock_factory):
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    await extract_text(image_source=_DATA_URI, language="en")

    call = mock_adapter.describe.await_args
    assert call.args[2] == _OCR_PROMPT + " 优先识别语言：en"


# ---------------------------------------------------------------------------
# ask_about_image
# ---------------------------------------------------------------------------


@patch("deepeye.tools.create_vision_adapter")
async def test_ask_about_image_prompt_assembly(mock_factory):
    mock_adapter = _build_mock_adapter()
    mock_factory.return_value = mock_adapter

    result = await ask_about_image(
        image_source=_DATA_URI,
        question="图中有几只猫？",
    )

    call = mock_adapter.describe.await_args
    assert call.args[2] == "根据图片回答：图中有几只猫？"

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "mocked description"


# ---------------------------------------------------------------------------
# 缓存集成（cache_enabled=True 时第二次相同调用不触发适配器）
# ---------------------------------------------------------------------------


@patch("deepeye.tools.create_vision_adapter")
async def test_cache_enabled_second_call_hits_cache(mock_factory, monkeypatch):
    """开启缓存后，第二次相同调用应命中缓存，不再次调用适配器。"""
    monkeypatch.setattr(settings, "cache_enabled", True)
    vision_cache.clear()  # 确保起始状态干净

    mock_adapter = _build_mock_adapter(return_value="first-result")
    mock_factory.return_value = mock_adapter

    # 第一次调用：未命中缓存，调用适配器
    result1 = await describe_image(image_source=_DATA_URI)
    assert mock_adapter.describe.await_count == 1
    assert "first-result" in result1[0].text

    # 第二次相同调用：应命中缓存，适配器不再被调用
    result2 = await describe_image(image_source=_DATA_URI)
    assert mock_adapter.describe.await_count == 1  # 仍为 1，未新增调用
    assert "first-result" in result2[0].text

    # 清理：恢复缓存状态
    vision_cache.clear()


@patch("deepeye.tools.create_vision_adapter")
async def test_cache_disabled_calls_adapter_each_time(mock_factory, monkeypatch):
    """关闭缓存时，每次调用都应触发适配器。"""
    monkeypatch.setattr(settings, "cache_enabled", False)
    vision_cache.clear()

    mock_adapter = _build_mock_adapter(return_value="result")
    mock_factory.return_value = mock_adapter

    await describe_image(image_source=_DATA_URI)
    await describe_image(image_source=_DATA_URI)

    assert mock_adapter.describe.await_count == 2

    vision_cache.clear()


@patch("deepeye.tools.create_vision_adapter")
async def test_cache_different_prompt_does_not_hit(mock_factory, monkeypatch):
    """相同图片但不同 prompt 应产生不同 key，缓存不命中。"""
    monkeypatch.setattr(settings, "cache_enabled", True)
    vision_cache.clear()

    mock_adapter = _build_mock_adapter(return_value="cached-desc")
    mock_factory.return_value = mock_adapter

    await describe_image(image_source=_DATA_URI, prompt="描述A")
    await describe_image(image_source=_DATA_URI, prompt="描述B")

    # 两次不同 prompt，应调用适配器两次
    assert mock_adapter.describe.await_count == 2

    vision_cache.clear()


@patch("deepeye.tools.create_vision_adapter")
async def test_cache_returns_cached_text_directly(mock_factory, monkeypatch):
    """缓存命中时应直接返回缓存文本（未经适配器重新生成）。"""
    monkeypatch.setattr(settings, "cache_enabled", True)
    vision_cache.clear()

    # 第一次调用返回 "v1"
    mock_adapter = _build_mock_adapter(return_value="v1")
    mock_factory.return_value = mock_adapter
    await describe_image(image_source=_DATA_URI)

    # 切换 mock 返回 "v2"，但缓存命中时不应调用，结果仍是 "v1"
    mock_adapter2 = _build_mock_adapter(return_value="v2")
    mock_factory.return_value = mock_adapter2

    result = await describe_image(image_source=_DATA_URI)
    assert "v1" in result[0].text
    assert mock_adapter2.describe.await_count == 0

    vision_cache.clear()
