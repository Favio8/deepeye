"""``deepeye.tools`` 三个 MCP 工具函数单元测试。

通过 ``unittest.mock.patch`` 替换 ``deepeye.tools.create_vision_adapter``，
避免任何真实 API 调用；使用 data URI 作为图像源，避免本地文件/网络 IO。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import TextContent

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
    assert call.args[2] == "请根据图片内容回答以下问题：图中有几只猫？"

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "mocked description"
