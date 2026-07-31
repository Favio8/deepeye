"""DeepEye MCP 工具实现。

三个工具均返回 ``list[TextContent]``：
- :func:`describe_image`：图片详细描述
- :func:`extract_text`：OCR 文字提取
- :func:`ask_about_image`：视觉问答
"""

from __future__ import annotations

from mcp.types import TextContent

from deepeye.image_utils import parse_image_source
from deepeye.vision import create_vision_adapter

_DEFAULT_DESCRIBE_PROMPT = (
    "请详细描述这张图片的内容，包括主要对象、场景、动作、色彩和氛围。"
)
_OCR_PROMPT = "请仅提取并返回图片中的所有文字，不要添加任何额外描述。保持原文排版格式。"


async def _run_vision(image_source: str, prompt: str, model: str | None = None) -> str:
    """内部统一流程：解析图像源 → 创建适配器 → 调用 describe。

    Args:
        image_source: 图像来源（本地路径 / URL / data URI）。
        prompt: 提示词。
        model: 可选模型名称覆盖。

    Returns:
        视觉模型返回的文本。
    """
    b64_data, mime_type = await parse_image_source(image_source)
    adapter = create_vision_adapter(model)
    return await adapter.describe(b64_data, mime_type, prompt)


async def describe_image(
    image_source: str,
    prompt: str = _DEFAULT_DESCRIBE_PROMPT,
    model: str | None = None,
) -> list[TextContent]:
    """描述图片内容。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        prompt: 描述提示词，默认为详细描述。
        model: 可选模型名称覆盖。

    Returns:
        包含图片描述文本的 ``list[TextContent]``。
    """
    description = await _run_vision(image_source, prompt, model)
    return [TextContent(type="text", text=f"图片分析结果：\n{description}")]


async def extract_text(
    image_source: str,
    language: str = "auto",
) -> list[TextContent]:
    """提取图片中的文字（OCR）。

    使用固定 OCR 提示词，强制仅返回文字、保持排版、不加描述。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        language: 识别语言，``"auto"`` 自动识别；其他值会附加语言提示。

    Returns:
        包含提取文字的 ``list[TextContent]``。
    """
    prompt = _OCR_PROMPT
    if language != "auto":
        prompt += f" 优先识别语言：{language}"
    text = await _run_vision(image_source, prompt)
    return [TextContent(type="text", text=text)]


async def ask_about_image(
    image_source: str,
    question: str,
) -> list[TextContent]:
    """根据图片内容回答问题。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        question: 要回答的问题。

    Returns:
        包含答案的 ``list[TextContent]``。
    """
    prompt = f"请根据图片内容回答以下问题：{question}"
    answer = await _run_vision(image_source, prompt)
    return [TextContent(type="text", text=answer)]
