"""DeepEye MCP 工具实现。

三个工具均返回 ``list[TextContent]``：
- :func:`describe_image`：图片详细描述
- :func:`extract_text`：OCR 文字提取
- :func:`ask_about_image`：视觉问答
"""

from __future__ import annotations

import hashlib

from mcp.types import TextContent

from deepeye.cache import vision_cache
from deepeye.config import settings
from deepeye.image_utils import parse_image_source, preprocess_image
from deepeye.vision import create_vision_adapter

_DEFAULT_DESCRIBE_PROMPT = (
    "请详细描述这张图片的内容，包括主要对象、场景、动作、色彩和氛围。"
)
_OCR_PROMPT = "请仅提取并返回图片中的所有文字，不要添加任何额外描述。保持原文排版格式。"


async def _run_vision(image_source: str, prompt: str, model: str | None = None) -> str:
    """内部统一流程：解析图像源 → 预处理 → 查缓存 → 调用适配器 → 写缓存。

    Args:
        image_source: 图像来源（本地路径 / URL / data URI）。
        prompt: 提示词。
        model: 可选模型名称覆盖。

    Returns:
        视觉模型返回的文本。
    """
    b64_data, mime_type = await parse_image_source(image_source)
    # 图片预处理（缩放 + 转 JPEG），失败时原样返回不阻断
    b64_data, mime_type = preprocess_image(b64_data, mime_type)

    # 用处理后的 b64 计算哈希，作为缓存 key 的一部分
    image_hash = hashlib.sha256(b64_data.encode()).hexdigest()
    effective_model = model if model is not None else ""

    # 开启缓存时先查缓存，命中则直接返回
    if settings.cache_enabled:
        cached = vision_cache.get(image_hash, prompt, effective_model)
        if cached is not None:
            return cached

    adapter = create_vision_adapter(model)
    text = await adapter.describe(b64_data, mime_type, prompt)

    # 写入缓存供下次复用
    if settings.cache_enabled:
        vision_cache.set(image_hash, prompt, effective_model, text)

    return text


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
