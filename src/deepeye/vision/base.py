"""视觉适配器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VisionAdapter(ABC):
    """视觉理解适配器抽象基类。

    具体实现通过 OpenAI / Gemini / 自定义兼容服务等后端完成图像理解，
    上层工具函数仅依赖此接口，实现后端可插拔。
    """

    @abstractmethod
    async def describe(self, image_b64: str, mime_type: str, prompt: str) -> str:
        """根据 ``prompt`` 对给定图像进行视觉理解并返回文本结果。

        Args:
            image_b64: 图像的 base64 编码字符串（不含 data URI 前缀）。
            mime_type: 图像 MIME 类型，例如 ``image/png``。
            prompt: 描述 / 提问 / OCR 等场景的提示词。

        Returns:
            视觉模型返回的文本结果。
        """
        ...
