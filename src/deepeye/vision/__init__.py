"""视觉适配器包。

提供抽象基类 :class:`VisionAdapter` 与工厂函数
:func:`create_vision_adapter`，根据配置创建具体适配器实例。
"""

from __future__ import annotations

from deepeye.config import settings
from deepeye.vision.base import VisionAdapter
from deepeye.vision.openai_adapter import OpenAIVisionAdapter

__all__ = ["VisionAdapter", "OpenAIVisionAdapter", "create_vision_adapter"]


def create_vision_adapter(model_override: str | None = None) -> VisionAdapter:
    """根据 ``settings.vision_provider`` 创建对应视觉适配器实例。

    Args:
        model_override: 可选的模型名称覆盖；未指定时使用配置默认值。

    Returns:
        :class:`VisionAdapter` 具体实例。

    Raises:
        NotImplementedError: 暂不支持的 provider。
    """
    provider = settings.vision_provider.lower().strip()
    if provider == "openai":
        return OpenAIVisionAdapter(model=model_override)
    # TODO: 实现 gemini / custom 适配器
    raise NotImplementedError(f"暂不支持的视觉后端: {settings.vision_provider}")
