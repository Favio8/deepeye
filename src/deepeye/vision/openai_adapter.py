"""OpenAI 兼容视觉适配器。

通过 OpenAI Chat Completions API（兼容任何 OpenAI 格式服务，例如 Azure
OpenAI、代理、第三方兼容服务）发送图文请求并返回文本描述。
"""

from __future__ import annotations

import httpx

from deepeye.config import settings
from deepeye.vision.base import VisionAdapter

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIVisionAdapter(VisionAdapter):
    """基于 OpenAI Chat Completions 的视觉适配器。

    Args:
        model: 模型名称，未指定时使用 ``settings.openai_model``。
        api_key: API Key，未指定时使用 ``settings.openai_api_key``。
        base_url: 接口地址，未指定时使用 ``settings.openai_base_url``，
            仍为空则回退到官方 ``https://api.openai.com/v1``。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.openai_model
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        base = base_url if base_url is not None else settings.openai_base_url
        self.base_url = base.strip() if base and base.strip() else _DEFAULT_BASE_URL

    async def describe(self, image_b64: str, mime_type: str, prompt: str) -> str:
        """调用 OpenAI Chat Completions 返回图像描述文本。

        Raises:
            httpx.HTTPStatusError: API 返回非 2xx 状态码时由
                ``raise_for_status()`` 抛出。
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        data_uri = f"data:{mime_type};base64,{image_b64}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
