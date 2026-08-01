"""pytest 全局配置：禁用缓存避免测试间干扰。"""

import pytest

from deepeye.cache import vision_cache
from deepeye.config import settings


@pytest.fixture(autouse=True)
def _disable_cache():
    """每个测试前禁用缓存并清空，避免测试间互相干扰。"""
    settings.cache_enabled = False
    vision_cache.clear()
    yield
    vision_cache.clear()
