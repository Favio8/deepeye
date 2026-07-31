"""``deepeye.image_utils`` 单元测试。

覆盖三种图像来源（本地文件 / 公网 URL / Base64 data URI）的解析逻辑，
URL 下载通过 mock ``httpx.AsyncClient`` 避免真实网络 IO。
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepeye.image_utils import (
    _parse_data_uri,
    load_image_as_base64,
    load_image_from_url_as_base64,
    parse_image_source,
)


def _write_png(path: Path) -> bytes:
    """用 Pillow 生成一张最小 PNG 写入 ``path``，返回图片原始 bytes。"""
    from PIL import Image

    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    img.save(path, format="PNG")
    return path.read_bytes()


# ---------------------------------------------------------------------------
# load_image_as_base64
# ---------------------------------------------------------------------------


def test_load_image_as_base64_local_png(tmp_path: Path):
    img_path = tmp_path / "test.png"
    raw = _write_png(img_path)
    expected_b64 = base64.b64encode(raw).decode("ascii")

    b64_data, mime_type = load_image_as_base64(str(img_path))

    assert b64_data == expected_b64
    assert mime_type == "image/png"


def test_load_image_as_base64_file_not_found(tmp_path: Path):
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError):
        load_image_as_base64(str(missing))


def test_load_image_as_base64_unknown_extension(tmp_path: Path, monkeypatch):
    """mimetypes 推断失败（返回 None）时应回退到默认 ``image/png``。

    使用 monkeypatch 强制 ``mimetypes.guess_type`` 返回 ``None``，避免
    不同操作系统下扩展名映射差异（如 Windows 上 ``.bin`` 会返回
    ``application/octet-stream``）造成测试不稳定。
    """
    import mimetypes

    monkeypatch.setattr(mimetypes, "guess_type", lambda path: (None, None))

    img_path = tmp_path / "weird.xyz"
    raw = b"\x89PNG\r\n\x1a\nfake-bytes"
    img_path.write_bytes(raw)
    expected_b64 = base64.b64encode(raw).decode("ascii")

    b64_data, mime_type = load_image_as_base64(str(img_path))

    assert b64_data == expected_b64
    assert mime_type == "image/png"


# ---------------------------------------------------------------------------
# _parse_data_uri
# ---------------------------------------------------------------------------


def test_parse_data_uri_jpeg():
    b64_data, mime_type = _parse_data_uri("data:image/jpeg;base64,/9j/4AAQ")
    assert b64_data == "/9j/4AAQ"
    assert mime_type == "image/jpeg"


def test_parse_data_uri_png():
    b64_data, mime_type = _parse_data_uri("data:image/png;base64,iVBOR")
    assert b64_data == "iVBOR"
    assert mime_type == "image/png"


def test_parse_data_uri_empty_data_raises():
    with pytest.raises(ValueError):
        _parse_data_uri("data:image/png;base64,")


# ---------------------------------------------------------------------------
# load_image_from_url_as_base64（mock httpx）
# ---------------------------------------------------------------------------


def _make_fake_client(content: bytes, content_type: str) -> AsyncMock:
    """构造一个 fake ``httpx.AsyncClient``，返回指定响应内容。"""
    fake_response = MagicMock()
    fake_response.content = content
    fake_response.headers = {"Content-Type": content_type} if content_type else {}
    fake_response.raise_for_status = MagicMock()

    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = None
    return fake_client


@patch("deepeye.image_utils.httpx.AsyncClient")
async def test_load_image_from_url_as_base64(mock_client_cls):
    raw = b"fake-image-bytes"
    mock_client_cls.return_value = _make_fake_client(raw, "image/jpeg; charset=utf-8")

    b64_data, mime_type = await load_image_from_url_as_base64(
        "https://example.com/cat.jpg"
    )

    assert b64_data == base64.b64encode(raw).decode("ascii")
    assert mime_type == "image/jpeg"
    mock_client_cls.return_value.get.assert_awaited_once_with(
        "https://example.com/cat.jpg"
    )


@patch("deepeye.image_utils.httpx.AsyncClient")
async def test_load_image_from_url_as_base64_missing_content_type(mock_client_cls):
    raw = b"more-bytes"
    mock_client_cls.return_value = _make_fake_client(raw, "")

    b64_data, mime_type = await load_image_from_url_as_base64(
        "https://example.com/img"
    )

    assert b64_data == base64.b64encode(raw).decode("ascii")
    assert mime_type == "image/png"


@patch("deepeye.image_utils.httpx.AsyncClient")
async def test_load_image_from_url_as_base64_raises_on_error_status(mock_client_cls):
    import httpx

    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=MagicMock(),
        response=fake_response,
    )
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = None
    mock_client_cls.return_value = fake_client

    with pytest.raises(httpx.HTTPStatusError):
        await load_image_from_url_as_base64("https://example.com/500.png")


# ---------------------------------------------------------------------------
# parse_image_source（统一入口）
# ---------------------------------------------------------------------------


async def test_parse_image_source_data_uri_no_io():
    """``data:`` 前缀走 _parse_data_uri，不进行任何文件/网络 IO。"""
    b64_data, mime_type = await parse_image_source("data:image/png;base64,ABC")
    assert b64_data == "ABC"
    assert mime_type == "image/png"


async def test_parse_image_source_local(tmp_path: Path):
    img_path = tmp_path / "local.png"
    raw = _write_png(img_path)

    b64_data, mime_type = await parse_image_source(str(img_path))

    assert b64_data == base64.b64encode(raw).decode("ascii")
    assert mime_type == "image/png"


@patch("deepeye.image_utils.httpx.AsyncClient")
async def test_parse_image_source_url(mock_client_cls):
    raw = b"webp-bytes"
    mock_client_cls.return_value = _make_fake_client(raw, "image/webp")

    b64_data, mime_type = await parse_image_source("https://example.com/img.webp")

    assert b64_data == base64.b64encode(raw).decode("ascii")
    assert mime_type == "image/webp"
