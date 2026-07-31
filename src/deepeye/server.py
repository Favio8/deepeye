"""DeepEye MCP Server 组装。

通过 ``mcp`` 官方库的 ``stdio_server`` 启动 MCP Server，注册
``list_tools`` 与 ``call_tool`` 处理器，Server 名称为 ``deepeye``。

注意：本实现适配 ``mcp>=2.0.0`` 的 API：使用构造器注册
``on_list_tools`` / ``on_call_tool`` 处理器（而非旧版装饰器），handler
返回 ``ListToolsResult`` / ``CallToolResult``。
"""

from __future__ import annotations

import asyncio

from loguru import logger

from mcp.server import NotificationOptions, Server
from mcp.server.context import ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from deepeye import __version__
from deepeye.tools import ask_about_image, describe_image, extract_text

_TOOLS: list[Tool] = [
    Tool(
        name="describe_image",
        description="描述图片内容，支持本地路径 / 公网 URL / Base64 data URI 三种来源。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图像来源：本地路径、http(s) URL 或 data:image/...;base64,... 形式的 data URI。",
                },
                "prompt": {
                    "type": "string",
                    "description": "描述提示词，可选；不传则使用默认详细描述提示词。",
                },
                "model": {
                    "type": "string",
                    "description": "可选模型名称覆盖，不传则使用配置默认模型。",
                },
            },
            "required": ["image_source"],
        },
    ),
    Tool(
        name="extract_text",
        description="提取图片中的文字（OCR），保持原文排版，不加额外描述。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图像来源：本地路径、http(s) URL 或 data:image/...;base64,... 形式的 data URI。",
                },
                "language": {
                    "type": "string",
                    "description": "识别语言，默认 auto 自动识别；其他值会附加语言提示。",
                },
            },
            "required": ["image_source"],
        },
    ),
    Tool(
        name="ask_about_image",
        description="根据图片内容回答问题（视觉问答）。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图像来源：本地路径、http(s) URL 或 data:image/...;base64,... 形式的 data URI。",
                },
                "question": {
                    "type": "string",
                    "description": "要回答的问题。",
                },
            },
            "required": ["image_source", "question"],
        },
    ),
]


async def list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    """返回 DeepEye 暴露的三个工具定义。"""
    return ListToolsResult(tools=_TOOLS)


async def call_tool(
    ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    """分发工具调用到对应实现。

    Raises:
        ValueError: 未知工具名时抛出。
    """
    name = params.name
    arguments: dict = params.arguments or {}
    logger.debug("call_tool name={} arguments={}", name, arguments)

    if name == "describe_image":
        describe_kwargs: dict = {"image_source": arguments["image_source"]}
        if arguments.get("prompt"):
            describe_kwargs["prompt"] = arguments["prompt"]
        if arguments.get("model"):
            describe_kwargs["model"] = arguments["model"]
        content: list[TextContent] = await describe_image(**describe_kwargs)
    elif name == "extract_text":
        content = await extract_text(
            image_source=arguments["image_source"],
            language=arguments.get("language", "auto"),
        )
    elif name == "ask_about_image":
        content = await ask_about_image(
            image_source=arguments["image_source"],
            question=arguments["question"],
        )
    else:
        raise ValueError(f"未知工具: {name}")

    return CallToolResult(content=content)


server = Server(
    "deepeye",
    version=__version__,
    on_list_tools=list_tools,
    on_call_tool=call_tool,
)


async def main() -> None:
    """启动 stdio MCP Server。"""
    logger.info("启动 DeepEye MCP Server v{}", __version__)
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions()
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    asyncio.run(main())
