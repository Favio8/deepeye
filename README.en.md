# DeepEye

> Give text-only LLMs a pair of eyes.

English | [简体中文](README.md)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-2.0+-purple.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](#changelog)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

DeepEye is an open-source **MCP (Model Context Protocol) Server** that brings vision capabilities to any MCP-compatible text-only large language model (e.g., DeepSeek V3/V4, Qwen text-only variants, Llama, etc.). The model itself never sees the image — DeepEye "sees" on its behalf and feeds the visual understanding back as text.

Deploy once, reuse from any MCP client (Claude Desktop, Cline, Cursor, custom agents, ...).

---

## Table of Contents

- [Why DeepEye](#why-deepeye)
- [Features](#features)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Tools](#tools)
- [Usage Examples](#usage-examples)
- [Configuration Reference](#configuration-reference)
- [MCP Client Integration](#mcp-client-integration)
- [Supported Vision Backends](#supported-vision-backends)
- [Development](#development)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why DeepEye

Powerful text-only LLMs like DeepSeek V4 Flash are exceptional reasoners — but they are born "blind": they cannot process images directly. Yet real-world tasks constantly call for vision: chart analysis, screenshot diagnostics, document OCR, UI reviews, and more.

The conventional answer is to switch to a multimodal model, but that means:

- Giving up the reasoning style and context window of your favorite text-only model
- Being locked into a specific multimodal vendor
- Every agent re-implementing vision support, reinventing the wheel

DeepEye uses the **MCP protocol** to decouple "visual understanding" from the model itself, exposing it as an independent tool layer:

- Keep using your favorite text-only model as the main reasoner
- When the model needs to see, it calls a DeepEye tool and receives a textual description
- The vision backend is pluggable — switch freely between OpenAI / Gemini / any OpenAI-compatible service
- Build once, use from every MCP client

> In one sentence: **a guide dog for "blind" models.**

---

## Features

- **Three core tools**: `describe_image` (caption), `extract_text` (OCR), `ask_about_image` (VQA)
- **Standard MCP protocol**: built on the official `mcp` library, stdio transport, compatible with all MCP clients
- **Pluggable vision backends**: strategy + adapter pattern; OpenAI implemented, Gemini / custom OpenAI-compatible extension points reserved
- **Three image sources**: local path / public URL / Base64 data URI, unified parsing
- **Minimal deployment**: clone → install → fill in an API key → run; no account, no sign-up
- **Zero intrusion**: doesn't modify the model itself; a transparent tool layer on top of the main reasoning loop
- **Open source**: MIT license, community-driven

---

## How It Works

```mermaid
graph LR
    U[User] -->|question + image| A[MCP Client<br/>DeepSeek / Claude / etc.]
    A -->|model decides to call tool| B[DeepEye MCP Server]
    B -->|parse image source| C[Local / URL / Base64]
    B -->|assemble prompt + image| D[Vision Backend Adapter]
    D --> E1[GPT-4o]
    D --> E2[Gemini]
    D --> E3[Any OpenAI-compatible service]
    E1 -->|text description| B
    E2 -->|text description| B
    E3 -->|text description| B
    B -->|TextContent| A
    A -->|reason over description| U
```

Core flow: **receive image → call vision model → return text description**. The main model reasons over the text DeepEye returns, as if it had "seen" the image itself.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- A vision model API key (recommend [OpenAI GPT-4o](https://platform.openai.com/); or any OpenAI-compatible service such as [Alibaba Qwen-VL](https://dashscope.aliyun.com/), [Zhipu GLM-4V](https://open.bigmodel.cn/), etc.)

### 1. Clone and install

```bash
git clone <your-repo-url> deepeye
cd deepeye

# Recommended: isolated virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

After installation, the `deepeye` command is registered on your PATH.

### 2. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and fill in your vision model API key:

```dotenv
VISION_PROVIDER=openai
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4o
# For a compatible service, set OPENAI_BASE_URL instead
# OPENAI_BASE_URL=https://your-compatible-service/v1
```

### 3. Run the server

```bash
deepeye
```

The server speaks stdio with MCP clients; running it standalone won't open an interactive UI. Pair it with an MCP client (see [MCP Client Integration](#mcp-client-integration)).

---

## Tools

DeepEye exposes three MCP-compliant tools:

### `describe_image` — General image understanding

Produces a detailed description of an image, with an optional custom angle.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_source` | string | yes | Local path / http(s) URL / `data:image/...;base64,...` |
| `prompt` | string | no | Captioning prompt; defaults to a detailed description |
| `model` | string | no | Override the vision model for this call |

**Returns**: `Image analysis result:\n{description}`

### `extract_text` — OCR

Extracts text only, preserving original layout, with no extra commentary.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_source` | string | yes | Same as above |
| `language` | string | no | Recognition language; `auto` (default) auto-detects; other values (e.g. `zh`, `en`) add a language hint |

**Returns**: The plain text extracted from the image.

### `ask_about_image` — Visual question answering

Answers a specific question about the image.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_source` | string | yes | Same as above |
| `question` | string | yes | The question to ask |

**Returns**: The answer to the question.

---

## Usage Examples

### Scenario: Diagnose "users can't find the settings button" with DeepSeek V4 Flash analyzing a screenshot

**User**:

> Please analyze `screenshot.png` and find the likely reasons users can't locate the settings button.

**Model (DeepSeek V4 Flash) internals**:

1. Recognizes it needs visual input → decides to call `describe_image`
2. Invokes the tool:

   ```json
   {
     "name": "describe_image",
     "arguments": {
       "image_source": "/Users/me/screenshot.png",
       "prompt": "From a UI/UX designer's perspective, describe this webpage screenshot in detail: layout, element visibility, and potential UX issues, with special attention to 'settings' buttons."
     }
   }
   ```

3. DeepEye returns a textual description:

   > The screenshot shows an e-commerce account page. The top has the user's avatar and name, followed by three large tab-style navigation buttons: 'My Orders', 'Coupons', 'My Favorites'. At the very bottom of the page is a line of very small, grayish-white text link: 'Account Settings'...

4. The model reasons over the description and answers:

   > Based on the image analysis, there are three reasons users can't find the settings button:
   > 1. **Extremely low visual hierarchy**: 'Account Settings' is tucked into small text at the bottom, in stark contrast to the brightly colored icons at the top — easy to overlook entirely.
   > 2. **Violates mental models**: In most apps, 'Settings' lives near a gear icon in the top-right or around the avatar. Placing it at the bottom breaks convention.
   > 3. **Insufficient color contrast**: Light gray text on a white background is hard to discern.
   > Suggestion: surface 'Settings' as an icon in the upper half of the page.

### Scenario: OCR a weekly report

```json
{
  "name": "extract_text",
  "arguments": {"image_source": "/Users/me/weekly-report.png", "language": "zh"}
}
```

Returns:

> Project Weekly Report
> Date: 2026-07-31
> Progress: 95% complete
> Risks: None

### Scenario: Visual Q&A

```json
{
  "name": "ask_about_image",
  "arguments": {"image_source": "/Users/me/cats.jpg", "question": "How many cats are in the image?"}
}
```

Returns:

> There are two cats. An orange tabby is lying on the sofa, and a black cat is peeking out from behind the door.

---

## Configuration Reference

All configuration is loaded from environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_PROVIDER` | `openai` | Vision backend provider: `openai` / `gemini` / `custom` (only `openai` fully implemented so far) |
| `OPENAI_API_KEY` | — | API key for OpenAI or a compatible service |
| `OPENAI_MODEL` | `gpt-4o` | Vision model name |
| `OPENAI_BASE_URL` | — | API endpoint; leave empty for the official `https://api.openai.com/v1`; can point to Azure / a proxy / a compatible service |
| `GEMINI_API_KEY` | — | Gemini backend (reserved) |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini model name (reserved) |
| `CUSTOM_API_KEY` | — | Custom OpenAI-compatible service key (reserved) |
| `CUSTOM_BASE_URL` | — | Custom service endpoint (reserved) |
| `CUSTOM_MODEL` | `qwen-vl-max` | Custom model name (reserved) |
| `OCR_BACKEND` | `openai` | The vision backend actually used by `extract_text` |

**Example with a compatible service** (Alibaba Qwen-VL):

```dotenv
VISION_PROVIDER=openai
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_MODEL=qwen-vl-max
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## MCP Client Integration

DeepEye works with any MCP-compatible client. Common configurations below.

### Claude Desktop

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "deepeye": {
      "command": "deepeye",
      "cwd": "/absolute/path/to/deepeye",
      "env": {
        "VISION_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-your-key",
        "OPENAI_MODEL": "gpt-4o"
      }
    }
  }
}
```

> You can also place a `.env` file in the `cwd` directory; DeepEye will read it on startup.

### Cline (VS Code)

Add to Cline's MCP settings:

```json
{
  "mcpServers": {
    "deepeye": {
      "command": "deepeye",
      "cwd": "/absolute/path/to/deepeye",
      "env": { "OPENAI_API_KEY": "sk-your-key" }
    }
  }
}
```

### Cursor

Add the same server config under Cursor's settings → MCP.

### Custom agent (Python `mcp` client)

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="deepeye",
        cwd="/absolute/path/to/deepeye",
        env={"OPENAI_API_KEY": "sk-your-key"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
            # Call describe_image
            result = await session.call_tool(
                "describe_image",
                {"image_source": "/path/to/image.png"},
            )
            print(result.content[0].text)

asyncio.run(main())
```

---

## Supported Vision Backends

| Backend | Status | Notes |
|---------|--------|-------|
| **OpenAI-compatible** | Implemented | Works with OpenAI official, Azure OpenAI, Alibaba Qwen-VL, Zhipu GLM-4V, Moonshot, etc. |
| Gemini | Reserved | Adapter interface ready, implementation pending |
| Custom OpenAI-compatible | Reserved | For any self-hosted service that speaks OpenAI Chat Completions (vLLM / Ollama, etc.) |
| Local OCR (Tesseract / PaddleOCR) | Planned | Keeps data on-device for privacy-sensitive scenarios |

---

## Development

### Local dev setup

```bash
git clone <your-repo-url> deepeye
cd deepeye
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v
```

Tests cover image source parsing, the vision adapter factory, and prompt assembly for all three tools. All tests use mocks — no real API calls are made.

### Code layout

See [Project Structure](#project-structure). Vision backends follow the strategy pattern; adding one is a matter of:

1. Creating `src/deepeye/vision/xxx_adapter.py` that subclasses `VisionAdapter` and implements `describe`
2. Registering a new branch in the `vision/__init__.py` factory

---

## Project Structure

```
deepeye/
├── pyproject.toml              # Project metadata, dependencies, entry command, pytest config
├── .env.example                # Configuration sample
├── README.md                   # Chinese README (primary)
├── README.en.md                # English README
├── src/
│   └── deepeye/
│       ├── __init__.py         # __version__
│       ├── server.py           # MCP Server assembly (mcp 2.0 API)
│       ├── tools.py            # The three MCP tools
│       ├── image_utils.py      # Image source parsing (local / URL / data URI)
│       ├── config.py           # pydantic-settings config loading
│       └── vision/
│           ├── __init__.py     # create_vision_adapter factory
│           ├── base.py         # VisionAdapter abstract base
│           └── openai_adapter.py
└── tests/
    ├── test_image_utils.py
    ├── test_vision_factory.py
    └── test_tools.py
```

---

## Roadmap

- [x] OpenAI-compatible vision backend
- [x] Three image sources (local / URL / Base64)
- [x] Three core tools (describe / OCR / VQA)
- [ ] Gemini adapter
- [ ] Custom OpenAI-compatible adapter (vLLM / Ollama local deployment)
- [ ] Tesseract / PaddleOCR local OCR backend
- [ ] Image preprocessing (smart compression, auto-downscaling oversized images)
- [ ] Result caching (same image + prompt hits cache)
- [ ] Video keyframe analysis tool
- [ ] Multi-model pipeline (e.g., GPT-4o classifies first, then routes to a specialized model)
- [ ] Publish to PyPI

---

## Contributing

Issues and PRs welcome!

- **Bug reports / feature requests**: open an Issue describing the scenario and expected behavior
- **Code contributions**: open a PR with a clear title; ensure `pytest` passes
- **New vision backends**: see `vision/openai_adapter.py` for a reference `VisionAdapter` subclass, then register it in the factory
- **Docs improvements**: README / examples / config docs all welcome

### Contribution flow

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Commit: `git commit -m "feat: add xxx"`
4. Push: `git push origin feat/your-feature`
5. Open a Pull Request

---

## License

[MIT](LICENSE) © DeepEye Contributors

---

## Acknowledgements

- [Model Context Protocol](https://modelcontextprotocol.io/) — the standardized model context protocol
- [DeepSeek](https://www.deepseek.com/) — a powerful text-only reasoner, the inspiration behind DeepEye
- All vision model providers (OpenAI / Google / Alibaba / Zhipu, etc.) — for making "seeing" possible

---

<p align="center">
  If DeepEye helps you, a Star goes a long way in helping others discover it.
</p>
