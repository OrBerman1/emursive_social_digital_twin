"""Which Claude to call, and how: one place for the provider, key and model (used by vlm.py and jeeves.py).

Provider order: Anthropic API key (env ANTHROPIC_API_KEY or the key saved on the projects page), else an Amazon
Bedrock API key (env AWS_BEARER_TOKEN_BEDROCK or the saved Bedrock key; region env AWS_REGION or the saved region,
default us-east-1). Keys live outside the repo in %LOCALAPPDATA%/ShukTwin/ and are never logged.
Model: Claude Sonnet 5.5 (user choice, 14 Oct: Opus is more than these tasks need); override with SHUK_LLM_MODEL.
"""
from __future__ import annotations

import os
from pathlib import Path

KEY_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ShukTwin"
MODELS = {"anthropic": "claude-sonnet-5-5", "bedrock": "global.anthropic.claude-sonnet-5-5"}
_client = None


def _read(name: str) -> str:
    try:
        return (KEY_DIR / name).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def provider() -> tuple[str, str]:
    """('anthropic' | 'bedrock' | '', key)."""
    k = os.environ.get("ANTHROPIC_API_KEY") or _read("anthropic_api_key")
    if k:
        return "anthropic", k
    k = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or _read("bedrock_api_key")
    if k:
        return "bedrock", k
    return "", ""


def available() -> bool:
    return bool(provider()[0])


def region() -> str:
    return os.environ.get("AWS_REGION") or _read("bedrock_region") or "us-east-1"


def model() -> str:
    return os.environ.get("SHUK_LLM_MODEL") or MODELS.get(provider()[0], MODELS["anthropic"])


def client():
    global _client
    if _client is None:
        import anthropic
        prov, key = provider()
        if prov == "anthropic":
            _client = anthropic.Anthropic(api_key=key, max_retries=4, timeout=300)
        elif prov == "bedrock":
            _client = anthropic.AnthropicBedrock(api_key=key, aws_region=region(), max_retries=4, timeout=300)
        else:
            raise RuntimeError("no Claude key: save an Anthropic or Bedrock key on the projects page")
    return _client


def create(**kw):
    """messages.create with the provider's extras: on the Anthropic API a server-side fallback on a refusal
    (not available on Bedrock, where a refusal is returned as stop_reason 'refusal')."""
    kw.setdefault("model", model())
    if provider()[0] == "anthropic":
        return client().beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    return client().messages.create(**kw)
