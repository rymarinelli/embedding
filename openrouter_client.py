#!/usr/bin/env python3
"""
Shared OpenRouter client helpers used by the benchmark_*.py scripts in this repo
(benchmark_norsumm.py, benchmark_norquad.py, benchmark_normedqa.py).
"""

import os
import sys
import time
import requests

BASE_URL = "https://openrouter.ai/api/v1"


def get_api_key(explicit_key: str = None) -> str:
    if explicit_key:
        key, source = explicit_key, "explicit argument"
    else:
        key = os.environ.get("OPENROUTER_API_KEY")
        source = "environment variable"

    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("OPENROUTER_API_KEY")
            source = "Colab Secrets"
        except Exception:
            pass

    if not key:
        sys.exit(
            "Error: no API key found. Use one of:\n"
            "  run(api_key=OPENROUTER_API_KEY)\n"
            "  os.environ['OPENROUTER_API_KEY'] = 'sk-or-...'\n"
            "  Colab Secrets named exactly OPENROUTER_API_KEY"
        )

    masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    print(f"Using API key from {source}: {masked}")
    return key


def call_model(
    model_id: str,
    prompt: str,
    headers: dict,
    max_tokens: int = 300,
    temperature: float = 0.3,
    max_retries: int = 2,
    base_backoff: float = 4.0,
    timeout: int = 60,
):
    """POST a single chat completion request to OpenRouter, with basic 429 retry."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    attempt = 0
    while True:
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 429 and attempt < max_retries:
                attempt += 1
                time.sleep(base_backoff * attempt)
                continue
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:150]}"
            body = resp.json()
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning") or ""
            content = content.strip()
            if not content:
                return None, "(empty response)"
            return content, None
        except requests.RequestException as e:
            return None, str(e)
        except (KeyError, IndexError) as e:
            return None, f"Unexpected response shape: {e}"


def build_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://local-test.script",
        "X-Title": "norwegian-llm-benchmark",
    }
