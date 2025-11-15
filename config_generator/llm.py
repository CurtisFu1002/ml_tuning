"""LLM interaction utilities."""

import logging
from typing import Any

from ollama import chat

from config_generator.prompts.format import ConfigYaml


def get_llm_response(model_name: str, prompt: str) -> str:
    """Get raw text response from LLM."""
    logging.info(f"Using model: {model_name}")
    resp = chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.message.content


def get_llm_response_parsed(model_name: str, prompt: str) -> dict[str, Any]:
    """Get structured response from LLM using JSON schema."""
    logging.info(f"Using model: {model_name}")
    resp = chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format=ConfigYaml.model_json_schema(),
        options={"temperature": 0},
    )
    return ConfigYaml.model_validate_json(resp.message.content).model_dump()
