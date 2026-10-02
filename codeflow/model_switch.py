"""Natural-language model switching for the interactive agent."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    model: str
    base_url: str
    description: str


MODEL_PROFILES = {
    "pro": ModelProfile(
        name="pro",
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/anthropic",
        description="DeepSeek Pro（更重视质量）",
    ),
    "flash": ModelProfile(
        name="flash",
        provider="deepseek",
        model="deepseek-flash",
        base_url="https://api.deepseek.com/anthropic",
        description="DeepSeek Flash（V4.1，稳定 Anthropic 路由）",
    ),
    "flash41": ModelProfile(
        name="flash41",
        provider="deepseek",
        model="deepseek-flash",
        base_url="https://api.deepseek.com/anthropic",
        description="DeepSeek V4.1 Flash（官方模型名：deepseek-flash）",
    ),
}

_ALIASES = {
    "deepseek-v4-pro": "pro",
    "v4-pro": "pro",
    "pro": "pro",
    "专业": "pro",
    "深度": "pro",
    "质量优先": "pro",
    "默认": "pro",
    "deepseek-v4-flash": "flash",
    "v4-flash": "flash",
    "flash": "flash",
    "快速": "flash",
    "速度快": "flash",
    "轻量": "flash",
    "省钱": "flash",
    "deepseek-flash": "flash41",
    "deepseek-v4.1-flash": "flash41",
    "deepseek-v41-flash": "flash41",
    "deepseekv4.1flash": "flash41",
    "v4.1-flash": "flash41",
    "v41-flash": "flash41",
    "4.1-flash": "flash41",
    "4.1flash": "flash41",
    "v4.1flash": "flash41",
    "flash41": "flash41",
    "flash-4.1": "flash41",
    "flash4.1": "flash41",
    "4.1快速": "flash41",
    "新版flash": "flash41",
}
_ALIASES_BY_LENGTH = tuple(sorted(_ALIASES, key=len, reverse=True))
_SWITCH_INTENT = re.compile(r"切换|换成|换到|改成|改用|启用|使用|用|恢复")


def detect_model_switch(text):
    """Return a profile for an explicit switch request, otherwise ``None``.

    This parser intentionally requires a switch verb. A normal question that
    merely mentions “flash” or “pro” must continue to go to the model.
    """
    normalized = re.sub(r"\s+", "", str(text or "").strip().lower())
    if not normalized or not _SWITCH_INTENT.search(normalized):
        return None
    for alias in _ALIASES_BY_LENGTH:
        if alias in normalized:
            return MODEL_PROFILES[_ALIASES[alias]]
    return None


__all__ = ["MODEL_PROFILES", "ModelProfile", "detect_model_switch"]
