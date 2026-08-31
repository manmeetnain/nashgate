"""Rough token estimates — good enough to feed the router's observation
and to price a request when a backend doesn't return usage. Real
tokenization is backend/model-specific; this is deliberately a cheap
heuristic (~4 chars/token, the commonly-cited English average) rather
than a hard dependency on any one tokenizer."""

CHARS_PER_TOKEN = 4


def estimate_request_tokens(messages: list) -> int:
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, total_chars // CHARS_PER_TOKEN)


def total_tokens_from_usage(usage: dict, fallback: int) -> int:
    if not usage:
        return fallback
    return usage.get("total_tokens") or (
        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    ) or fallback
