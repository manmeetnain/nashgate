from nashgate.gateway.tokens import estimate_request_tokens, total_tokens_from_usage


def test_estimate_request_tokens_uses_chars_per_token_heuristic():
    messages = [{"role": "user", "content": "a" * 40}]
    assert estimate_request_tokens(messages) == 10


def test_estimate_request_tokens_sums_across_messages():
    messages = [{"role": "system", "content": "a" * 20}, {"role": "user", "content": "b" * 20}]
    assert estimate_request_tokens(messages) == 10


def test_estimate_request_tokens_never_returns_zero():
    assert estimate_request_tokens([{"role": "user", "content": ""}]) == 1
    assert estimate_request_tokens([]) == 1


def test_total_tokens_from_usage_prefers_total_tokens_field():
    usage = {"total_tokens": 42, "prompt_tokens": 10, "completion_tokens": 5}
    assert total_tokens_from_usage(usage, fallback=1) == 42


def test_total_tokens_from_usage_sums_prompt_and_completion_when_total_missing():
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    assert total_tokens_from_usage(usage, fallback=1) == 15


def test_total_tokens_from_usage_falls_back_when_usage_empty():
    assert total_tokens_from_usage({}, fallback=99) == 99
    assert total_tokens_from_usage(None, fallback=99) == 99
