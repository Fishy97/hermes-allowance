"""Run: <hermes venv>/bin/python -m pytest tests/ -q   (the transport test needs Hermes importable)."""
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = Path.home() / ".hermes" / "hermes-agent"


def load_plugin():
    spec = importlib.util.spec_from_file_location("hermes_allowance", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ha(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("HERMES_ALLOWANCE_KEY", "PAPERCLIP_TASK_ID"):
        monkeypatch.delenv(var, raising=False)
    mod = load_plugin()
    monkeypatch.setattr(mod, "_home", lambda: tmp_path)
    return mod


def test_tool_call_limit_blocks_and_does_not_count_refused(ha, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_ALLOWANCE_TOOL_CALLS", "3")
    results = [ha.on_pre_tool_call(tool_name="terminal", session_id="s1") for _ in range(5)]
    assert results[:3] == [None, None, None]
    assert all(r["action"] == "block" and "tool_calls 4 > limit 3" in r["message"] for r in results[3:])
    assert "tool_calls=3 " in ha.status() and "TRIPPED" in ha.status()
    receipts = (tmp_path / "allowance" / "receipts.jsonl").read_text().splitlines()
    assert len(receipts) == 1 and json.loads(receipts[0])["used"]["tool_calls"] == 3


def test_tokens_trip_then_model_call_refused_with_final_answer(ha, monkeypatch):
    monkeypatch.setenv("HERMES_ALLOWANCE_TOKENS", "100")
    calls = []
    nxt = lambda req: calls.append(req) or "real"
    assert ha.on_llm_execution(request={}, next_call=nxt, session_id="s", api_mode="chat_completions") == "real"
    ha.on_post_api_request(session_id="s", usage={"total_tokens": 60})
    assert ha.on_llm_execution(request={}, next_call=nxt, session_id="s", api_mode="chat_completions") == "real"
    ha.on_post_api_request(session_id="s", usage={"total_tokens": 60})  # 120 > 100: trips
    resp = ha.on_llm_execution(request={}, next_call=nxt, session_id="s", api_mode="chat_completions")
    assert len(calls) == 2 and resp.model == ha.SENTINEL
    assert resp.choices[0].finish_reason == "stop" and "tokens 120 > limit 100" in resp.choices[0].message.content
    assert ha.on_pre_tool_call(tool_name="terminal", session_id="s")["action"] == "block"
    # synthetic response must not be charged
    ha.on_post_api_request(session_id="s", usage=None, response_model=ha.SENTINEL)


def test_unsupported_api_mode_falls_back_to_real_call(ha, monkeypatch):
    monkeypatch.setenv("HERMES_ALLOWANCE_MODEL_CALLS", "1")
    nxt = lambda req: "real"
    assert ha.on_llm_execution(request={}, next_call=nxt, session_id="s", api_mode="codex_responses") == "real"
    assert ha.on_llm_execution(request={}, next_call=nxt, session_id="s", api_mode="codex_responses") == "real"
    assert ha.on_pre_tool_call(tool_name="x", session_id="s")["action"] == "block"
    # calls that went out after the trip, and their tokens, are still counted
    assert ha.on_llm_execution(request={}, next_call=nxt, session_id="s", api_mode="codex_responses") == "real"
    ha.on_post_api_request(session_id="s", usage={"total_tokens": 70})
    assert "tokens=70 model_calls=3 " in ha.status()


def test_subagents_roll_up_to_root_and_delegation_is_counted(ha, monkeypatch):
    monkeypatch.setenv("HERMES_ALLOWANCE_SUBAGENTS", "2")
    ha.on_subagent_start(parent_session_id="root", child_session_id="child")
    ha.on_subagent_start(parent_session_id="child", child_session_id="grandchild")
    assert ha.unit_key("grandchild") == "session:root"
    assert ha.on_pre_tool_call(tool_name="delegate_task", args={"tasks": [{}, {}]}, session_id="root") is None
    blocked = ha.on_pre_tool_call(tool_name="delegate_task", args={"goal": "x"}, session_id="grandchild")
    assert "subagents 3 > limit 2" in blocked["message"]


def test_paperclip_issue_and_explicit_key(ha, monkeypatch):
    monkeypatch.setenv("PAPERCLIP_TASK_ID", "ISS-7")
    assert ha.unit_key("any") == "paperclip:ISS-7"
    monkeypatch.setenv("HERMES_ALLOWANCE_KEY", "cron:nightly")
    assert ha.unit_key("any") == "cron:nightly"


def test_warning_once_at_80_percent(ha, monkeypatch):
    monkeypatch.setenv("HERMES_ALLOWANCE_TOOL_CALLS", "5")
    notes = []
    for _ in range(5):
        ha.on_pre_tool_call(tool_name="t", session_id="w")
        notes.append(ha.on_transform_tool_result(result="ok", session_id="w"))
    assert [n is not None for n in notes] == [False, False, False, True, False]
    assert "80% of this task's allowance" in notes[3]


def test_minutes_limit(ha, monkeypatch):
    monkeypatch.setenv("HERMES_ALLOWANCE_MINUTES", "1")
    assert ha.on_pre_tool_call(tool_name="t", session_id="m") is None
    real = ha.time.time
    monkeypatch.setattr(ha.time, "time", lambda: real() + 61)
    assert "minutes" in ha.on_pre_tool_call(tool_name="t", session_id="m")["message"]


def test_reset_and_config_validation(ha, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_ALLOWANCE_TOOL_CALLS", "1")
    ha.on_pre_tool_call(tool_name="t", session_id="r")
    assert ha.on_pre_tool_call(tool_name="t", session_id="r")["action"] == "block"
    assert ha._slash("reset session:r") == "reset 1 unit(s)"
    assert ha.on_pre_tool_call(tool_name="t", session_id="r") is None
    (tmp_path / "allowance.yaml").write_text("limits:\n  dollars: 5\n")
    with pytest.raises(ValueError, match="unknown limit"):
        ha.load_config()
    (tmp_path / "allowance.yaml").write_text("limits:\n  tokens: -1\n")
    with pytest.raises(ValueError, match=">= 0"):
        ha.load_config()


def _worker(home, n, start_at, out):
    import os
    import time
    os.environ["HERMES_HOME"] = home
    os.environ["HERMES_ALLOWANCE_TOOL_CALLS"] = "100"
    os.environ["HERMES_ALLOWANCE_KEY"] = "shared"
    mod = load_plugin()
    mod._home = lambda: Path(home)
    mod._db().close()
    time.sleep(max(0, start_at - time.time()))  # all processes hammer the ledger at once
    ok = 0
    for _ in range(n):
        try:
            ok += mod.on_pre_tool_call(tool_name="t", session_id="p") is None
        except Exception:
            ok += 1000  # an error here is a failure, make it visible in the sum
    out.put(ok)


def test_limit_is_exact_across_processes(tmp_path):
    import time
    ctx = mp.get_context("spawn")
    q, start_at = ctx.Queue(), time.time() + 3
    procs = [ctx.Process(target=_worker, args=(str(tmp_path), 40, start_at, q)) for _ in range(8)]
    for p in procs:
        p.start()
    allowed = sum(q.get(timeout=60) for _ in procs)
    for p in procs:
        p.join(10)
    assert allowed == 100  # 320 concurrent attempts, exactly 100 allowed


@pytest.mark.skipif(not (HERMES_SRC / "agent").exists(), reason="Hermes source not available")
def test_synthetic_responses_pass_real_hermes_normalizers(ha):
    sys.path.insert(0, str(HERMES_SRC))
    from agent.transports.anthropic import AnthropicTransport
    from agent.transports.chat_completions import ChatCompletionsTransport

    for transport, mode in ((ChatCompletionsTransport(), "chat_completions"),
                            (AnthropicTransport(), "anthropic_messages")):
        out = transport.normalize_response(ha._synthetic_response(mode, "stopped"))
        assert out.content == "stopped" and not out.tool_calls and out.finish_reason == "stop", mode
