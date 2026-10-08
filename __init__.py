"""hermes-allowance: a fixed allowance of tokens, model calls, tool calls, subagents and
minutes per unit of work, enforced inside Hermes.

Nothing here looks at price, so limits still hold when a subscription or local model
reports $0. A unit of work is, in order of precedence:
  1. ``HERMES_ALLOWANCE_KEY`` (any string you choose, e.g. a cron job name),
  2. ``PAPERCLIP_TASK_ID`` (set by Paperclip's Hermes adapter: one budget per issue,
     shared across runs and processes),
  3. the root session of the current session tree (a session plus its subagents).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger("hermes_allowance")

SENTINEL = "hermes-allowance"
COUNTERS = ("tokens", "model_calls", "tool_calls", "subagents")
LIMITS = COUNTERS + ("minutes",)
DEFAULTS = {"tokens": 5_000_000, "model_calls": 300, "tool_calls": 600, "subagents": 30, "minutes": 0}
REFUSABLE_MODES = {"chat_completions", "anthropic_messages"}

# ponytail: in-process maps; delegation runs in-process so this covers subagents. Grows by one
# entry per child for the life of the process, fine for agent processes, prune if a gateway lives for months.
_parent: dict[str, str] = {}
_pending_warning: dict[str, str] = {}
_cfg_cache: dict = {"mtime": None, "file": {}}
_cfg_lock = threading.Lock()


# --- config ---------------------------------------------------------------------------------

def _home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _number(name: str, value) -> float:
    n = float(value)
    if n < 0:
        raise ValueError(f"hermes-allowance: limit {name} must be >= 0, got {value!r}")
    return n


def load_config() -> dict:
    """Defaults < $HERMES_HOME/allowance.yaml < HERMES_ALLOWANCE_<LIMIT> env vars. 0 = unlimited."""
    path = _home() / "allowance.yaml"
    mtime = path.stat().st_mtime if path.exists() else None
    with _cfg_lock:
        if mtime != _cfg_cache["mtime"]:
            raw = {}
            if mtime is not None:
                import yaml
                raw = yaml.safe_load(path.read_text()) or {}
            _cfg_cache.update(mtime=mtime, file=raw)
        raw = _cfg_cache["file"]
    limits = dict(DEFAULTS)
    for name, value in (raw.get("limits") or {}).items():
        if name not in LIMITS:
            raise ValueError(f"hermes-allowance: unknown limit {name!r} in {path}")
        limits[name] = _number(name, value)
    for name in LIMITS:
        env = os.environ.get(f"HERMES_ALLOWANCE_{name.upper()}")
        if env not in (None, ""):
            limits[name] = _number(name, env)
    warn_at = float(raw.get("warn_at", 0.8))
    return {"limits": limits, "warn_at": warn_at}


# --- unit of work ---------------------------------------------------------------------------

def unit_key(session_id: str) -> str:
    explicit = os.environ.get("HERMES_ALLOWANCE_KEY")
    if explicit:
        return explicit
    task = os.environ.get("PAPERCLIP_TASK_ID")
    if task:
        return f"paperclip:{task}"
    root, seen = session_id or "unknown", set()
    while root in _parent and root not in seen:
        seen.add(root)
        root = _parent[root]
    return f"session:{root}"


# --- ledger ---------------------------------------------------------------------------------

_SCHEMA = """CREATE TABLE IF NOT EXISTS units(
  key TEXT PRIMARY KEY,
  tokens INTEGER NOT NULL DEFAULT 0, model_calls INTEGER NOT NULL DEFAULT 0,
  tool_calls INTEGER NOT NULL DEFAULT 0, subagents INTEGER NOT NULL DEFAULT 0,
  started REAL NOT NULL, updated REAL NOT NULL,
  warned INTEGER NOT NULL DEFAULT 0, tripped TEXT)"""


def _dir() -> Path:
    d = _home() / "allowance"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(_dir() / "ledger.db", timeout=10, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(_SCHEMA)
    return con


def _over(row: dict, limits: dict, now: float) -> str | None:
    for name in COUNTERS:
        if limits[name] and row[name] > limits[name]:
            return f"{name} {row[name]} > limit {int(limits[name])}"
    if limits["minutes"] and (now - row["started"]) / 60 > limits["minutes"]:
        return f"minutes {(now - row['started']) / 60:.1f} > limit {limits['minutes']:g}"
    return None


def charge(key: str, *, pre: bool, session_id: str = "", **delta: int) -> dict:
    """Atomically add ``delta`` to the unit (across processes) and decide.

    ``pre=True``: the delta is an action about to happen (model call, tool call, subagent);
    if it would exceed a limit the action is refused and NOT counted. ``pre=False``: the delta
    already happened (tokens); exceeding the limit trips the unit for every later action.
    Returns ``{"allowed", "reason", "warn"}``. A unit stays tripped until reset.
    """
    cfg, now = load_config(), time.time()
    limits = cfg["limits"]
    con = _db()
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute("INSERT OR IGNORE INTO units(key, started, updated) VALUES(?,?,?)", (key, now, now))
        row = dict(con.execute("SELECT * FROM units WHERE key=?", (key,)).fetchone())
        if row["tripped"]:
            if not pre:  # it already happened: record it so receipts don't undercount
                con.execute("UPDATE units SET updated=?, " + ", ".join(f"{c}={c}+?" for c in COUNTERS)
                            + " WHERE key=?", (now, *(int(delta.get(c, 0)) for c in COUNTERS), key))
            con.execute("COMMIT")
            return {"allowed": False, "reason": row["tripped"], "warn": None}
        new = {**row, **{k: row[k] + int(v) for k, v in delta.items()}}
        reason = _over(new, limits, now)
        warn = None
        if reason:
            if pre:  # refused action is not counted
                new = row
            con.execute("UPDATE units SET tripped=?, updated=?, " + ", ".join(f"{c}=?" for c in COUNTERS)
                        + " WHERE key=?", (reason, now, *(new[c] for c in COUNTERS), key))
        else:
            used = max([new[c] / limits[c] for c in COUNTERS if limits[c]]
                       + ([(now - row["started"]) / 60 / limits["minutes"]] if limits["minutes"] else [])
                       + [0])
            if not row["warned"] and used >= cfg["warn_at"]:
                warn = f"{used:.0%}"
            con.execute("UPDATE units SET updated=?, warned=?, " + ", ".join(f"{c}=?" for c in COUNTERS)
                        + " WHERE key=?", (now, int(bool(row["warned"] or warn)), *(new[c] for c in COUNTERS), key))
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    if reason:
        _receipt(key, reason, new, limits, session_id)
        return {"allowed": not pre, "reason": reason, "warn": None}
    if warn:
        _pending_warning[key] = (
            f"[hermes-allowance] {warn} of this task's allowance is used ({key}). "
            "Finish the essential work and wrap up; new model and tool calls stop at 100%.")
    return {"allowed": True, "reason": None, "warn": warn}


def _receipt(key: str, reason: str, row: dict, limits: dict, session_id: str) -> None:
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "key": key, "reason": reason,
             "session_id": session_id, "pid": os.getpid(),
             "used": {c: row[c] for c in COUNTERS} | {"minutes": round((time.time() - row["started"]) / 60, 2)},
             "limits": limits}
    with open(_dir() / "receipts.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    log.warning("hermes-allowance tripped %s: %s", key, reason)


def refusal(key: str, reason: str) -> str:
    return (f"[hermes-allowance] Stopped: this task used its allowance ({reason}). Unit: {key}. "
            f"No further model or tool calls will run for it. To continue, raise the limit or run "
            f"`hermes allowance reset {key}`.")


# --- hooks ----------------------------------------------------------------------------------

def _synthetic_response(api_mode: str, text: str):
    """A final assistant answer in the provider's native shape, so the turn ends cleanly."""
    if api_mode == "anthropic_messages":
        return SimpleNamespace(id=SENTINEL, model=SENTINEL, type="message", role="assistant",
                               content=[SimpleNamespace(type="text", text=text)],
                               stop_reason="end_turn", stop_sequence=None, usage=None)
    message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    return SimpleNamespace(id=SENTINEL, model=SENTINEL, usage=None,
                           choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")])


def on_llm_execution(request=None, next_call=None, session_id="", api_mode="", **_):
    key = unit_key(session_id)
    refusable = api_mode in REFUSABLE_MODES
    # unrefusable modes: the call goes out regardless, so count it as having happened
    verdict = charge(key, pre=refusable, session_id=session_id, model_calls=1)
    if verdict["allowed"] or not refusable:
        return next_call(request)  # unsupported modes fall back to tool blocking
    return _synthetic_response(api_mode, refusal(key, verdict["reason"]))


def on_post_api_request(session_id="", usage=None, response_model=None, **_):
    if response_model == SENTINEL:
        return
    tokens = int((usage or {}).get("total_tokens") or 0)
    if tokens:
        charge(unit_key(session_id), pre=False, session_id=session_id, tokens=tokens)


def on_pre_tool_call(tool_name="", args=None, session_id="", task_id="", **_):
    subagents = 0
    if tool_name == "delegate_task":
        tasks = (args or {}).get("tasks")
        subagents = len(tasks) if isinstance(tasks, list) and tasks else 1
    key = unit_key(session_id or task_id)
    verdict = charge(key, pre=True, session_id=session_id, tool_calls=1, subagents=subagents)
    if not verdict["allowed"]:
        return {"action": "block", "message": refusal(key, verdict["reason"])}
    return None


def on_subagent_start(parent_session_id=None, child_session_id=None, **_):
    if parent_session_id and child_session_id and child_session_id != parent_session_id:
        _parent[child_session_id] = parent_session_id


def on_transform_tool_result(result=None, session_id="", task_id="", **_):
    note = _pending_warning.pop(unit_key(session_id or task_id), None)
    if note and isinstance(result, str):
        return f"{result}\n\n{note}"
    return None


# --- operator commands ----------------------------------------------------------------------

def status(limit: int = 15) -> str:
    con = _db()
    try:
        rows = con.execute("SELECT * FROM units ORDER BY updated DESC LIMIT ?", (limit,)).fetchall()
    finally:
        con.close()
    lim = load_config()["limits"]
    head = "limits: " + ", ".join(f"{k}={'∞' if not v else f'{v:,.0f}' if v == int(v) else f'{v:g}'}"
                                  for k, v in lim.items())
    if not rows:
        return head + "\n(no units yet)"
    lines = [head]
    for r in rows:
        mins = (r["updated"] - r["started"]) / 60
        state = f"TRIPPED: {r['tripped']}" if r["tripped"] else "ok"
        lines.append(f"{r['key']}  tokens={r['tokens']} model_calls={r['model_calls']} "
                     f"tool_calls={r['tool_calls']} subagents={r['subagents']} minutes={mins:.1f}  {state}")
    return "\n".join(lines)


def reset(key: str) -> str:
    con = _db()
    try:
        n = con.execute("DELETE FROM units" + ("" if key == "all" else " WHERE key=?"),
                        () if key == "all" else (key,)).rowcount
    finally:
        con.close()
    return f"reset {n} unit(s)"


def _slash(raw_args: str = "") -> str:
    parts = (raw_args or "").split()
    if len(parts) == 2 and parts[0] == "reset":
        return reset(parts[1])
    return status()


def _cli_setup(parser) -> None:
    sub = parser.add_subparsers(dest="allowance_cmd")
    sub.add_parser("status", help="show recent units and limits")
    r = sub.add_parser("reset", help="clear one unit (or 'all') so it can run again")
    r.add_argument("key")


def _cli_run(args) -> None:
    print(reset(args.key) if getattr(args, "allowance_cmd", None) == "reset" else status())


def register(ctx) -> None:
    ctx.register_middleware("llm_execution", on_llm_execution)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    ctx.register_command("allowance", _slash, description="Show or reset task allowances",
                         args_hint="[reset <key|all>]")
    ctx.register_cli_command("allowance", "Show or reset hermes-allowance units", _cli_setup, _cli_run)
