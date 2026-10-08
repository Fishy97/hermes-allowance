# hermes-allowance

A fixed allowance of **tokens, model calls, tool calls, subagents and minutes** for each unit of work in [Hermes Agent](https://github.com/NousResearch/hermes-agent). When a task uses its allowance, Hermes stops it with a clear final message and writes a receipt.

Nothing here looks at price, so the limits still hold when your model is a subscription, a token plan or a local model that reports **$0**. Dollar caps can't stop those runs.

```
$ hermes chat -q "fix the build"
...
[hermes-allowance] Stopped: this task used its allowance (model_calls 301 > limit 300).
Unit: session:20261008_212750_0dfd26. No further model or tool calls will run for it.
To continue, raise the limit or run `hermes allowance reset session:20261008_212750_0dfd26`.
```

## Why

An agent loop can run away without any single call looking wrong: a stuck retry loop, a delegation fan-out, a cron job that never converges. Two self-reported examples:
- 18.7M tokens in 5 hours from one stuck session ([hermes-agent#91713](https://github.com/NousResearch/hermes-agent/issues/91713#issuecomment-5858422172)).
- About 30% of a weekly token-plan quota used by a two-sentence image task ([same issue](https://github.com/NousResearch/hermes-agent/issues/91713#issuecomment-5639771887)).

Existing tools leave three gaps for that case:
- **Dollar-only limits.** They cap USD, and a subscription or local model costs $0, so they never trip.
- **Per-call view.** They count each call alone, while the harm comes from the total.
- **Missed subagents.** Subagents get their own budget instead of spending their parent's.

## What counts as a unit of work

The first match wins:

| Unit | Key | When |
|---|---|---|
| Anything you name | `HERMES_ALLOWANCE_KEY` | Cron jobs, scripts, CI: `HERMES_ALLOWANCE_KEY=cron:nightly-report` |
| A Paperclip issue | `paperclip:<PAPERCLIP_TASK_ID>` | Automatic under Paperclip's Hermes adapter, which sets `PAPERCLIP_TASK_ID`. One budget per issue, shared across every run and process |
| A session tree | `session:<root session id>` | Default. A session plus every subagent it delegates to |

The ledger is SQLite at `$HERMES_HOME/allowance/ledger.db`. Updates are atomic across processes, so concurrent runs on the same issue share one exact count.

## What it enforces

| Limit | Default | Counted | Enforced |
|---|---|---|---|
| `model_calls` | 300 | Before each model call | The call that would exceed the limit is **not sent**. Hermes gets a final answer explaining the stop and the turn ends.¹ |
| `tool_calls` | 600 | Before each tool call | The tool call is blocked |
| `subagents` | 30 | Before `delegate_task` (one per task in a batch) | The delegation is blocked |
| `tokens` | 5,000,000 | After each model call (`total_tokens` reported by the provider) | Every later model and tool call is refused |
| `minutes` | off | Wall clock since the unit's first activity | Every later model and tool call is refused |

Any limit trips the whole unit, and it stays tripped until reset. At 80% (`warn_at`) the model gets a one-time note on its next tool result telling it to wrap up.

¹ For the `chat_completions` and `anthropic_messages` API modes, which cover OpenAI-compatible endpoints, OpenRouter, custom and local servers, and Anthropic. Other modes (Codex Responses, Bedrock) still count model calls, but the plugin cannot refuse them. There, enforcement falls back to blocking tool calls, which ends the useful work but not the turn.

## Install

```bash
hermes plugins install Fishy97/hermes-allowance --enable
```

The plugin requests no capabilities, secrets or network access. Tested on Hermes v0.21.5 (2026.9.24).

## Configure

Optional `$HERMES_HOME/allowance.yaml`. A limit of 0 means unlimited, unknown keys are rejected, and edits apply without a restart.

```yaml
limits:
  tokens: 2000000
  model_calls: 200
  tool_calls: 400
  subagents: 10
  minutes: 60
warn_at: 0.8
```

Environment variables override the file for one process: `HERMES_ALLOWANCE_TOKENS`, `HERMES_ALLOWANCE_MODEL_CALLS`, `HERMES_ALLOWANCE_TOOL_CALLS`, `HERMES_ALLOWANCE_SUBAGENTS`, `HERMES_ALLOWANCE_MINUTES`.

```bash
hermes allowance status            # recent units, counts and limits
hermes allowance reset <key|all>   # let a tripped unit run again
```

`/allowance` and `/allowance reset <key>` do the same inside a chat. Every trip is appended to `$HERMES_HOME/allowance/receipts.jsonl` with the unit, reason, counts, limits, session and pid.

## Hard backstop: `bin/allowance-run`

No Hermes plugin can cancel a model call that is already streaming, or a shell command a tool has already started. For a guarantee that does not depend on the agent, run the whole process in a transient systemd scope. The kernel then kills the process and everything it spawned:

```bash
ALLOWANCE_MINUTES=60 ALLOWANCE_MEMORY=4G ALLOWANCE_TASKS=512 bin/allowance-run hermes chat -q "..."
```

Exit 143 means the deadline stopped it. Requires Linux with a systemd user session.

## Paperclip

With Paperclip's Hermes adapter, every run on an issue gets the issue's allowance automatically, through `PAPERCLIP_TASK_ID`. Put the limits in `allowance.yaml` under the `HERMES_HOME` the adapter uses. If your adapter configuration lets you change the launch command, prefix it with `bin/allowance-run` to get the OS backstop as well.

## Honest limits

- **Late token count.** Tokens are counted after each call returns, so the call that crosses the limit still completes. `model_calls` is the limit that is checked before sending.
- **Missing usage.** A provider that returns no `usage` adds no tokens. `model_calls` and `tool_calls` still apply.
- **No mid-call stop.** Calls and tools already running finish. Use `allowance-run` to stop them.
- **Per-process session trees.** The session tree is tracked inside one process. A Paperclip issue or `HERMES_ALLOWANCE_KEY` works across processes.

## Tests

```bash
python -m pytest tests -q      # 10 tests: logic, real Hermes response normalizers, 8-process ledger race
python tests/e2e.py            # real `hermes` processes against a fake runaway model, throwaway HERMES_HOME
sh tests/backstop.sh           # systemd deadline and fork cap
```

`tests/e2e.py` runs a fake OpenAI-compatible model that calls a tool on every turn and never stops.
- **Plugin off:** Hermes runs until its own turn cap (14 calls with `max_turns: 12`).
- **Plugin on:** the run stops exactly at `model_calls`, `tool_calls` or `tokens`.
- **Paperclip issue:** a second process on the same exhausted issue sends 0 calls, and a different issue keeps its own budget.
- **Delegation:** a parent and its subagent together stop at one shared limit.

## Issues and contributions

Bug reports and small PRs are welcome. The repo is maintained with help from a Hermes agent, which reviews new issues regularly and ships small, tested fixes. The most useful bug reports include:
- `hermes --version`
- the provider and API mode
- the receipt line from `receipts.jsonl`

## License

MIT
