# HANDOVER — MCP Subagent MVP (Injection-first)

## 1. Goals and constraints

MVP requirements implemented by this skeleton:
- Python implementation
- Standard MCP server interface using `mcp` (PyPI)
- Single subagent backend selected at startup (Codex-ACP)
- No database / persistence; state is in-memory structs
- Core feature: result injection into parent context (no status polling loop)
- `block=true` behaves like `block=false` except it waits and returns
  `{subagent_id, state:"completed"}` immediately followed by the same injected message

Non-goals for MVP:
- No durable state across restarts
- No nested delegation
- No streaming progress
- No security/auth
- No cancellation tool

## 2. Architecture overview

Single process, asyncio-based:

- MCP server (`mcp_server.py`)
  - Tools:
    - `delegate(prompt, block=false)`
    - `status(subagent_id)`
  - Wires together:
    - `JobManager` (in-memory job registry + concurrency limiter)
    - `ACPBackend` (Codex-ACP execution)
    - `inject_into_parent()` (injection hook; MVP prints to stdout)

- Core manager (`manager.py`)
  - `JobState` enum
  - `Job` dataclass
  - `JobManager`
    - Creates jobs
    - Runs jobs via backend with a semaphore for concurrency control
    - Calls injection callback with a canonical completion message

- Codex-ACP backend adapter (`backend_acp.py`)
  - `ACPBackend.run(prompt) -> str` (async)
  - Uses `run_in_executor()` for synchronous clients

## 3. Injection-first invariant (critical)

- MCP tool results never contain subagent output.
- Subagent output is delivered only via the injection callback.

MVP injection behavior:
- `inject_into_parent()` prints to stdout.

Next step:
- Replace the injector to append a message into the parent Codex run context.

## 4. Tool semantics

### delegate(prompt, block=false)

Common:
- Create Job (state queued)
- Schedule background task `manager.run_job(job)`

If `block=false`:
- Return immediately: `{subagent_id, state:"queued"}`
- Later, injection delivers completion/failure message

If `block=true`:
- Await job completion
- Return: `{subagent_id, state:"completed"|"failed"}`
- Injection delivers the same canonical message as non-blocking

### status(subagent_id)

- Returns in-memory state
- Debug/manual only; avoid polling in normal flow

## 5. Concurrency

- `JobManager` uses `asyncio.Semaphore(max_concurrent)`
- Adjust via `--max-concurrent`

## 6. Implementation plan (incremental)

1) Verify MCP tool surface
2) Stub backend (fake delay + static result)
3) Integrate codex-acp
4) Integrate real injection into Codex host
5) Add cancel tool (optional)

## 7. Notes on codex-acp integration

`backend_acp.py` contains placeholder pseudocode.
Update:
- `start_session()`
- `session.send(prompt)`
- `session.wait()`
- `response.text`
to match your codex-acp version.

## 8. Suggested next extensions

- `cancel(subagent_id)`
- Structured JSON injection
- Progress events
- Per-parent job grouping
- Pub/sub for multiple parent sessions

## 9. How to run

```bash
pip install mcp
# install codex-acp per its repo instructions
python mcp_server.py --acp-endpoint <YOUR_ENDPOINT> --max-concurrent 4
