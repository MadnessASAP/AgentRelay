# HANDOVER — AgentRelay MCP Subagent MVP (Injection-first)

## 1. Goals and constraints

MVP requirements implemented:
- Python 3.10+ implementation with asyncio
- Standard MCP server interface using `mcp` (PyPI)
- Pluggable subagent backends (Codex-ACP, Mock)
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

## 2. Project structure

```
AgentRelay/
├── src/agent_relay/              # Main package
│   ├── __init__.py               # Package exports (Job, JobManager, JobState)
│   ├── __main__.py               # Entry point for `python -m agent_relay`
│   ├── server.py                 # MCP server setup and main() entry point
│   ├── manager.py                # Core job management logic
│   ├── handlers.py               # Tool handler functions (testable)
│   ├── backends/                 # Pluggable backend implementations
│   │   ├── __init__.py           # Backend exports
│   │   ├── acp.py                # Codex-ACP backend adapter
│   │   └── mock.py               # Mock backend for testing
│   └── py.typed                  # PEP 561 marker for type hints
├── tests/                        # Comprehensive test suite
│   ├── __init__.py
│   ├── test_mcp_tools.py         # MCP protocol integration tests
│   └── test_mock_agent.py        # Mock agent workflow tests
├── pyproject.toml                # Package configuration (hatch build system)
├── README.md                     # Project vision and goals
├── HANDOVER.md                   # This file
└── LICENSE                       # MIT license
```

## 3. Architecture overview

Single process, asyncio-based:

- **MCP server** (`src/agent_relay/server.py`)
  - Tools:
    - `delegate(prompt, block=false)`
    - `status(subagent_id)`
  - Wires together:
    - `JobManager` (in-memory job registry + concurrency limiter)
    - Backend (ACPBackend or MockBackend)
    - `inject_into_parent()` (injection hook; MVP prints to stdout)

- **Tool handlers** (`src/agent_relay/handlers.py`)
  - Separated from server for testability
  - Pure async functions without MCP dependencies
  - `create_tool_handlers(manager)` returns delegate and status handlers

- **Core manager** (`src/agent_relay/manager.py`)
  - `JobState` enum: QUEUED, RUNNING, COMPLETED, FAILED
  - `Job` dataclass: id, prompt, state, result, error
  - `JobManager`:
    - Creates jobs
    - Runs jobs via backend with a semaphore for concurrency control
    - Calls injection callback with a canonical completion message

- **Backends** (`src/agent_relay/backends/`)
  - `ACPBackend` (`acp.py`): Codex-ACP execution adapter
  - `MockBackend` (`mock.py`): Echo backend for testing (fails on `"<FAIL>"`)

## 4. Injection-first invariant (critical)

- MCP tool results **never** contain subagent output.
- Subagent output is delivered **only** via the injection callback.

MVP injection behavior:
- `inject_into_parent()` prints to stdout.

Next step:
- Replace the injector to append a message into the parent Codex run context.

## 5. Tool semantics

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

## 6. Concurrency

- `JobManager` uses `asyncio.Semaphore(max_concurrent)`
- Adjust via `--max-concurrent` CLI flag (default: 4)

## 7. Testing

The project has comprehensive test coverage:

- **`test_mcp_tools.py`**: MCP protocol integration tests
  - Tests tool calls through actual MCP server
  - Validates delegate and status tool responses
  - Tests blocking vs non-blocking behavior
  - Verifies injection-first invariant

- **`test_mock_agent.py`**: Mock agent workflow tests
  - Tests tool handlers in isolation
  - Uses MockManager for deterministic behavior
  - Covers state transitions and error handling

Run tests:
```bash
pip install -e ".[dev]"
pytest
```

### Fixing the last two failing tests

Current `pytest` run (with `pytest-asyncio` installed) leaves two failing tests in `tests/test_manager.py`:
1) `test_delegate_injects_result_into_calling_agent_prompt` fails because the client never receives any
   injected messages via MCP notifications.
2) `test_delegate_tool_requires_prompt` fails because the `delegate` tool call does not raise when
   `prompt` is missing.

To get them passing:

1) **Make injection visible to the client session.**
   - Today, `JobManager` calls the injected callback, but `inject_into_parent()` only prints.
   - Add a transport that emits an MCP `notifications/message` notification to the client (or another
     observable channel for the MCP session).
   - The `PromptCapture` helper in `tests/test_manager.py` is listening for `LoggingMessageNotification`
     with `params.data` containing the injected message. Hook the injection to send a logging notification
     for that message.
   - Suggested implementation: extend `create_server()` so it can accept an injection callback that can
     access the MCP server/session and send `notifications/message` via the MCP server’s logging facilities,
     or implement a simple in-memory broadcast that `PromptCapture` can receive (and update the test to
     match the chosen approach).

2) **Ensure tool input validation rejects missing `prompt`.**
   - FastMCP’s `@tool` decorator does not enforce schema validation by default the way the old `Tool`
     registration did.
   - Options:
     - Add explicit argument validation inside the `delegate` tool wrapper in `src/agent_relay/server.py`
       (e.g., raise a `ValueError` when `prompt` is falsy/missing).
     - Or, switch to using FastMCP’s structured tools with input schema validation if supported by the
       installed `mcp` version.
   - The test expects an exception to be raised on missing `prompt`.

## 8. How to run

### Installation

```bash
# Install package with dev dependencies
pip install -e ".[dev]"

# Or install runtime only
pip install -e .
```

### Running the server

```bash
# Using the entry point
agent-relay --acp-endpoint <YOUR_ENDPOINT> --max-concurrent 4

# Or as a module
python -m agent_relay --acp-endpoint <YOUR_ENDPOINT> --max-concurrent 4
```

## 9. Notes on codex-acp integration

`src/agent_relay/backends/acp.py` contains placeholder pseudocode.
Update:
- `start_session()`
- `session.send(prompt)`
- `session.wait()`
- `response.text`
to match your codex-acp version.

## 10. Implementation status

Completed:
- [x] MCP tool surface (delegate, status)
- [x] Core job manager with state machine
- [x] Concurrency control via semaphore
- [x] Injection-first result delivery
- [x] Mock backend for testing
- [x] Comprehensive test suite
- [x] Package structure with pyproject.toml

In progress:
- [ ] Integrate real codex-acp API
- [ ] Integrate real injection into Codex host

## 11. Suggested next extensions

- `cancel(subagent_id)` tool
- Structured JSON injection
- Progress events
- Per-parent job grouping
- Pub/sub for multiple parent sessions
- Durable state and recovery
