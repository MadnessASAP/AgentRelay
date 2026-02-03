# AgentRelay

**AgentRelay** is a planned MCP-based orchestration service for delegating work to sub-agents and delivering their results back into a parent agent’s context via injection rather than polling.

The goal is to make parallel, multi-agent workflows feel natural to a single agent: sub-tasks run asynchronously, and their completed outputs appear as new context when ready.

---

## Vision

AgentRelay aims to provide:

- **Delegation via MCP**  
  A standard MCP interface for launching and tracking sub-agent tasks.

- **Injection-first results**  
  Sub-agent outputs are pushed back into the parent agent’s prompt/context instead of being fetched with repeated status calls.

- **Parallel execution**  
  Parent agents can continue working while sub-agents execute in the background.

- **Backend-agnostic design**  
  Sub-agents can be powered by:
  - OpenAI / Codex-style APIs  
  - Agent Control Protocol (ACP) implementations  
  - Local CLI tools or scripts  

- **Simple orchestration semantics**  
  One call to delegate work, one injected result when it’s done.

---

## Intended Interface

AgentRelay is designed around two core operations:

### `delegate(prompt, block=false)`

Launches a sub-agent task.

- If `block=false`:
  - Returns immediately with a task ID.
  - Result is injected later.

- If `block=true`:
  - Waits for completion.
  - Returns a completion acknowledgement.
  - Result is still injected in exactly the same way as non-blocking tasks.

### `status(subagent_id)`

Provides debug or manual visibility into task state.  
Normal workflows should rely on injection rather than polling.

---

## Core Principle

> **Sub-agent outputs are never returned directly from tools.**  
> They are always delivered via injection into the parent agent’s context.

This avoids:
- Polling loops
- Tool-call spam
- Split reasoning paths between blocking and non-blocking calls

And enables:
- Clean parallel reasoning
- Deterministic integration of sub-task results
- Agent-driven orchestration

---

## Long-Term Goals

AgentRelay is intended to evolve toward:

- Multiple pluggable execution backends  
- Structured (JSON) result injection  
- Progress and partial-result injection  
- Cancellation and prioritization  
- Policy controls and quotas  
- Durable state and recovery  
- Observability and tracing  
- Multi-parent subscription models  

---

## Current Status

AgentRelay is currently a design and prototype effort.

The present focus is on:
- Defining clean semantics
- Establishing injection-first result delivery
- Validating MCP as a control plane
- Proving interoperability with Codex/ACP-style agents

Expect rapid iteration and architectural churn.

---

## Why “AgentRelay”

AgentRelay acts as a relay point between:
- A parent agent that wants work done  
- One or more sub-agents that can perform it  

It does not reason on its own.  
It routes tasks and relays results.

---

## License

TBD
