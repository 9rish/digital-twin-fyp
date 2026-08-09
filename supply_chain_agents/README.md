# Agent Layer - Supply Chain Digital Twin

This is Member B's slice (Section 8 of the project plan): the four agents that
detect disruptions, simulate responses, and recommend an action.

It's built against a **twin stub** (`twin_stub.py`), not the real SimPy engine —
so it runs standalone right now, and Member A can swap in the real twin later
without touching any agent code (see the "swap-out contract" comment at the
top of `twin_stub.py`).

## Structure

```
schemas.py              # Section 7 shared schemas (TwinState, DisruptionEvent, ActionRecommendation)
llm_client.py            # Provider-agnostic LLM call: Groq -> Ollama -> deterministic fallback
twin_stub.py              # Mocked twin state + forecast_action(), stands in for Member A's SimPy engine
agents/
  monitoring_agent.py     # Section 4.1 — detects disruptions (rule-based, cheap)
  simulation_agent.py     # Section 4.2 — proposes 2-3 candidate actions, forecasts each
  decision_agent.py       # Section 4.3 — picks best candidate, writes justification
orchestrator.py           # Section 4.4 — LangGraph state graph wiring the 7-stage loop
demo.py                   # Runs one full disruption through the pipeline end-to-end
```

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env      # then edit .env and paste your real GROQ_API_KEY in
python demo.py             # auto-approves the recommendation
python demo.py --reject    # walks the reject path instead
```

`.env` is gitignored — never commit it. Each teammate keeps their own local
`.env` with their own key; the code never needs a key hardcoded anywhere.

No API key is required to run this at all — see "LLM provider" below.

## LLM provider (free options)

`llm_client.py` tries providers in this order and never errors out to the caller:

1. **Groq** (recommended) — free tier, fast, OpenAI-compatible.
   Get a free key at https://console.groq.com → API Keys → Create API Key,
   then put it in `.env` (see Setup above).
2. **Ollama** — fully local, no key needed. Install from https://ollama.com, then:
   ```bash
   ollama pull llama3.1
   ollama serve
   ```
3. **Template fallback** — if neither is available, agents still run using
   deterministic, plain-language phrasing built from the real numbers. The
   *facts* (cost, delay, risk) always come from the simulation regardless of
   which tier answered — only the sentence phrasing degrades. This is
   deliberate: Section 12 of the plan flags flaky LLM calls as a live-demo
   risk, so the pipeline is designed to never break because of it.

## Why LangGraph

The pipeline has a hard requirement — Section 6.3 "Approval UI" / Section 4
"Approve — human review" — that a human must review the recommendation before
anything commits. LangGraph's `interrupt()` / `Command(resume=...)` pauses the
graph exactly at that point and resumes with the human's decision, which maps
onto the dashboard's future Approve/Reject buttons without extra plumbing.

```python
from langgraph.types import Command
from orchestrator import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "run-1"}}

result = graph.invoke({}, config)                   # runs Perceive -> Recommend, then pauses
rec = result["__interrupt__"][0].value["recommendation"]

# ... show `rec` on the dashboard, wait for the planner to click Approve/Reject ...

final = graph.invoke(Command(resume=True), config)  # True = approve, False = reject
```

## What's stubbed vs. real

| Piece | Status | Owner to replace |
|---|---|---|
| `get_twin_state()` | Stubbed (hardcoded snapshot) | Member A — swap for live SimPy read |
| `forecast_action()` | Stubbed (randomized-but-directional) | Member A — swap for forked SimPy fast-forward |
| Everything in `agents/` | Real logic, ready to integrate | — |
| `orchestrator.py` | Real LangGraph graph, ready to integrate | Member C wires this behind a FastAPI endpoint |

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Tests run fully offline — an autouse fixture (`tests/conftest.py`) patches
`call_llm` so every agent falls back to its deterministic template. No API
key or network access is needed to run the suite, and results are not
dependent on what a live LLM happens to say.

- `tests/test_pipeline_flow.py` — integration tests that run the agents in
  their real order (**Monitoring -> Simulation -> Decision**) across the 5
  evaluation scenarios from Section 10 of the project plan: a single
  shipment delay, a demand spike, a severe/lost shipment, a compound
  disruption (delay + spike together), and a low-severity case that should
  *not* trigger an overreaction.
- `tests/test_edge_cases.py` — regression tests for specific bugs found and
  fixed during review (savings-calculation error, severity-aware risk
  ceiling, alert dedup/escalation, reproducible forecasting, and the
  orchestrator's zero-event and multi-event handling). Each test is named
  after the bug it guards against.

## Next steps to integrate with the rest of the team

- **Member C (backend)**: wrap `build_graph()` + the invoke/resume calls above
  in a FastAPI endpoint; push `result["__interrupt__"]` payloads to the
  frontend over WebSocket, and call `Command(resume=...)` when the planner
  clicks Approve/Reject.
- **Member A (twin)**: once the real SimPy engine exists, replace the bodies
  of `get_twin_state()` and `forecast_action()` in `twin_stub.py` (or point
  `orchestrator.py`'s `perceive`/`simulate` nodes at the real module) —
  signatures should stay identical.
- **Audit log**: `orchestrator.py`'s `act()` node already builds a log entry
  per run; Member C can persist `audit_log` entries to SQLite as specified
  in Section 8.
