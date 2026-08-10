# Agentic AI Digital Twin — Project Status

_Last updated: 10 Aug 2026_

This document tracks what's been built, what's been tested, known issues, and what's next — for the team and for anyone picking this up mid-project.

---

## 1. Where things stand

| Layer | Owner | Status |
|---|---|---|
| Digital twin (simulation engine) | Member A | ✅ Built, tested, working |
| Agents (Monitoring, Simulation, Decision, Orchestrator) | Member B | ✅ Built, integrated with twin, tested end-to-end |
| Backend / API | Member C | ⬜ Not started — ready to start |
| Frontend / dashboard | Member D | ⬜ Not started — ready to start |

**Bottom line: the twin + agent layers are tested and ready to hand off to C and D.** Both items flagged to Member B have been resolved and verified — see Section 4.

---

## 2. What's been built

### Digital twin (`digital-twin-fyp-main/`)
- SimPy-based discrete-event engine (`twin.py`) modeling warehouses, shipments, and suppliers.
- Synthetic data generator (`data_loader.py`) as a fallback when the real Kaggle DataCo CSV isn't present — lets everyone build without needing the dataset.
- Fork-based simulation: candidate actions run on an isolated copy of the twin, never touching live state. Verified explicitly — live twin cost stays unchanged across simulation runs.
- Disruption injector (`disruption_injector.py`): `inject_shipment_delay`, `inject_demand_spike`, `inject_stock_imbalance`, `random_disruption`.
- Schemas match the shared contract in the original project plan (Section 7): twin state, disruption event, action/recommendation.

### Agents (`supply_chain_agents/`)
- **Monitoring/Exception agent** — polls twin state, flags disruptions (shipment delays, demand spikes, stock imbalance, overstock, supplier lead-time/reliability issues) against configurable thresholds, with cooldown/dedup so the same issue doesn't re-alert every cycle.
- **Simulation agent** — builds 2–3 candidate response actions per disruption (reorder, reroute, reallocate, wait) and runs each through the twin's fork mechanism to get cost/delay/stockout/service-level outcomes. Some event types (e.g. supplier reliability issues) only ever offer "wait" by design — there's no reorder/reroute response that addresses "this supplier is unreliable."
- **Decision/Recommendation agent** — compares candidate outcomes against a configurable risk ceiling, picks one using a configurable objective (`profit` by default — see Section 4), and writes a plain-language justification with the real numbers (not invented by the LLM).
- **Orchestrator** — built with LangGraph; sequences Perceive → Detect → Simulate → Recommend → (human approval via `interrupt()`) → Act → Log. Full audit trail persisted per disruption. Node-level failures (timeouts, exceptions) fail loudly rather than silently retrying in a way that could corrupt shared state (see Section 4).
- **LLM fallback chain** (`llm_client.py`): Groq → Ollama → deterministic template, so the pipeline never hard-depends on one LLM provider being up.
- 70/70 tests passing (`pytest tests/`), covering candidate-building logic, the twin/schema adapter, edge cases (dedup, cooldown, reproducibility, objective-based savings math), and all 5 of the Section 10 evaluation scenarios (shipment delay, demand spike, severe shipment loss, compound disruption, low-severity no-overreaction).

---

## 3. Bugs found during testing and fixed

1. **`demo.py` never actually injected a disruption.** It only read the twin's baseline state, so it usually reported "no disruptions" unless the synthetic data happened to already look unhealthy. Fixed by adding a real `inject_shipment_delay` call before Perceive runs, so the demo now exercises the actual multi-candidate decision pipeline.
2. **Duplicate twin code in the repo.** `twin_adapter.py` expects `digital-twin-fyp-main/` as a sibling folder to `supply_chain_agents/`, which led to a second, redundant copy of the twin engine being committed to avoid path issues. This is still a repo-hygiene item to clean up (see Next Steps).
3. **`_resilient_call` retry race condition — fixed.** The orchestrator was retrying a node on timeout, but the timed-out attempt wasn't actually cancelled — it kept running in the background. Because `monitoring_agent`'s alert-dedup state is shared, an abandoned attempt could silently poison what the retry saw, causing the retry to return an empty result instead of an error. This surfaced when LLM calls (Groq) were slow/unconfigured and fell through to a local Ollama fallback that isn't running, causing `describe_event()` calls to eat the 15-second node timeout. Fixed the trigger (added a Groq API key so LLM calls resolve fast) **and** the underlying race: `_resilient_call` no longer retries on timeout — since Python threads can't be forcibly cancelled, retrying while the first attempt is still alive was never safe, so a timeout now fails loudly instead of silently returning a possibly-corrupted result. All existing tests still pass.
4. **Decision agent was optimizing the wrong metric — fixed.** In the `SHP-482` shipment-delay scenario, "wait" was chosen over "reroute" because it had the lower `cost_delta` — but `cost_delta` only counts direct operational spend and ignores revenue lost to a worse service level, which `profit_impact` does account for (`profit_impact = -cost_delta - lost_sales_effect`). On this scenario `profit_impact` was ~3x worse for "wait" (−₹20,868 vs −₹6,708 for reroute), and reroute also gave a perfect service level. Added a `"profit"` objective option to the Decision agent and made it the default (confirmed with Member B). Also fixed a related bug this exposed: the justification text's "savings" figure was hardcoded to `cost_delta` regardless of which objective actually drove the ranking, so it said "saves ~₹0" for an action that actually saved ~₹14,160 in profit terms — now it reports savings in whichever metric was actually used. Verified end-to-end: `SHP-482` now correctly picks `reroute` with an honest justification, across multiple runs. Added a new test for the profit-objective savings math; 70/70 tests passing.

---

## 4. Questions raised with Member B — both resolved

1. ~~Decision agent may be optimizing the wrong metric.~~ **Resolved** — see bug #4 above. Default objective changed to `profit`, verified against the real `SHP-482` scenario, B confirmed the change.
2. ~~`_resilient_call`'s retry-on-timeout isn't safe for stateful nodes.~~ **Resolved** — see bug #3 above. Timeouts now fail loudly instead of retrying.

---

## 5. Known repo-hygiene issues (non-blocking, clean up when convenient)

- Duplicate copy of the twin engine files exists inside `supply_chain_agents/digital-twin-fyp-main/` to satisfy `twin_adapter.py`'s relative-path assumption. Should be replaced with a configurable path (env var or argument) instead of relying on folder duplication.
- `README.md` in `supply_chain_agents/` still describes an old `twin_stub.py` setup — stale, should be updated to reflect the current `twin_adapter.py`-based integration.
- `__pycache__` folders were committed to git despite being in `.gitignore` (gitignore only blocks new files, not already-tracked ones) — run `git rm -r --cached __pycache__` to clean up.
- `requirements.txt` for the agent layer doesn't pin exact versions (`langgraph>=0.2.0`, etc.) — worth pinning once the team settles on versions, to avoid environment drift between machines.

---

## 6. Next steps

**For Member C (Backend/API):**
- Build FastAPI endpoints around the existing schemas — `twin.state()`, the disruption event / candidates / decision JSON shapes are stable and documented in `supply_chain_agents/README.md` and Section 7 of the original project plan.
- Build the WebSocket layer for live push updates.
- Wire the approve/reject action to the Orchestrator's `interrupt()`/`Command(resume=...)` pattern — this is already built and tested; C just needs to call it from an endpoint instead of the demo script's hardcoded approval.
- Persist the audit log (SQLite is enough per the original plan).

**For Member D (Frontend/dashboard):**
- Build against the real JSON shapes shown in this doc's audit log examples — a disruption event, 1–3 candidates, a decision with justification and quantified metrics.
- Can start immediately against mocked WebSocket data before C's backend is live, per the original plan's parallel-build approach.
- Note for the approval card UI: some disruption types (supplier issues) will only ever show 1 candidate ("wait") — that's expected, not a bug, and the UI should handle that gracefully rather than assuming there are always 2–3 options to compare.

**For Member A (you) / Member B:**
- Clean up the repo-hygiene items in Section 5 when convenient — none are urgent, but the duplicate-folder issue in particular caused real confusion during testing and is worth fixing before more people are working in this repo.

**For the team on the evaluation plan (Section 10 of the original project plan):**
- Now that the Decision agent is optimizing the right metric, run the 5 disruption scenarios end-to-end and start collecting the actual stockout-rate / cost / response-time / service-level numbers for the agentic-vs-baseline comparison — this hasn't been done yet and is a deliverable in its own right.
