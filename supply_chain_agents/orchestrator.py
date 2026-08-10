"""
Orchestrator agent — Section 4.4, now wired to the REAL digital twin via
twin_adapter.py instead of twin_stub.py.

Sequences: Perceive -> Detect -> Simulate -> Recommend -> (human approval)
-> Act -> Log, per disruption detected each cycle. Built with LangGraph so
human approval is a real graph interrupt/resume, not a hand-rolled if/else.

The one unavoidable structural change from every previous version: LangGraph's
checkpointed state must be JSON-serializable. The live DigitalTwin object is
NOT — it holds a live simpy.Environment with running generator processes.
So the live twin can never go into graph state. Instead:
  - build_graph(twin) takes the live twin as an argument and closes over it —
    every node function is defined INSIDE build_graph so they all share the
    same twin reference.
  - Graph state ("twin" key) only ever holds the READ-ONLY JSON snapshot
    (via twin_adapter.to_pydantic_state(...).model_dump()), used for
    detection and display.
  - Only act() ever calls a real mutating method on the live twin, and only
    after human approval.

What replaced the old twin_stub-based heuristic write-back:
  _apply_action_to_twin() (which GUESSED at state changes) is gone entirely.
  act() now calls the exact same twin.place_order / .reroute_shipment /
  .reallocate_stock that simulation_agent.py already validated during
  simulation, using the chosen candidate's action_params field (added to
  CandidateOutcome specifically for this) — what gets approved is exactly
  what gets applied, not a re-guess.

Resume values for Command(resume=...), unchanged from before:
    True     -> approve the current recommendation
    False    -> reject it; try the next-best alternative if one exists
    "cancel" -> abort the rest of this cycle entirely
"""
from __future__ import annotations
from typing import TypedDict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import time

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from schemas import DisruptionEvent, CandidateOutcome, ActionRecommendation
import twin_adapter
from agents import monitoring_agent, simulation_agent, decision_agent

ApprovalDecision = Union[bool, str]


@dataclass
class OrchestratorConfig:
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    node_timeout_seconds: float = 15.0
    max_alternatives_per_event: int = 3
    days_per_cycle: float = 1.0  # how far to advance simulated time each perceive cycle


DEFAULT_CONFIG = OrchestratorConfig()
SIMULATION_CONFIG = simulation_agent.SimulationConfig()
DECISION_CONFIG = decision_agent.DecisionConfig()


class PipelineState(TypedDict, total=False):
    twin: dict                       # READ-ONLY snapshot only — never the live object
    events: list[dict]
    current_event_idx: int
    recommendation: dict
    recommendations: list[dict]
    approved: Optional[bool]
    approval_decision: Optional[ApprovalDecision]
    audit_log: list[dict]
    failed_agents: list[dict]
    current_stage: str
    rejected_actions_this_event: list[str]
    alternatives_offered_this_event: int
    abort_cycle: bool
    _event: dict
    _outcomes: list[dict]
    _stage_ok: bool


# ---------------------------------------------------------------------------
# Retry + timeout wrapper (unchanged from before — generic, no twin dependency)
# ---------------------------------------------------------------------------

def _resilient_call(fn, args: tuple, node_name: str, config: OrchestratorConfig = DEFAULT_CONFIG):
    last_exc: Exception | None = None
    for attempt in range(config.max_retries + 1):
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(fn, *args)
            result = future.result(timeout=config.node_timeout_seconds)
            pool.shutdown(wait=False)
            return result
        except FutureTimeoutError:
            # Python threads can't be forcibly cancelled -- shutdown(wait=False)
            # does NOT stop fn from still running in the background. Retrying
            # here would start a second concurrent call to fn while the first
            # is still alive; for nodes with shared/mutable state (e.g.
            # monitoring_agent's alert dedup) that lets the abandoned attempt
            # silently corrupt what the retry sees, producing a wrong-but-
            # successful result instead of a visible failure. So: no retry on
            # timeout, fail loudly instead.
            pool.shutdown(wait=False)
            raise RuntimeError(
                f"{node_name} timed out after {config.node_timeout_seconds}s -- "
                f"not retrying (see _resilient_call docstring: a timed-out "
                f"attempt keeps running and can corrupt a retry's result)"
            ) from None
        except Exception as exc:  # noqa: BLE001 - deliberately broad: this is the retry boundary
            pool.shutdown(wait=False)
            last_exc = exc
        if attempt < config.max_retries:
            time.sleep(config.retry_backoff_seconds * (attempt + 1))
    raise RuntimeError(f"{node_name} failed after {config.max_retries + 1} attempt(s): {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Real twin mutation dispatch — replaces the old heuristic write-back
# ---------------------------------------------------------------------------

def _apply_chosen_action(twin, action_params: dict) -> None:
    """Calls the real twin method matching exactly what was simulated and
    approved. Raises on failure (missing/invalid params, unknown supplier,
    etc.) so the caller can log it rather than silently pretend it worked."""
    atype = action_params.get("type", "wait")
    if atype == "reorder":
        twin.place_order(
            supplier_id=action_params["supplier_id"], warehouse_id=action_params["warehouse_id"],
            sku=action_params["sku"], quantity=action_params["quantity"],
            rush=action_params.get("rush", False),
        )
    elif atype == "reroute":
        twin.reroute_shipment(
            shipment_id=action_params["shipment_id"], new_destination=action_params["new_destination"],
            extra_delay_days=action_params.get("extra_delay_days", 1),
        )
    elif atype == "reallocate":
        twin.reallocate_stock(
            from_warehouse=action_params["from_warehouse"], to_warehouse=action_params["to_warehouse"],
            sku=action_params["sku"], quantity=action_params["quantity"],
        )
    elif atype == "wait":
        pass  # genuinely nothing to apply
    else:
        raise ValueError(f"Unrecognized action type in action_params: {atype!r}")


# ---------------------------------------------------------------------------
# Graph construction — every node closes over `twin`, the live object
# ---------------------------------------------------------------------------

def build_graph(twin, orch_config: OrchestratorConfig = DEFAULT_CONFIG):
    """twin: a live object from twin_adapter.build_live_twin() (or any real
    DigitalTwin instance). Held via closure for the lifetime of the returned
    graph — never placed into checkpointed state."""

    def perceive(state: PipelineState) -> PipelineState:
        """No fallback on failure: if this fails after retries, there's
        genuinely nothing else to do this cycle, so the exception propagates
        out of graph.invoke() uncaught — intentional, same as before."""
        def _tick_and_snapshot():
            twin.advance(orch_config.days_per_cycle)
            return twin_adapter.to_pydantic_state(twin.state())

        snapshot = _resilient_call(_tick_and_snapshot, (), "perceive", orch_config)
        return {
            "twin": snapshot.model_dump(),
            "audit_log": state.get("audit_log", []),
            "failed_agents": state.get("failed_agents", []),
            "current_stage": "perceived",
        }

    def detect(state: PipelineState) -> PipelineState:
        from schemas import TwinState
        snapshot = TwinState(**state["twin"])
        try:
            events = _resilient_call(monitoring_agent.run, (snapshot,), "detect", orch_config)
        except Exception as exc:
            failed = state.get("failed_agents", []) + [{"stage": "detect", "error": str(exc), "timestamp": time.time()}]
            return {"events": [], "current_event_idx": 0, "failed_agents": failed, "current_stage": "detect_failed"}
        return {"events": [e.model_dump() for e in events], "current_event_idx": 0, "current_stage": "detected"}

    def simulate(state: PipelineState) -> PipelineState:
        """Calls simulation_agent against the LIVE twin (closure), not the
        read-only snapshot in state — only the live object can actually run
        simulate_candidate()."""
        event = DisruptionEvent(**state["events"][state["current_event_idx"]])
        try:
            outcomes = _resilient_call(simulation_agent.run, (event, twin, SIMULATION_CONFIG), "simulate", orch_config)
        except Exception as exc:
            failed = state.get("failed_agents", []) + [
                {"stage": "simulate", "event_id": event.event_id, "error": str(exc), "timestamp": time.time()}
            ]
            return {"_event": event.model_dump(), "_outcomes": [], "failed_agents": failed,
                    "current_stage": "simulate_failed", "_stage_ok": False}
        return {
            "_outcomes": [o.model_dump() for o in outcomes], "_event": event.model_dump(),
            "current_stage": "simulated", "_stage_ok": True,
            "rejected_actions_this_event": [], "alternatives_offered_this_event": 0,
        }

    def recommend(state: PipelineState) -> PipelineState:
        event = DisruptionEvent(**state["_event"])
        outcomes = [CandidateOutcome(**o) for o in state["_outcomes"]]
        try:
            rec = _resilient_call(decision_agent.run, (event, outcomes, DECISION_CONFIG), "recommend", orch_config)
        except Exception as exc:
            failed = state.get("failed_agents", []) + [
                {"stage": "recommend", "event_id": event.event_id, "error": str(exc), "timestamp": time.time()}
            ]
            return {"failed_agents": failed, "current_stage": "recommend_failed", "_stage_ok": False}
        return {"recommendation": rec.model_dump(), "current_stage": "awaiting_approval", "_stage_ok": True}

    def approve(state: PipelineState) -> Command:
        decision: ApprovalDecision = interrupt({
            "message": "Human approval required",
            "recommendation": state["recommendation"],
        })
        return Command(update={
            "approval_decision": decision,
            "approved": decision is True,
            "current_stage": "decision_received",
        })

    def offer_alternative(state: PipelineState) -> PipelineState:
        event = DisruptionEvent(**state["_event"])
        all_outcomes = [CandidateOutcome(**o) for o in state["_outcomes"]]
        tried = set(state.get("rejected_actions_this_event", [])) | {state["recommendation"]["chosen_action"]}
        remaining = [c for c in all_outcomes if c.action not in tried]

        try:
            new_rec = _resilient_call(decision_agent.run, (event, remaining, DECISION_CONFIG), "recommend_alternative", orch_config)
        except Exception as exc:
            failed = state.get("failed_agents", []) + [
                {"stage": "offer_alternative", "event_id": event.event_id, "error": str(exc), "timestamp": time.time()}
            ]
            return {"approval_decision": False, "approved": False, "failed_agents": failed}

        return {
            "recommendation": new_rec.model_dump(),
            "rejected_actions_this_event": list(tried),
            "alternatives_offered_this_event": state.get("alternatives_offered_this_event", 0) + 1,
            "current_stage": "offering_alternative",
        }

    def act(state: PipelineState) -> PipelineState:
        rec = ActionRecommendation(**state["recommendation"])
        decision = state.get("approval_decision")
        abort = decision == "cancel"
        rec.status = "approved" if decision is True else "rejected"

        twin_snapshot = state["twin"]
        apply_error = None
        if decision is True:
            chosen = next((o for o in state["_outcomes"] if o["action"] == rec.chosen_action), None)
            action_params = chosen["action_params"] if chosen else {"type": "wait"}
            try:
                _resilient_call(_apply_chosen_action, (twin, action_params), "act", orch_config)
                twin_snapshot = twin_adapter.to_pydantic_state(twin.state()).model_dump()
            except Exception as exc:
                # Approved, but the real mutation failed -- must not silently
                # pretend it succeeded. The recommendation stays "approved"
                # (that WAS the human's decision) but the failure is recorded
                # distinctly so it's visible in the audit trail.
                apply_error = str(exc)

        log_entry = {
            "timestamp": time.time(),
            "event": state["_event"],
            "candidates": state["_outcomes"],
            "decision": rec.model_dump(),
            "human_decision": "cancel" if abort else bool(decision),
            "apply_error": apply_error,
        }
        audit_log = state.get("audit_log", []) + [log_entry]
        recommendations = state.get("recommendations", []) + [rec.model_dump()]
        return {
            "recommendation": rec.model_dump(), "recommendations": recommendations, "audit_log": audit_log,
            "twin": twin_snapshot, "abort_cycle": abort, "current_stage": "acted",
        }

    def mark_failed_and_continue(state: PipelineState) -> PipelineState:
        audit_log = state.get("audit_log", []) + [{
            "timestamp": time.time(),
            "event": state.get("_event"),
            "status": "skipped_due_to_agent_failure",
        }]
        return {"audit_log": audit_log, "current_stage": "skipped"}

    def advance_event(state: PipelineState) -> PipelineState:
        return {"current_event_idx": state["current_event_idx"] + 1}

    # ---- Routers ----

    def has_events(state: PipelineState) -> str:
        return "simulate" if state.get("events") else "done"

    def has_outcomes(state: PipelineState) -> str:
        return "recommend" if state.get("_stage_ok") else "failed"

    def has_recommendation(state: PipelineState) -> str:
        return "approve" if state.get("_stage_ok") else "failed"

    def route_after_approval(state: PipelineState) -> str:
        decision = state.get("approval_decision")
        if decision is True or decision == "cancel":
            return "act"
        rec = state["recommendation"]
        tried = set(state.get("rejected_actions_this_event", [])) | {rec["chosen_action"]}
        remaining = [o for o in state["_outcomes"] if o["action"] not in tried]
        offered = state.get("alternatives_offered_this_event", 0)
        if remaining and offered < orch_config.max_alternatives_per_event:
            return "offer_alternative"
        return "act"

    def route_after_advance(state: PipelineState) -> str:
        if state.get("abort_cycle"):
            return "done"
        idx = state["current_event_idx"]
        events = state["events"]
        return "simulate" if idx < len(events) else "done"

    # ---- Wiring ----

    graph = StateGraph(PipelineState)
    graph.add_node("perceive", perceive)
    graph.add_node("detect", detect)
    graph.add_node("simulate", simulate)
    graph.add_node("recommend", recommend)
    graph.add_node("approve", approve)
    graph.add_node("offer_alternative", offer_alternative)
    graph.add_node("act", act)
    graph.add_node("mark_failed_and_continue", mark_failed_and_continue)
    graph.add_node("advance", advance_event)

    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "detect")
    graph.add_conditional_edges("detect", has_events, {"simulate": "simulate", "done": END})
    graph.add_conditional_edges("simulate", has_outcomes, {"recommend": "recommend", "failed": "mark_failed_and_continue"})
    graph.add_conditional_edges("recommend", has_recommendation, {"approve": "approve", "failed": "mark_failed_and_continue"})
    graph.add_conditional_edges("approve", route_after_approval, {"act": "act", "offer_alternative": "offer_alternative"})
    graph.add_edge("offer_alternative", "approve")
    graph.add_edge("act", "advance")
    graph.add_edge("mark_failed_and_continue", "advance")
    graph.add_conditional_edges("advance", route_after_advance, {"simulate": "simulate", "done": END})

    return graph.compile(checkpointer=MemorySaver())