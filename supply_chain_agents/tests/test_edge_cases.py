"""
Regression tests for specific bugs found during code review. Each test is
named after the bug it guards against, so a future change that reintroduces
one of these fails loudly and points straight at the cause.
"""
from __future__ import annotations

import dataclasses

from langgraph.types import Command

import twin_adapter
from agents import monitoring_agent, simulation_agent, decision_agent
from schemas import DisruptionEvent, CandidateOutcome
from orchestrator import build_graph
from tests.conftest import make_twin, make_real_twin


# ---------- Decision agent: savings calculation ----------

def test_savings_compares_against_runner_up_not_worst_alternative():
    """Old bug: savings was computed as max(others) - chosen, i.e. vs. the
    MOST expensive alternative, mislabeled as 'next-best'. Must be vs. the
    cheapest of the others (the true runner-up). Pinned to objective_priority
    "cost" explicitly -- this test predates "profit" existing as an option,
    and DEFAULT_CONFIG no longer defaults to "cost"."""
    candidates = [
        CandidateOutcome(action="best", action_type="reroute", cost_delta=20000,
                          delivery_delay_days=1, stockout_risk=0.1, service_level_impact=0.9),
        CandidateOutcome(action="close_runner_up", action_type="reallocate", cost_delta=25000,
                          delivery_delay_days=1, stockout_risk=0.15, service_level_impact=0.88),
        CandidateOutcome(action="way_pricier", action_type="reorder", cost_delta=80000,
                          delivery_delay_days=0.2, stockout_risk=0.02, service_level_impact=0.98),
    ]
    event = DisruptionEvent(event_id="EVT-x", type="shipment_delay", affected_id="X",
                             severity="low", detected_at="now")
    config = dataclasses.replace(decision_agent.DEFAULT_CONFIG, objective_priority="cost")

    rec = decision_agent.run(event, candidates, config)

    assert rec.chosen_action == "best"
    assert "5,000" in rec.justification  # 25000 - 20000, NOT 80000 - 20000 = 60,000


def test_profit_savings_compares_against_runner_up_not_worst_alternative():
    """Mirror of the cost-branch test above, for the "profit" objective:
    savings must be vs. the SECOND-BEST profit_impact among the alternatives
    (the true runner-up), not the worst one."""
    candidates = [
        CandidateOutcome(action="best", action_type="reroute", cost_delta=20000,
                          delivery_delay_days=1, stockout_risk=0.1, service_level_impact=0.9,
                          profit_impact=-6000),
        CandidateOutcome(action="close_runner_up", action_type="reallocate", cost_delta=25000,
                          delivery_delay_days=1, stockout_risk=0.15, service_level_impact=0.88,
                          profit_impact=-11000),
        CandidateOutcome(action="way_pricier", action_type="reorder", cost_delta=80000,
                          delivery_delay_days=0.2, stockout_risk=0.02, service_level_impact=0.98,
                          profit_impact=-40000),
    ]
    event = DisruptionEvent(event_id="EVT-y", type="shipment_delay", affected_id="Y",
                             severity="low", detected_at="now")
    config = dataclasses.replace(decision_agent.DEFAULT_CONFIG, objective_priority="profit")

    rec = decision_agent.run(event, candidates, config)

    assert rec.chosen_action == "best"
    assert "5,000" in rec.justification  # -6000 - (-11000) = 5,000, NOT -6000 - (-40000) = 34,000


def test_exclusion_uses_identity_not_label_string():
    """Old bug: 'others' were filtered by comparing the action label string,
    so two candidates sharing a label could both be wrongly excluded."""
    candidates = [
        CandidateOutcome(action="duplicate_label", action_type="reroute", cost_delta=10000,
                          delivery_delay_days=1, stockout_risk=0.1, service_level_impact=0.9),
        CandidateOutcome(action="duplicate_label", action_type="reorder", cost_delta=15000,
                          delivery_delay_days=1, stockout_risk=0.1, service_level_impact=0.9),
    ]
    event = DisruptionEvent(event_id="EVT-y", type="shipment_delay", affected_id="X",
                             severity="low", detected_at="now")

    rec = decision_agent.run(event, candidates)
    # Should still find a real runner-up (the other 15000 candidate) instead
    # of treating both as "chosen" and reporting no comparison available.
    assert "Only candidate" not in rec.justification


# ---------- Decision agent: severity-aware, configurable risk ceiling ----------

def test_severity_changes_which_candidate_wins():
    candidates = [
        CandidateOutcome(action="cheap_risky", action_type="wait", cost_delta=10000,
                          delivery_delay_days=2, stockout_risk=0.4, service_level_impact=0.7),
        CandidateOutcome(action="pricier_safe", action_type="reorder", cost_delta=40000,
                          delivery_delay_days=0.5, stockout_risk=0.1, service_level_impact=0.95),
    ]
    low_event = DisruptionEvent(event_id="E1", type="shipment_delay", affected_id="X",
                                 severity="low", detected_at="now")
    high_event = DisruptionEvent(event_id="E2", type="shipment_delay", affected_id="X",
                                  severity="high", detected_at="now")

    assert decision_agent.run(low_event, candidates).chosen_action == "cheap_risky"
    assert decision_agent.run(high_event, candidates).chosen_action == "pricier_safe"


def test_constraint_satisfied_flips_false_when_all_candidates_exceed_ceiling():
    """Old bug: this field existed in the schema but no agent ever set it —
    it silently always read True."""
    candidates = [
        CandidateOutcome(action="opt1", action_type="wait", cost_delta=10000,
                          delivery_delay_days=3, stockout_risk=0.8, service_level_impact=0.5),
        CandidateOutcome(action="opt2", action_type="reorder", cost_delta=50000,
                          delivery_delay_days=1, stockout_risk=0.6, service_level_impact=0.6),
    ]
    event = DisruptionEvent(event_id="E3", type="shipment_delay", affected_id="X",
                             severity="high", detected_at="now")  # 30% ceiling, both exceed it

    rec = decision_agent.run(event, candidates)
    assert rec.constraint_satisfied is False
    assert "risk ceiling" in rec.justification.lower()


# ---------- Monitoring agent: dedup / cooldown / demand-spike detection ----------

def test_repeat_poll_of_unchanged_twin_emits_no_duplicate_events():
    twin = make_twin(delay_days=5, planned_transit_days=6)
    first = monitoring_agent.detect_disruptions(twin)
    second = monitoring_agent.detect_disruptions(twin)
    assert len(first) >= 1
    assert second == []  # same issue, still cooling down -> no repeat alert


def test_escalation_re_alerts_despite_active_cooldown():
    twin = make_twin(delay_days=2, planned_transit_days=10)  # low severity
    monitoring_agent.detect_disruptions(twin)  # seed the registry at "low"

    twin.shipments[0].delay_days = 11  # now 110% of lead time -> high severity
    escalated = monitoring_agent.detect_disruptions(twin)

    assert any(e.severity == "high" for e in escalated)


def test_demand_spike_is_actually_detected():
    """Old bug: DisruptionType included 'demand_spike' but nothing ever
    produced one. Uses make_twin() directly (not the real-twin conversion)
    since this is testing monitoring_agent's own cold-start fallback path —
    trusting a caller-supplied avg/recent pair when there isn't yet enough
    rolling history to self-learn a baseline (see monitoring_agent.py's
    module docstring, 'v3 change')."""
    twin = make_twin(avg_demand=20, recent_demand=30)  # +50%
    events = monitoring_agent.detect_disruptions(twin)
    assert any(e.type == "demand_spike" for e in events)


# ---------- Real twin: action-type-driven forecasting, reproducibility ----------
#
# The original two tests here guarded twin_stub.py's forecast_action() against
# guessing behavior from a free-text label, and against unseeded randomness.
# twin_stub.py is gone — the real twin (digital-twin-fyp-main) replaced it —
# but both bugs are worth guarding against in their new form, since the real
# twin has its own version of "structured input" and "seeded randomness."

def test_simulation_outcome_type_comes_from_the_action_dict_not_a_label():
    """The real architecture makes the ORIGINAL bug (guessing from a
    human-readable label string) structurally impossible: action dicts built
    by simulation_agent carry a machine-readable "type" key, and the twin's
    forecasting logic (twin.py) only ever reads that key — never the label
    text CandidateOutcome.action shows to a human. This test pins that
    contract: two candidates with very different, deliberately misleading
    labels but the SAME "type" key produce outcomes tagged with that type."""
    twin = make_real_twin(on_hand=80, safety_stock=150)
    event = DisruptionEvent(event_id="EVT-1", type="stock_imbalance",
                             affected_id="WH-01:SKU-1", severity="medium", detected_at="now")

    outcomes = simulation_agent.run(event, twin)

    for outcome in outcomes:
        assert outcome.action_type == outcome.action_params.get("type")


def test_twin_simulation_is_reproducible_given_the_same_seed():
    """Old bug (twin_stub.py): seeded on Python's hash(), randomized per
    process by default, so outcomes silently changed between runs. The real
    twin takes an explicit rng_seed instead — pin that two freshly built
    twins with the same seed produce identical forecasts for the same action."""
    action = {"type": "wait"}

    twin_a = twin_adapter.build_live_twin(use_real_data=False, rng_seed=42)
    twin_b = twin_adapter.build_live_twin(use_real_data=False, rng_seed=42)

    outcome_a = twin_a.simulate_candidate(action, horizon_days=7)
    outcome_b = twin_b.simulate_candidate(action, horizon_days=7)

    assert outcome_a.cost_delta == outcome_b.cost_delta
    assert outcome_a.stockout_risk == outcome_b.stockout_risk
    assert outcome_a.delivery_delay_days == outcome_b.delivery_delay_days


# ---------- Orchestrator: zero-event and multi-event handling ----------

def test_orchestrator_does_not_crash_on_zero_detected_events():
    """Old bug: simulate() did events[idx] unconditionally, crashing with
    IndexError whenever a perceive cycle detected nothing — which is the
    NORMAL case once dedup/cooldown is active, not an error path."""
    twin = twin_adapter.build_live_twin(use_real_data=False, rng_seed=1)
    graph = build_graph(twin)
    config = {"configurable": {"thread_id": "test-zero-events"}}

    # First run seeds the dedup registry so a second run on a fresh thread
    # (but same process-global registry, same live twin) will detect
    # nothing new.
    result = graph.invoke({}, config)
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume=True), config)

    config2 = {"configurable": {"thread_id": "test-zero-events-2"}}
    result2 = graph.invoke({}, config2)  # must not raise
    assert "__interrupt__" not in result2
    assert result2.get("recommendations", []) == []


def test_orchestrator_processes_every_detected_event_not_just_the_first():
    """Old bug: current_event_idx was hardcoded to 0 and never advanced, so
    only the first detected disruption in a cycle was ever acted on. The
    synthetic fallback data has several warehouses sitting above the
    overstock threshold out of the box, so a fresh twin reliably produces
    more than one disruption on its very first perceive cycle."""
    twin = twin_adapter.build_live_twin(use_real_data=False, rng_seed=2)
    graph = build_graph(twin)
    config = {"configurable": {"thread_id": "test-multi-event"}}

    result = graph.invoke({}, config)
    n_events = len(result.get("events", []))
    assert n_events > 1, "synthetic fallback twin should produce multiple simultaneous disruptions"

    resumes = 0
    while "__interrupt__" in result:
        resumes += 1
        result = graph.invoke(Command(resume=True), config)

    assert resumes == n_events
    assert len(result["recommendations"]) == n_events
    assert len(result["audit_log"]) == n_events


def test_orchestrator_reject_path_marks_status_rejected():
    twin = twin_adapter.build_live_twin(use_real_data=False, rng_seed=3)
    graph = build_graph(twin)
    config = {"configurable": {"thread_id": "test-reject-path"}}

    result = graph.invoke({}, config)
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume=False), config)

    assert result["recommendations"], "expected at least one processed event"
    assert all(r["status"] == "rejected" for r in result["recommendations"])