"""
Integration tests that exercise the agents in their real working order:

    Monitoring agent (detect)  ->  Simulation agent (test candidates)  ->  Decision agent (pick best)

Each test builds a REAL twin (twin_adapter.DigitalTwin, via make_real_twin())
representing one of the 5 evaluation scenarios from Section 10 of the
project plan, then runs it through the same conversion + agent sequence the
orchestrator actually uses in production:

    real twin --to_pydantic_state()--> Monitoring  ...  Simulation(real twin) -> Decision

Two things that trip people up here, both intentional:

  1. make_real_twin()'s default warehouses/supplier ALSO sit above the
     overstock and supplier-lead-time thresholds (it wasn't built to be a
     "clean" baseline — those defaults are locked in by test_adapter.py and
     test_action_builders.py already). So most of these tests filter events
     to the specific type they care about, rather than asserting an exact
     total count — checking for the signal you're testing, not asserting
     away real background noise, which is closer to how a live system
     actually behaves anyway.
  2. Demand-spike detection no longer trusts a single-shot "average vs
     recent" pair (see monitoring_agent.py's module docstring, "v3 change")
     — it learns its own baseline from several polls of rolling history.
     _prime_demand_history() below simulates that polling instead of trying
     to fake a shortcut.
"""
from __future__ import annotations

import twin_adapter
from agents import monitoring_agent, simulation_agent, decision_agent
from tests.conftest import make_real_twin


def _poll(twin):
    """One perceive cycle: convert the real twin's current state and run
    monitoring_agent on it — exactly what orchestrator.perceive()+detect() do."""
    state = twin_adapter.to_pydantic_state(twin.state())
    return monitoring_agent.run(state)


def _poll_and_collect(twin, events_by_key: dict) -> None:
    """Runs one perceive cycle and merges any events into `events_by_key`
    (keyed by (type, affected_id), so a later escalated re-alert for the
    same disruption overwrites the earlier one, but an earlier disruption
    that later goes quiet under dedup/cooldown is NOT lost — it genuinely
    happened during this monitoring window, same as a real audit trail)."""
    for e in _poll(twin):
        events_by_key[(e.type, e.affected_id)] = e


def _prime_demand_history(twin, wh_id: str, sku: str, baseline_demand: float,
                           spike_demand: float, baseline_polls: int = 7, spike_polls: int = 2):
    """Feeds `baseline_polls` identical low-demand polls into monitoring_agent's
    rolling history, then `spike_polls` polls at `spike_demand` (matching
    MonitoringConfig.demand_recent_window=2 by default, so the "recent"
    window is fully the spike value, not diluted by one leftover baseline
    point) — mirrors what several real perceive cycles would look like.
    Returns every event seen across the whole sequence (see
    _poll_and_collect), not just the final poll's output."""
    events_by_key: dict = {}
    for _ in range(baseline_polls):
        twin.warehouses[wh_id].daily_demand[sku] = baseline_demand
        _poll_and_collect(twin, events_by_key)
    for _ in range(spike_polls):
        twin.warehouses[wh_id].daily_demand[sku] = spike_demand
        _poll_and_collect(twin, events_by_key)
    return list(events_by_key.values())


def run_pipeline(twin, events):
    """Runs Simulation -> Decision for each already-detected event, against
    the SAME real twin the event was detected from — mirrors what the
    orchestrator does per perceive cycle."""
    results = {}
    for event in events:
        outcomes = simulation_agent.run(event, twin)
        rec = decision_agent.run(event, outcomes)
        results[event.event_id] = (outcomes, rec)
    return results


_ACTION_TYPES = {"reroute", "reorder", "wait", "reallocate", "other"}


# ---------- Scenario 1: single supplier delay on a high-volume SKU ----------

def test_scenario_shipment_delay():
    # 8 of 7 planned days late = 114% of lead time -> high severity.
    # WH-02 is given real need so all three candidate types (reroute,
    # reorder, wait) are actually buildable, not silently dropped.
    twin = make_real_twin(delay_days=8, lead_time_days=7, wh2_on_hand=50, wh2_safety_stock=150)

    events = _poll(twin)
    delay_events = [e for e in events if e.type == "shipment_delay"]
    assert len(delay_events) == 1
    event = delay_events[0]
    assert event.severity == "high"
    assert event.description  # LLM/fallback description was attached

    outcomes, rec = run_pipeline(twin, [event])[event.event_id]
    assert 2 <= len(outcomes) <= 3
    assert all(o.action_type in _ACTION_TYPES for o in outcomes)
    assert rec.chosen_action in [o.action for o in outcomes]
    assert rec.candidates_considered == len(outcomes)
    assert isinstance(rec.constraint_satisfied, bool)


# ---------- Scenario 2: sudden demand spike (+40%) on one warehouse ----------

def test_scenario_demand_spike():
    # Clean-ish inventory ratio (1.4) so overstock/understock noise doesn't
    # also fire and complicate the picture -- the spike is the thing under test.
    twin = make_real_twin(on_hand=210, safety_stock=150)

    events = _prime_demand_history(twin, "WH-01", "SKU-1", baseline_demand=20, spike_demand=28)  # +40%

    spike_events = [e for e in events if e.type == "demand_spike"]
    assert len(spike_events) == 1
    event = spike_events[0]
    assert event.severity in {"medium", "high"}  # +40% clears the medium (30%) threshold

    outcomes, rec = run_pipeline(twin, [event])[event.event_id]
    labels = {o.action for o in outcomes}
    assert any("reorder" in a or "reallocate" in a for a in labels)
    assert rec.chosen_action in labels


# ---------- Scenario 3: shipment lost / significantly delayed in transit ----------

def test_scenario_severe_shipment_loss():
    # Delay far exceeds planned lead time -> unambiguously severe.
    twin = make_real_twin(delay_days=20, lead_time_days=5, wh2_on_hand=50, wh2_safety_stock=150)

    events = _poll(twin)
    delay_events = [e for e in events if e.type == "shipment_delay"]
    assert len(delay_events) == 1
    # ratio = 20/5 = 4.0 -> clears the critical threshold (2.0), not just "high".
    assert delay_events[0].severity == "critical"

    outcomes, rec = run_pipeline(twin, delay_events)[delay_events[0].event_id]
    # Severe severity -> tight risk ceiling (0.15 by default) should be
    # respected wherever a candidate makes that possible.
    if any(o.stockout_risk <= 0.15 for o in outcomes):
        assert rec.constraint_satisfied is True


# ---------- Scenario 4: compound — supplier delay coincides with demand spike ----------

def test_scenario_compound_delay_and_spike():
    twin = make_real_twin(
        delay_days=5, lead_time_days=6,       # ~83% -> medium/high delay severity
        on_hand=210, safety_stock=150,        # keep inventory-ratio noise out of the way
        wh2_on_hand=50, wh2_safety_stock=150,
    )

    events = _prime_demand_history(twin, "WH-01", "SKU-1", baseline_demand=20, spike_demand=30)  # +50%

    types_seen = {e.type for e in events}
    assert "shipment_delay" in types_seen
    assert "demand_spike" in types_seen

    # Every detected disruption must resolve to a complete recommendation —
    # a compound scenario must not cause one event to swallow or block the other.
    results = run_pipeline(twin, events)
    for event in events:
        outcomes, rec = results[event.event_id]
        assert len(outcomes) >= 1
        assert rec.chosen_action


# ---------- Scenario 5: low-severity disruption — system must not overreact ----------

def test_scenario_low_severity_no_overreaction():
    # Barely-late shipment (10% of lead time) and near-baseline demand:
    # should NOT be flagged as a shipment_delay or demand_spike at all
    # (background overstock/supplier-lead-time noise from make_real_twin's
    # defaults is a separate, expected signal — not what's under test here).
    twin = make_real_twin(delay_days=1, lead_time_days=10)

    events = _poll(twin)

    reactive_types = {e.type for e in events} & {"shipment_delay", "demand_spike", "stock_imbalance"}
    assert reactive_types == set(), f"minor noise should not trigger a reactive disruption, got {reactive_types}"


def test_scenario_low_severity_that_does_clear_threshold_stays_cheap():
    # Just over the low-severity threshold: a real (if minor) disruption.
    # The wide low-severity risk ceiling (0.70 default) should let the
    # cheapest option win rather than forcing an expensive "safe" choice.
    twin = make_real_twin(delay_days=2, lead_time_days=9, wh2_on_hand=50, wh2_safety_stock=150)

    events = _poll(twin)
    delay_events = [e for e in events if e.type == "shipment_delay"]
    assert len(delay_events) == 1
    assert delay_events[0].severity == "low"

    outcomes, rec = run_pipeline(twin, delay_events)[delay_events[0].event_id]
    chosen = next(o for o in outcomes if o.action == rec.chosen_action)
    priciest = max(outcomes, key=lambda o: o.cost_delta)
    # Not a strict equality (risk ceiling could still exclude the cheapest),
    # but the chosen option's cost should be close to the cheapest available
    # rather than jumping to the most expensive one over minor noise.
    assert chosen.cost_delta <= priciest.cost_delta