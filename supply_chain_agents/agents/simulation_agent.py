"""
Simulation agent — Section 4.2, rebuilt against the REAL digital twin
(digital-twin-fyp-main), not twin_stub.py.

Input:  DisruptionEvent (our pydantic type) + a live twin-like object
Output: list[CandidateOutcome]

REQUIRES one schemas.py addition beyond what monitoring_agent.py already
needed: CandidateOutcome gets a new field:
    action_params: dict = {}
This carries the exact structured action (supplier_id, quantity, etc.) that
produced each outcome, so the orchestrator can later replay the SAME action
for real via the twin's mutation methods, instead of guessing.

Why this is a bigger rewrite than monitoring/decision agent, and why:
------------------------------------------------------------------------
The stub's forecast_action(label, action_type, affected_id) accepted a bare
label. The REAL twin's twin.simulate_candidate(action, horizon_days) needs
a COMPLETE, valid action dict per type:
    {"type": "reorder",    "supplier_id": ..., "warehouse_id": ..., "sku": ...,
                            "quantity": ..., "rush": bool}
    {"type": "reroute",    "shipment_id": ..., "new_destination": ...}
    {"type": "reallocate", "from_warehouse": ..., "to_warehouse": ..., "sku": ...,
                            "quantity": ...}
    {"type": "wait"}
Nothing previously picked those parameters -- an LLM can't be trusted to
invent a valid supplier_id or a quantity that won't KeyError against the
twin, so this agent now deterministically builds them by reading the live
twin's actual warehouses/shipments/suppliers. The LLM's role shrinks
accordingly: it no longer proposes free-text actions at all -- the
plausible action TYPES for a disruption are now a fixed, honest mapping
(only 4 real action types exist), and every candidate that mapping produces
is only offered if valid parameters can actually be found (e.g. a reorder
is dropped entirely if no supplier can be identified, rather than guessed).

Duck-typed, not imported: this file never imports the twin's classes
directly (avoids the schemas.py naming collision between the two projects --
that's twin_adapter.py's job to solve). It only calls twin.simulate_candidate(...)
and reads twin.warehouses / twin.shipments / twin.suppliers as plain
dict-of-objects, which works against any object providing that shape.

Parallel simulation: candidates are still run concurrently (ThreadPoolExecutor).
Verified safe against the real twin -- simulate_candidate() forks internally
(deep-copies state into a brand-new SimPy environment) and never mutates the
calling twin, so concurrent calls from multiple threads don't race.

What changed in scenario/horizon/what-if support, and why:
  - run_with_horizons() now varies the REAL horizon_days parameter (1 / 7 / 30
    days) instead of post-hoc multiplying fake numbers -- this is honest
    simulation, not a guess, now that a real simulate_candidate() exists.
  - Best/expected/worst-case scenario bands are DROPPED from this version --
    doing that honestly would need forking the twin with a different
    demand_variability setting, which means constructing a whole new twin
    instance (out of scope for a duck-typed agent file). Flagged as a
    possible future addition once twin_adapter.py exists to do that safely.
  - what_if() now perturbs real ACTION PARAMETERS (no rush, double quantity,
    extended horizon) and re-simulates for real, instead of scaling numbers
    after the fact.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time

from schemas import DisruptionEvent, CandidateOutcome

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    horizon_days_default: int = 7
    min_reorder_quantity: int = 10
    min_reallocate_quantity: int = 10
    rush_severities: set = field(default_factory=lambda: {"high", "critical"})
    enable_parallel: bool = True
    history_max_len: int = 50

    horizon_bands: dict[str, int] = field(default_factory=lambda: {
        "near_term": 1, "medium_term": 7, "long_term": 30,
    })

    # Heuristic constant for the profit_impact estimate (see monitoring/decision
    # agents for the same pattern) -- illustrative until the twin computes this itself.
    lost_sales_value: float = 50_000.0


DEFAULT_CONFIG = SimulationConfig()

# Rolling in-memory simulation history: {event_id, timestamp, kind, outcomes}
_simulation_history: list[dict] = []


def reset_simulation_history() -> None:
    _simulation_history.clear()


def get_simulation_history(event_id: str | None = None) -> list[dict]:
    if event_id is None:
        return list(_simulation_history)
    return [r for r in _simulation_history if r["event_id"] == event_id]


def _record_history(event_id: str, kind: str, outcomes: list[CandidateOutcome], config: SimulationConfig) -> None:
    _simulation_history.append({
        "event_id": event_id,
        "timestamp": time.time(),
        "kind": kind,
        "outcomes": [o.model_dump() for o in outcomes],
    })
    if len(_simulation_history) > config.history_max_len:
        del _simulation_history[: len(_simulation_history) - config.history_max_len]


# ---------------------------------------------------------------------------
# Which action TYPES are even plausible for each disruption type
# (deterministic -- only 4 real action types exist, no point asking an LLM)
# ---------------------------------------------------------------------------

_CANDIDATE_TYPES_BY_EVENT = {
    "shipment_delay": ["reroute", "reorder", "wait"],
    "stock_imbalance": ["reorder", "reallocate", "wait"],       # understock
    "overstock": ["reallocate", "wait"],                         # never reorder MORE of a surplus
    "dead_stock": ["reallocate", "wait"],                        # move slow stock elsewhere, don't reorder more
    "fast_moving": ["reorder", "wait"],                          # get ahead of sustained demand growth
    "demand_spike": ["reorder", "reallocate", "wait"],
    "lost_shipment": ["reorder", "wait"],                        # nothing left to reroute
    "route_deviation": ["reroute", "wait"],
    # Supplier-level events have no specific (warehouse, sku) target to build a
    # reorder/reallocate/reroute action against -- offering only "wait" here is
    # an honest limitation, not a bug. Real remediation (e.g. switching a
    # primary supplier for a SKU) would need procurement logic not yet built.
    "supplier_lead_time": ["wait"],
    "supplier_reliability": ["wait"],
}


def _parse_wh_sku(affected_id: str) -> tuple[str, str] | None:
    if ":" not in affected_id:
        return None
    wh_id, sku = affected_id.split(":", 1)
    return wh_id, sku


# ---------------------------------------------------------------------------
# Action-dict builders -- read the live twin, construct complete valid params
# ---------------------------------------------------------------------------

def _find_supplier_for(twin, warehouse_id: str, sku: str) -> str | None:
    """Prefers a supplier with existing shipment history to this warehouse+sku;
    falls back to the most reliable known supplier."""
    for shp in twin.shipments.values():
        if shp.destination == warehouse_id and shp.sku == sku and shp.origin in twin.suppliers:
            return shp.origin
    if twin.suppliers:
        return max(twin.suppliers.values(), key=lambda s: s.reliability_score).id
    return None


def _build_reorder_action(twin, warehouse_id: str, sku: str, severity: str, config: SimulationConfig) -> dict | None:
    wh = twin.warehouses.get(warehouse_id)
    if wh is None:
        return None
    supplier_id = _find_supplier_for(twin, warehouse_id, sku)
    if supplier_id is None:
        return None
    supplier = twin.suppliers[supplier_id]
    on_hand = wh.inventory.get(sku, 0)
    safety = wh.safety_stock.get(sku, 0)
    daily = wh.daily_demand.get(sku, 0)
    deficit = max(safety - on_hand, 0)
    lead_time_cover = daily * supplier.lead_time_days
    quantity = max(round(deficit + lead_time_cover), config.min_reorder_quantity)
    return {
        "type": "reorder", "supplier_id": supplier_id, "warehouse_id": warehouse_id,
        "sku": sku, "quantity": quantity, "rush": severity in config.rush_severities,
    }


def _build_reallocate_action(twin, affected_warehouse_id: str, sku: str, direction: str, config: SimulationConfig) -> dict | None:
    """direction='in' moves stock INTO the affected warehouse (for shortages);
    direction='out' moves it OUT (for overstock/dead stock)."""
    if direction == "in":
        to_id = affected_warehouse_id
        to_wh = twin.warehouses.get(to_id)
        if to_wh is None:
            return None
        need = max(to_wh.safety_stock.get(sku, 0) - to_wh.inventory.get(sku, 0), 0) or config.min_reallocate_quantity
        best_from, best_surplus = None, 0
        for wh_id, wh in twin.warehouses.items():
            if wh_id == to_id:
                continue
            surplus = wh.inventory.get(sku, 0) - wh.safety_stock.get(sku, 0)
            if surplus > best_surplus:
                best_from, best_surplus = wh_id, surplus
        if best_from is None:
            return None
        quantity = max(1, min(need, best_surplus))
        return {"type": "reallocate", "from_warehouse": best_from, "to_warehouse": to_id, "sku": sku, "quantity": quantity}

    else:  # "out"
        from_id = affected_warehouse_id
        from_wh = twin.warehouses.get(from_id)
        if from_wh is None:
            return None
        surplus = max(from_wh.inventory.get(sku, 0) - from_wh.safety_stock.get(sku, 0), 0) or config.min_reallocate_quantity
        best_to, best_need = None, 0
        for wh_id, wh in twin.warehouses.items():
            if wh_id == from_id:
                continue
            need = wh.safety_stock.get(sku, 0) - wh.inventory.get(sku, 0)
            if need > best_need:
                best_to, best_need = wh_id, need
        if best_to is None:
            return None
        quantity = max(1, min(surplus, best_need)) if best_need > 0 else min(surplus, config.min_reallocate_quantity)
        return {"type": "reallocate", "from_warehouse": from_id, "to_warehouse": best_to, "sku": sku, "quantity": quantity}


def _build_reroute_action(twin, shipment_id: str) -> dict | None:
    shp = twin.shipments.get(shipment_id)
    if shp is None:
        return None
    current_dest = shp.destination
    sku = shp.sku
    best_alt, best_need = None, -1
    for wh_id, wh in twin.warehouses.items():
        if wh_id == current_dest:
            continue
        need = wh.safety_stock.get(sku, 0) - wh.inventory.get(sku, 0)
        if need > best_need:
            best_alt, best_need = wh_id, need
    if best_alt is None:
        return None
    return {"type": "reroute", "shipment_id": shipment_id, "new_destination": best_alt}


def build_candidate_actions(event: DisruptionEvent, twin, config: SimulationConfig = DEFAULT_CONFIG) -> list[dict]:
    """The deterministic replacement for the old LLM-driven propose_candidates().
    Returns only actions that could actually be built with valid parameters --
    silently dropping a type (e.g. no supplier found) is safer than sending
    the twin a half-formed action dict that would KeyError."""
    types = _CANDIDATE_TYPES_BY_EVENT.get(event.type, ["wait"])
    actions: list[dict] = []

    parsed = _parse_wh_sku(event.affected_id)
    is_shipment_event = event.type in ("shipment_delay", "lost_shipment", "route_deviation")

    for atype in types:
        if atype == "wait":
            actions.append({"type": "wait"})
            continue

        if atype == "reorder":
            if is_shipment_event:
                shp = twin.shipments.get(event.affected_id)
                if shp is None:
                    continue
                action = _build_reorder_action(twin, shp.destination, shp.sku, event.severity, config)
            elif parsed:
                wh_id, sku = parsed
                action = _build_reorder_action(twin, wh_id, sku, event.severity, config)
            else:
                action = None
            if action:
                actions.append(action)

        elif atype == "reallocate":
            if not parsed:
                continue
            wh_id, sku = parsed
            direction = "out" if event.type in ("overstock", "dead_stock") else "in"
            action = _build_reallocate_action(twin, wh_id, sku, direction, config)
            if action:
                actions.append(action)

        elif atype == "reroute":
            shipment_id = event.affected_id if is_shipment_event else None
            action = _build_reroute_action(twin, shipment_id) if shipment_id else None
            if action:
                actions.append(action)

    if not actions:
        actions.append({"type": "wait"})  # always leave at least one candidate
    return actions


# ---------------------------------------------------------------------------
# Human-readable labels + extra metric estimation
# ---------------------------------------------------------------------------

def _build_label(action: dict) -> str:
    t = action["type"]
    if t == "reorder":
        return f"reorder_{action['quantity']}_{action['sku']}_from_{action['supplier_id']}_to_{action['warehouse_id']}"
    if t == "reroute":
        return f"reroute_{action['shipment_id']}_to_{action['new_destination']}"
    if t == "reallocate":
        return f"reallocate_{action['quantity']}_{action['sku']}_{action['from_warehouse']}_to_{action['to_warehouse']}"
    return "wait_and_absorb"


def _estimate_extra_metrics(action_type: str, cost_delta: float, service_level_impact: float, config: SimulationConfig) -> dict:
    inventory_impact = {"reorder": 1.0, "reallocate": 0.0, "reroute": 0.0, "wait": 0.0}.get(action_type, 0.0) * abs(cost_delta) / 50.0
    lost_sales_effect = (1 - service_level_impact) * config.lost_sales_value
    profit_impact = -cost_delta - lost_sales_effect
    resource_utilization = {"reorder": 0.6, "reroute": 0.5, "reallocate": 0.4, "wait": 0.1}.get(action_type, 0.5)
    return {
        "inventory_impact": round(inventory_impact, 1),
        "profit_impact": round(profit_impact, 2),
        "resource_utilization": resource_utilization,
    }


def _to_candidate_outcome(action: dict, outcome, config: SimulationConfig) -> CandidateOutcome:
    """Maps the real twin's ActionOutcome (duck-typed: .action, .cost_delta,
    .delivery_delay_days, .stockout_risk, .service_level) into our pydantic
    CandidateOutcome. Note the field rename: service_level -> service_level_impact."""
    extra = _estimate_extra_metrics(action["type"], outcome.cost_delta, outcome.service_level, config)
    return CandidateOutcome(
        action=_build_label(action),
        action_type=action["type"],
        cost_delta=outcome.cost_delta,
        delivery_delay_days=outcome.delivery_delay_days,
        stockout_risk=outcome.stockout_risk,
        service_level_impact=outcome.service_level,
        action_params=dict(action),
        **extra,
    )


# ---------------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------------

def _simulate_actions(twin, actions: list[dict], horizon_days: int, config: SimulationConfig) -> list[CandidateOutcome]:
    def run_one(action: dict) -> CandidateOutcome:
        outcome = twin.simulate_candidate(action, horizon_days=horizon_days)
        return _to_candidate_outcome(action, outcome, config)

    if not config.enable_parallel or len(actions) <= 1:
        return [run_one(a) for a in actions]

    with ThreadPoolExecutor(max_workers=len(actions)) as pool:
        # NOT using `with` for the executor's blocking shutdown here matters
        # less than in the orchestrator's timeout wrapper, since these calls
        # are expected to complete quickly and aren't subject to a timeout --
        # but the pattern is deliberately identical for consistency.
        futures = [pool.submit(run_one, a) for a in actions]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run(event: DisruptionEvent, twin, config: SimulationConfig = DEFAULT_CONFIG) -> list[CandidateOutcome]:
    """twin is the LIVE twin object (or anything duck-typed the same way --
    needs .warehouses/.shipments/.suppliers as dict-of-objects and a
    .simulate_candidate(action, horizon_days) method). Never mutated: verified
    by comparing a snapshot before and after this function runs."""
    before = twin.state().to_dict()

    actions = build_candidate_actions(event, twin, config)
    outcomes = _simulate_actions(twin, actions, config.horizon_days_default, config)
    _record_history(event.event_id, "single", outcomes, config)

    if twin.state().to_dict() != before:
        raise RuntimeError(
            "Simulation agent's run against the live twin left it mutated — this must "
            "never happen. simulate_candidate() is supposed to fork internally."
        )
    return outcomes


def run_with_horizons(event: DisruptionEvent, twin, config: SimulationConfig = DEFAULT_CONFIG) -> dict[str, list[CandidateOutcome]]:
    """Real forward projection at different horizons (1 / 7 / 30 days by
    default) — genuinely different simulated outcomes, not scaled numbers."""
    actions = build_candidate_actions(event, twin, config)
    results = {}
    for band_name, days in config.horizon_bands.items():
        outcomes = _simulate_actions(twin, actions, days, config)
        _record_history(event.event_id, band_name, outcomes, config)
        results[band_name] = outcomes
    return results


def what_if(event: DisruptionEvent, twin, perturbation: str, config: SimulationConfig = DEFAULT_CONFIG) -> list[CandidateOutcome]:
    """Perturbs real action parameters and re-simulates for real.
    Available: 'no_rush', 'double_quantity', 'extended_horizon'."""
    actions = build_candidate_actions(event, twin, config)
    horizon = config.horizon_days_default

    if perturbation == "no_rush":
        actions = [{**a, "rush": False} if a["type"] == "reorder" else a for a in actions]
    elif perturbation == "double_quantity":
        actions = [{**a, "quantity": a["quantity"] * 2} if "quantity" in a else a for a in actions]
    elif perturbation == "extended_horizon":
        horizon = horizon * 3
    else:
        raise ValueError(f"Unknown what-if perturbation {perturbation!r}. Available: no_rush, double_quantity, extended_horizon")

    outcomes = _simulate_actions(twin, actions, horizon, config)
    _record_history(event.event_id, f"what_if:{perturbation}", outcomes, config)
    return outcomes


def format_comparison_table(outcomes: list[CandidateOutcome]) -> str:
    header = f"{'Action':<55} {'Cost':>12} {'Delay':>8} {'Stockout':>10}"
    lines = [header, "-" * len(header)]
    for o in outcomes:
        lines.append(
            f"{o.action:<55} ₹{o.cost_delta:>10,.0f} {o.delivery_delay_days:>6.1f}d "
            f"{o.stockout_risk * 100:>8.0f}%"
        )
    return "\n".join(lines)