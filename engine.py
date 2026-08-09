"""
Digital Twin layer -- Section 3/8, Member A ownership.

DigitalTwinEngine wraps a TwinState in a SimPy discrete-event simulation.
It is the ONLY thing in the whole system allowed to mutate a TwinState's
inventory/shipment numbers -- agents (Member B) never hand-edit twin data,
they call engine methods.

Two usage modes, matching the plan's "never on the live state" rule
(Section 4.2):

1. LIVE mode -- the one engine instance backing the actual dashboard.
   Backend (Member C) calls `engine.inject_disruption(...)` when something
   really happens, and `engine.apply_action(...)` + `engine.run(1)` (a day
   at a time, or however the backend's clock is driven) to advance reality.

2. FORKED mode -- the Simulation agent (Member B) calls `engine.clone()`
   to get an independent copy, then applies a candidate action and calls
   `run(horizon_days)` on the *clone*, fast-forwarding without touching the
   live twin. `simulate_candidates()` below is a ready-made helper for
   exactly this.

Cost constants below are placeholders in INR -- tune them with the group
once you agree on realistic numbers for your demo scenarios (Section 10).
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import simpy

from schemas import (
    TwinState, Warehouse, Shipment, Supplier,
    DisruptionEvent, CandidateOutcome,
)

# ---- tunable cost model (confirm with the group, Section 10 metrics) ----
HOLDING_COST_PER_UNIT_PER_DAY = 2.0
LOST_SALE_COST_PER_UNIT = 150.0
REORDER_COST_PER_UNIT = 50.0
RUSH_REORDER_MULTIPLIER = 1.8
REROUTE_FLAT_FEE = 3000.0
REALLOCATE_COST_PER_UNIT = 10.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DigitalTwinEngine:
    def __init__(self, state: TwinState, demand_noise: float = 0.25, seed: Optional[int] = None):
        self.state = state
        self.demand_noise = demand_noise
        self.rng = random.Random(seed)
        self.env = simpy.Environment()
        # metrics accumulated over the most recent run() call
        self.metrics = {
            "cost": 0.0,
            "demand_total": 0.0,
            "demand_met": 0.0,
            "stockout_events": 0,
            "sku_days": 0,
            "delay_days_sum": 0.0,
            "delay_days_count": 0,
        }
        self._pending_action = None  # (action_type, params) queued to apply at run() start

    # ------------------------------------------------------------------
    # Perceive
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """What the Monitoring/Exception agent 'perceives' each tick."""
        return self.state.to_dict()

    def clone(self) -> "DigitalTwinEngine":
        """Fork for the Simulation agent -- never simulate on the live engine."""
        return DigitalTwinEngine(
            self.state.clone(), demand_noise=self.demand_noise,
            seed=self.rng.randint(0, 2**31 - 1),
        )

    # ------------------------------------------------------------------
    # Disruption injection (manual demo button, or synthetic generator)
    # ------------------------------------------------------------------
    def inject_disruption(self, disruption_type: str, **params) -> DisruptionEvent:
        event_id = f"EVT-{uuid.uuid4().hex[:6].upper()}"
        if disruption_type == "shipment_delay":
            shp = self._find_shipment(params["shipment_id"])
            delay_days = int(params.get("delay_days", 3))
            shp.delay_days += delay_days
            shp.status = "delayed"
            desc = f"Shipment {shp.id} is projected {shp.delay_days} days late"
            severity = "high" if shp.delay_days >= 5 else ("medium" if shp.delay_days >= 2 else "low")
            affected_id = shp.id

        elif disruption_type == "demand_spike":
            wh = self._find_warehouse(params["warehouse_id"])
            sku = params["sku"]
            multiplier = float(params.get("multiplier", 1.4))
            wh.sim_daily_demand[sku] = round(wh.sim_daily_demand.get(sku, 5.0) * multiplier, 2)
            desc = f"Demand for {sku} at {wh.id} spiked by {int((multiplier - 1) * 100)}%"
            severity = "high" if multiplier >= 1.5 else "medium"
            affected_id = wh.id

        elif disruption_type == "stock_imbalance":
            wh = self._find_warehouse(params["warehouse_id"])
            sku = params["sku"]
            delta = int(params.get("delta", -50))
            wh.inventory[sku] = max(0, wh.inventory.get(sku, 0) + delta)
            desc = f"Stock imbalance for {sku} at {wh.id} ({'shortage' if delta < 0 else 'excess'} of {abs(delta)} units)"
            severity = "medium"
            affected_id = wh.id

        else:
            raise ValueError(f"Unknown disruption_type: {disruption_type}")

        return DisruptionEvent(
            event_id=event_id, type=disruption_type, affected_id=affected_id,
            severity=severity, detected_at=_now_iso(), description=desc,
        )

    # ------------------------------------------------------------------
    # Act (candidate actions -- reorder / reroute / reallocate / wait)
    # ------------------------------------------------------------------
    def apply_action(self, action_type: str, params: dict) -> float:
        """
        Applies an action immediately (queues cost/state effects). Returns
        the one-off cost incurred. Recurring costs (holding, lost sales)
        are accounted for day-by-day inside run().
        """
        if action_type == "wait":
            return 0.0

        if action_type == "reorder":
            wh = self._find_warehouse(params["warehouse_id"])
            sku = params["sku"]
            qty = int(params["qty"])
            supplier = self._find_supplier(params["supplier_id"])
            expedite = bool(params.get("expedite", False))
            lead_time = max(1, supplier.lead_time_days // 2) if expedite else supplier.lead_time_days
            unit_cost = REORDER_COST_PER_UNIT * (RUSH_REORDER_MULTIPLIER if expedite else 1.0)
            cost = qty * unit_cost
            wh.sim_pending_orders.append({
                "sku": sku, "qty": qty,
                "arrival_day": self.state.sim_day + lead_time,
                "supplier_id": supplier.id,
            })
            self.metrics["cost"] += cost
            return cost

        if action_type == "reroute":
            shp = self._find_shipment(params["shipment_id"])
            via_warehouse = params.get("via_warehouse")
            reduction = int(params.get("delay_reduction_days", max(1, shp.delay_days // 2)))
            shp.delay_days = max(0, shp.delay_days - reduction)
            if via_warehouse:
                shp.destination = via_warehouse
            cost = REROUTE_FLAT_FEE
            self.metrics["cost"] += cost
            return cost

        if action_type == "reallocate":
            from_wh = self._find_warehouse(params["from_warehouse"])
            to_wh = self._find_warehouse(params["to_warehouse"])
            sku = params["sku"]
            qty = int(params["qty"])
            available = from_wh.inventory.get(sku, 0)
            moved = min(qty, available)
            from_wh.inventory[sku] = available - moved
            # modeled as a 1-day internal transfer via pending order on destination
            to_wh.sim_pending_orders.append({
                "sku": sku, "qty": moved,
                "arrival_day": self.state.sim_day + 1,
                "supplier_id": None,
            })
            cost = moved * REALLOCATE_COST_PER_UNIT
            self.metrics["cost"] += cost
            return cost

        raise ValueError(f"Unknown action_type: {action_type}")

    # ------------------------------------------------------------------
    # Simulate forward (SimPy discrete-event loop, one tick = one day)
    # ------------------------------------------------------------------
    def run(self, days: int) -> None:
        proc = self.env.process(self._day_cycle(days))
        self.env.run(until=proc)

    def _day_cycle(self, days: int):
        for _ in range(days):
            yield self.env.timeout(1)
            self.state.sim_day += 1
            self._tick_shipments()
            self._tick_pending_orders()
            self._tick_demand()

    def _tick_shipments(self):
        for shp in self.state.shipments:
            if shp.status == "delivered":
                continue
            planned_arrival_day = shp.day_created + shp.planned_transit_days + shp.delay_days
            if self.state.sim_day >= planned_arrival_day:
                shp.status = "delivered"
                dest = self._find_warehouse(shp.destination, required=False)
                if dest and shp.sku:
                    dest.inventory[shp.sku] = dest.inventory.get(shp.sku, 0) + shp.qty
                self.metrics["delay_days_sum"] += shp.delay_days
                self.metrics["delay_days_count"] += 1

    def _tick_pending_orders(self):
        for wh in self.state.warehouses:
            arrived, still_pending = [], []
            for order in wh.sim_pending_orders:
                (arrived if order["arrival_day"] <= self.state.sim_day else still_pending).append(order)
            for order in arrived:
                wh.inventory[order["sku"]] = wh.inventory.get(order["sku"], 0) + order["qty"]
            wh.sim_pending_orders = still_pending

    def _tick_demand(self):
        for wh in self.state.warehouses:
            for sku, avg_daily in wh.sim_daily_demand.items():
                noise = 1.0 + self.rng.uniform(-self.demand_noise, self.demand_noise)
                demand = max(0.0, avg_daily * noise)
                on_hand = wh.inventory.get(sku, 0)
                met = min(on_hand, demand)
                unmet = demand - met
                wh.inventory[sku] = max(0, on_hand - demand)

                self.metrics["demand_total"] += demand
                self.metrics["demand_met"] += met
                self.metrics["sku_days"] += 1
                if unmet > 0.01:
                    self.metrics["stockout_events"] += 1
                    self.metrics["cost"] += unmet * LOST_SALE_COST_PER_UNIT
                self.metrics["cost"] += wh.inventory[sku] * HOLDING_COST_PER_UNIT_PER_DAY / 30.0

    # ------------------------------------------------------------------
    # Metrics for the Decision/Recommendation agent (Section 4.3)
    # ------------------------------------------------------------------
    def compute_metrics(self) -> dict:
        m = self.metrics
        service_level = (m["demand_met"] / m["demand_total"]) if m["demand_total"] > 0 else 1.0
        stockout_risk = (m["stockout_events"] / m["sku_days"]) if m["sku_days"] > 0 else 0.0
        avg_delay = (m["delay_days_sum"] / m["delay_days_count"]) if m["delay_days_count"] > 0 else 0.0
        return {
            "cost": round(m["cost"], 2),
            "service_level": round(service_level, 4),
            "stockout_risk": round(stockout_risk, 4),
            "avg_delay_days": round(avg_delay, 2),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _find_warehouse(self, wh_id: str, required: bool = True) -> Optional[Warehouse]:
        for w in self.state.warehouses:
            if w.id == wh_id:
                return w
        if required:
            raise KeyError(f"Warehouse not found: {wh_id}")
        return None

    def _find_shipment(self, shp_id: str) -> Shipment:
        for s in self.state.shipments:
            if s.id == shp_id:
                return s
        raise KeyError(f"Shipment not found: {shp_id}")

    def _find_supplier(self, sup_id: str) -> Supplier:
        for s in self.state.suppliers:
            if s.id == sup_id:
                return s
        raise KeyError(f"Supplier not found: {sup_id}")


def simulate_candidates(
    state: TwinState,
    candidates: list[Tuple[str, dict]],
    horizon_days: int = 14,
    demand_noise: float = 0.25,
    seed: int = 7,
) -> Tuple[list[CandidateOutcome], dict]:
    """
    Convenience wrapper for the Simulation agent (Section 4.2). `state`
    should already reflect the disruption (i.e. call
    engine.inject_disruption(...) on the LIVE engine first, then pass
    engine.state here). Runs a 'wait' baseline plus every candidate on
    independent forked clones (same RNG seed so runs are comparable), and
    returns outcomes with cost/delay expressed as deltas vs. the baseline --
    matching the "saves X, adds N days" framing in Section 4.3.
    """
    baseline_engine = DigitalTwinEngine(state.clone(), demand_noise=demand_noise, seed=seed)
    baseline_engine.run(horizon_days)
    baseline = baseline_engine.compute_metrics()

    outcomes = []
    for action_type, params in candidates:
        eng = DigitalTwinEngine(state.clone(), demand_noise=demand_noise, seed=seed)
        if action_type != "wait":
            eng.apply_action(action_type, params)
        eng.run(horizon_days)
        m = eng.compute_metrics()
        outcomes.append(CandidateOutcome(
            action_type=action_type,
            params=params,
            cost_delta=round(m["cost"] - baseline["cost"], 2),
            delivery_delay_days=round(m["avg_delay_days"] - baseline["avg_delay_days"], 2),
            stockout_risk=m["stockout_risk"],
            service_level=m["service_level"],
        ))
    return outcomes, baseline
