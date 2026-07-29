"""
twin.py
-------
The digital twin itself: a SimPy discrete-event simulation of warehouses,
in-transit shipments, and suppliers.

This is Member A's core deliverable (Section 8):
  - the SimPy simulation engine (state machine, updates on events)
  - the twin state schema (owned jointly with schemas.py)
  - the fork() mechanism the Simulation agent (Member B) depends on to
    test candidate actions "never on the live state" (Section 4.2)

How the agent layer is expected to use this module
----------------------------------------------------
    twin = DigitalTwin(...)                       # the live twin
    twin.advance(1)                                # tick simulated time forward
    state = twin.state()                           # read-only snapshot -> dict via state().to_dict()

    # Simulation agent evaluates 2-3 candidates on FORKS, live twin untouched:
    outcome = twin.simulate_candidate({"type": "reorder", ...}, horizon_days=7)

    # Orchestrator, after human approval, applies the SAME action to the
    # live twin for real:
    twin.place_order(...)   # or reroute_shipment / reallocate_stock
"""

from __future__ import annotations
import copy
import random
from datetime import date, timedelta
from typing import Dict, List, Optional

import simpy

from schemas import (
    Warehouse, Shipment, Supplier, TwinState, DisruptionEvent, ActionOutcome,
    next_shipment_id,
)

# ---- Simple, tunable cost model (arbitrary currency units - tune freely) ----
HOLDING_COST_PER_UNIT_DAY = 0.5
STOCKOUT_COST_PER_UNIT_DAY = 15.0
RUSH_ORDER_PREMIUM = 1.6          # multiplier for an expedited reorder
NORMAL_UNIT_COST = 20.0
REROUTE_HANDLING_COST = 2500.0
INTERNAL_TRANSFER_COST_PER_UNIT = 3.0


class DigitalTwin:
    def __init__(self, warehouses: List[Warehouse], shipments: List[Shipment],
                 suppliers: List[Supplier], sim_date: str,
                 metrics: Optional[dict] = None,
                 demand_variability: float = 0.15,
                 rng_seed: Optional[int] = None):
        self.env = simpy.Environment()
        self.start_date = date.fromisoformat(sim_date)
        self.warehouses: Dict[str, Warehouse] = {w.id: w for w in warehouses}
        self.shipments: Dict[str, Shipment] = {s.id: s for s in shipments}
        self.suppliers: Dict[str, Supplier] = {s.id: s for s in suppliers}
        self.metrics = dict(metrics) if metrics else {
            "total_cost": 0.0,
            "stockout_days": 0,
            "orders_fulfilled_on_time": 0,
            "orders_total": 0,
        }
        self.demand_variability = demand_variability
        self.rng = random.Random(rng_seed)
        self._bootstrap_processes()

    # ---------------- setup ----------------
    def _bootstrap_processes(self):
        for wh in self.warehouses.values():
            for sku, rate in wh.daily_demand.items():
                if rate > 0:
                    self.env.process(self._demand_loop(wh.id, sku))
        for shp in self.shipments.values():
            if shp.status == "in_transit":
                self.env.process(self._shipment_loop(shp.id))

    def _days_until(self, iso_date: str) -> float:
        target = date.fromisoformat(iso_date)
        return max((target - self.start_date).days, 0)

    # ---------------- SimPy processes ----------------
    def _demand_loop(self, warehouse_id: str, sku: str):
        """One tick per simulated day: realize demand, deplete inventory,
        accrue holding/stockout cost. This is what makes 'wait and absorb'
        actually risk a stockout instead of being a free no-op."""
        wh = self.warehouses[warehouse_id]
        while True:
            yield self.env.timeout(1)
            base_rate = wh.daily_demand.get(sku, 0)
            noise = 1 + self.rng.uniform(-self.demand_variability, self.demand_variability)
            demand_today = max(0, round(base_rate * noise))

            on_hand = wh.inventory.get(sku, 0)
            fulfilled = min(on_hand, demand_today)
            shortfall = demand_today - fulfilled
            wh.inventory[sku] = on_hand - fulfilled

            self.metrics["total_cost"] += wh.inventory.get(sku, 0) * HOLDING_COST_PER_UNIT_DAY
            if shortfall > 0:
                self.metrics["stockout_days"] += 1
                self.metrics["total_cost"] += shortfall * STOCKOUT_COST_PER_UNIT_DAY

    def _shipment_loop(self, shipment_id: str):
        """Waits until a shipment's (possibly delayed) ETA, then delivers
        it into the destination warehouse's inventory."""
        shp = self.shipments[shipment_id]
        remaining = max(self._days_until(shp.eta) + shp.delay_days, 0)
        yield self.env.timeout(remaining)
        shp.status = "delivered"
        dest = self.warehouses.get(shp.destination)
        if dest is not None and shp.sku:
            dest.inventory[shp.sku] = dest.inventory.get(shp.sku, 0) + shp.quantity
        self.metrics["orders_total"] += 1
        if shp.delay_days <= 0:
            self.metrics["orders_fulfilled_on_time"] += 1

    # ---------------- mutation API (agents call these) ----------------
    def place_order(self, supplier_id: str, warehouse_id: str, sku: str,
                     quantity: int, rush: bool = False) -> Shipment:
        supplier = self.suppliers[supplier_id]
        lead_time = supplier.lead_time_days * (0.5 if rush else 1.0)
        eta = (self.start_date + timedelta(days=self.env.now + lead_time)).isoformat()
        delay = 0 if self.rng.random() < supplier.reliability_score else self.rng.randint(1, 3)
        shp = Shipment(
            id=next_shipment_id(), origin=supplier_id, destination=warehouse_id,
            eta=eta, status="in_transit", delay_days=delay, sku=sku, quantity=quantity,
        )
        self.shipments[shp.id] = shp
        unit_cost = NORMAL_UNIT_COST * (RUSH_ORDER_PREMIUM if rush else 1.0)
        self.metrics["total_cost"] += unit_cost * quantity
        self.env.process(self._shipment_loop(shp.id))
        return shp

    def reroute_shipment(self, shipment_id: str, new_destination: str,
                          extra_delay_days: int = 1) -> None:
        shp = self.shipments[shipment_id]
        shp.destination = new_destination
        shp.delay_days += extra_delay_days
        self.metrics["total_cost"] += REROUTE_HANDLING_COST

    def reallocate_stock(self, from_warehouse: str, to_warehouse: str,
                          sku: str, quantity: int) -> None:
        src, dst = self.warehouses[from_warehouse], self.warehouses[to_warehouse]
        moved = min(src.inventory.get(sku, 0), quantity)
        src.inventory[sku] = src.inventory.get(sku, 0) - moved
        dst.inventory[sku] = dst.inventory.get(sku, 0) + moved
        self.metrics["total_cost"] += moved * INTERNAL_TRANSFER_COST_PER_UNIT

    def apply_disruption(self, event: DisruptionEvent) -> DisruptionEvent:
        """Mutates the LIVE twin. Used when a disruption is injected for
        real (as opposed to being tested inside a fork)."""
        if event.type == "shipment_delay":
            shp = self.shipments.get(event.affected_id)
            if shp:
                shp.delay_days += 3
        elif event.type == "demand_spike":
            for wh in self.warehouses.values():
                if event.affected_id in wh.daily_demand:
                    wh.daily_demand[event.affected_id] *= 1.4
        elif event.type == "stock_imbalance":
            # Imbalance is a detected condition, not an injected mutation -
            # the Monitoring agent flags it from existing state; nothing to
            # apply here. Kept as an explicit no-op for clarity.
            pass
        return event

    # ---------------- simulation control ----------------
    def advance(self, days: float) -> None:
        self.env.run(until=self.env.now + days)

    @property
    def sim_date(self) -> str:
        return (self.start_date + timedelta(days=self.env.now)).isoformat()

    # ---------------- state I/O ----------------
    def state(self) -> TwinState:
        return TwinState(
            warehouses=[copy.deepcopy(w) for w in self.warehouses.values()],
            shipments=[copy.deepcopy(s) for s in self.shipments.values()],
            suppliers=[copy.deepcopy(s) for s in self.suppliers.values()],
            sim_date=self.sim_date,
            metrics=dict(self.metrics),
        )

    def fork(self) -> "DigitalTwin":
        """Deep-copies current state into a brand-new twin with its own
        SimPy environment reset to t=0 ('now'). This is the mechanism the
        Simulation agent uses to test candidate actions without ever
        touching the live twin (Section 4.2)."""
        snap = self.state()
        return DigitalTwin(
            warehouses=[copy.deepcopy(w) for w in snap.warehouses],
            shipments=[copy.deepcopy(s) for s in snap.shipments],
            suppliers=[copy.deepcopy(s) for s in snap.suppliers],
            sim_date=snap.sim_date,
            metrics=snap.metrics,
            demand_variability=self.demand_variability,
            rng_seed=self.rng.randint(0, 2**31 - 1),
        )

    # ---------------- what-if evaluation ----------------
    def simulate_candidate(self, action: dict, horizon_days: int = 7) -> ActionOutcome:
        """
        The main hook the Simulation agent calls. Runs `action` on a FORK,
        fast-forwards `horizon_days`, and returns a quantified outcome.

        action examples:
          {"type": "reorder", "supplier_id": "SUP-07", "warehouse_id": "WH-01",
           "sku": "SKU-100", "quantity": 300, "rush": True}
          {"type": "reroute", "shipment_id": "SHP-482", "new_destination": "WH-02"}
          {"type": "reallocate", "from_warehouse": "WH-01", "to_warehouse": "WH-03",
           "sku": "SKU-100", "quantity": 150}
          {"type": "wait"}
        """
        twin = self.fork()
        cost_before = twin.metrics["total_cost"]
        stockout_days_before = twin.metrics["stockout_days"]
        orders_before = twin.metrics["orders_total"]
        ontime_before = twin.metrics["orders_fulfilled_on_time"]

        atype = action.get("type", "wait")
        delay_days = 0
        if atype == "reorder":
            twin.place_order(action["supplier_id"], action["warehouse_id"],
                              action["sku"], action["quantity"], action.get("rush", False))
            delay_days = twin.suppliers[action["supplier_id"]].lead_time_days
        elif atype == "reroute":
            extra = action.get("extra_delay_days", 1)
            twin.reroute_shipment(action["shipment_id"], action["new_destination"], extra)
            delay_days = extra
        elif atype == "reallocate":
            twin.reallocate_stock(action["from_warehouse"], action["to_warehouse"],
                                   action["sku"], action["quantity"])
        elif atype == "wait":
            pass
        else:
            raise ValueError(f"Unknown action type: {atype!r}")

        twin.advance(horizon_days)

        cost_delta = twin.metrics["total_cost"] - cost_before
        stockout_days_after = twin.metrics["stockout_days"] - stockout_days_before
        stockout_risk = min(stockout_days_after / horizon_days, 1.0)

        orders_after = twin.metrics["orders_total"] - orders_before
        ontime_after = twin.metrics["orders_fulfilled_on_time"] - ontime_before
        service_level = (ontime_after / orders_after) if orders_after else 1.0

        return ActionOutcome(
            action=atype,
            cost_delta=round(cost_delta, 2),
            delivery_delay_days=round(delay_days, 1),
            stockout_risk=round(stockout_risk, 2),
            service_level=round(service_level, 2),
        )