"""
disruption_injector.py
-----------------------
Member A's other deliverable: "the synthetic disruption injector used for
the demo and for testing" (Section 8).

Two entry points:
  - inject_shipment_delay / inject_demand_spike / inject_stock_imbalance:
    manual, precise triggers -> wired to the dashboard's "inject disruption"
    button (Section 6.1) via the API layer.
  - random_disruption(twin): picks a plausible disruption on its own -
    used for automated testing and for generating the 4-5 evaluation
    scenarios in Section 10.

Every function returns a DisruptionEvent built to the Section 7.2 shape,
and (except for the "detect only" case) applies it to the live twin via
twin.apply_disruption(). The Monitoring/Exception agent (Member B) is what
actually *notices* organically-arising problems in the twin state; this
module is for deliberately creating them.
"""

from __future__ import annotations
import random
from datetime import datetime, timezone

from schemas import DisruptionEvent, next_event_id
from twin import DigitalTwin


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inject_shipment_delay(twin: DigitalTwin, shipment_id: str,
                           severity: str = "medium") -> DisruptionEvent:
    shp = twin.shipments.get(shipment_id)
    if shp is None:
        raise KeyError(f"No such shipment: {shipment_id}")
    event = DisruptionEvent(
        event_id=next_event_id(),
        type="shipment_delay",
        affected_id=shipment_id,
        severity=severity,
        detected_at=_now_iso(),
        description=f"Shipment {shipment_id} is projected to run late.",
    )
    twin.apply_disruption(event)
    event.description = (
        f"Shipment {shipment_id} is now projected {shp.delay_days} day(s) late."
    )
    return event


def inject_demand_spike(twin: DigitalTwin, sku: str,
                         severity: str = "medium") -> DisruptionEvent:
    event = DisruptionEvent(
        event_id=next_event_id(),
        type="demand_spike",
        affected_id=sku,
        severity=severity,
        detected_at=_now_iso(),
        description=f"Demand for {sku} spiked ~40% above baseline.",
    )
    twin.apply_disruption(event)
    return event


def inject_stock_imbalance(twin: DigitalTwin, warehouse_id: str, sku: str,
                            severity: str = "low") -> DisruptionEvent:
    """Stock imbalance is a *detected* condition (inventory below safety
    stock somewhere while another warehouse is oversupplied) rather than
    something we mutate into existence - we just flag it here for demo
    purposes so the pipeline has something to react to."""
    wh = twin.warehouses.get(warehouse_id)
    on_hand = wh.inventory.get(sku, 0) if wh else 0
    safety = wh.safety_stock.get(sku, 0) if wh else 0
    event = DisruptionEvent(
        event_id=next_event_id(),
        type="stock_imbalance",
        affected_id=warehouse_id,
        severity=severity,
        detected_at=_now_iso(),
        description=(
            f"{warehouse_id} holds {on_hand} units of {sku} against a "
            f"safety stock of {safety}."
        ),
    )
    return event


def random_disruption(twin: DigitalTwin, rng: random.Random | None = None) -> DisruptionEvent:
    """Picks a plausible disruption at random from the twin's current
    state. Handy for `for _ in range(N): random_disruption(twin)` style
    stress-testing, or for generating repeatable evaluation scenarios by
    seeding `rng`."""
    rng = rng or random.Random()
    choice = rng.choice(["shipment_delay", "demand_spike"])

    if choice == "shipment_delay":
        candidates = [s for s in twin.shipments.values() if s.status == "in_transit"]
        if not candidates:
            return random_disruption(twin, rng)  # retry with the other kind
        shp = rng.choice(candidates)
        severity = rng.choice(["low", "medium", "high"])
        return inject_shipment_delay(twin, shp.id, severity)

    else:  # demand_spike
        skus = {sku for wh in twin.warehouses.values() for sku in wh.daily_demand}
        if not skus:
            return random_disruption(twin, rng)
        sku = rng.choice(list(skus))
        severity = rng.choice(["low", "medium", "high"])
        return inject_demand_spike(twin, sku, severity)