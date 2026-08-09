"""
Tests for twin_adapter.py — the bridge between digital-twin-fyp-main's
dataclasses and this project's pydantic schemas.py.

Three things matter here, matching twin_adapter.py's own two stated jobs
plus the field-mapping decisions documented in its `to_pydantic_state`
docstring:

  1. The schemas.py naming collision is actually resolved, and (critically)
     resolved WITHOUT leaking — our own pydantic `schemas` module must be
     exactly what every other module in this package sees, both before and
     after twin_adapter has loaded the twin's dataclass version under the
     same bare name.
  2. build_live_twin() produces a real, working DigitalTwin (using the
     synthetic-data fallback, so this suite never needs the Kaggle CSV).
  3. to_pydantic_state() converts correctly, including every documented
     fallback: demand duplication, planned_transit_days derivation vs.
     default, off_route always False, status pass-through vs. inference,
     and supplier lead_time rounding/clamping.
"""
from __future__ import annotations
import dataclasses

import schemas
import twin_adapter
from agents import monitoring_agent
from tests.conftest import make_real_twin


# ---------------------------------------------------------------------------
# 1. Module collision: our schemas.py must stay ours, everywhere, always
# ---------------------------------------------------------------------------

def test_our_schemas_module_is_pydantic():
    assert hasattr(schemas.TwinState, "model_fields"), (
        "schemas.TwinState should be a pydantic model"
    )


def test_twin_schemas_module_is_the_real_dataclasses():
    assert dataclasses.is_dataclass(twin_adapter.twin_schemas.TwinState), (
        "twin_adapter.twin_schemas should be digital-twin-fyp-main's plain-dataclass schemas"
    )
    assert not hasattr(twin_adapter.twin_schemas.TwinState, "model_fields"), (
        "the twin's TwinState must NOT be the pydantic one — that would mean the collision leaked"
    )


def test_our_schemas_and_twin_schemas_are_genuinely_different_classes():
    # Same class names on both sides (Warehouse, Shipment, Supplier, TwinState)
    # is exactly the scenario that causes silent collisions if unresolved.
    assert schemas.Warehouse is not twin_adapter.twin_schemas.Warehouse
    assert schemas.TwinState is not twin_adapter.twin_schemas.TwinState


def test_agents_that_import_schemas_get_our_pydantic_version_not_the_twins():
    # monitoring_agent.py does `from schemas import TwinState, DisruptionEvent`
    # at its own module scope — if the collision ever leaked, this binding
    # would silently become the twin's dataclass instead.
    assert monitoring_agent.TwinState is schemas.TwinState


def test_schemas_module_identity_is_unaffected_by_reimporting_twin_adapter():
    # twin_adapter is only ever executed once per process (Python caches
    # module imports), but re-fetching it from sys.modules must never
    # perturb the bare `schemas` binding other modules rely on.
    import sys
    before = sys.modules["schemas"]
    import importlib
    reimported = importlib.import_module("twin_adapter")
    assert reimported is twin_adapter
    assert sys.modules["schemas"] is before


# ---------------------------------------------------------------------------
# 2. build_live_twin() — synthetic fallback, no CSV required
# ---------------------------------------------------------------------------

def test_build_live_twin_returns_a_working_twin():
    twin = twin_adapter.build_live_twin(use_real_data=False, rng_seed=1)
    assert isinstance(twin, twin_adapter.DigitalTwin)
    assert twin.warehouses, "synthetic fallback should still populate warehouses"
    assert twin.suppliers, "synthetic fallback should still populate suppliers"


def test_build_live_twin_twin_can_actually_simulate():
    twin = twin_adapter.build_live_twin(use_real_data=False, rng_seed=1)
    outcome = twin.simulate_candidate({"type": "wait"}, horizon_days=3)
    assert hasattr(outcome, "cost_delta")
    assert hasattr(outcome, "stockout_risk")


def test_build_live_twin_never_mutates_across_calls():
    # Building a second twin must not be influenced by (or influence) the first.
    twin_a = twin_adapter.build_live_twin(use_real_data=False, rng_seed=1)
    twin_b = twin_adapter.build_live_twin(use_real_data=False, rng_seed=2)
    assert twin_a is not twin_b
    assert twin_a.warehouses is not twin_b.warehouses


# ---------------------------------------------------------------------------
# 3. to_pydantic_state() — field-mapping decisions
# ---------------------------------------------------------------------------

def test_daily_demand_populates_both_avg_and_recent_identically():
    twin = make_real_twin(daily_demand=33.0)
    state = twin_adapter.to_pydantic_state(twin.state())
    wh = next(w for w in state.warehouses if w.id == "WH-01")
    assert wh.avg_daily_demand["SKU-1"] == 33.0
    assert wh.recent_daily_demand["SKU-1"] == 33.0
    assert wh.avg_daily_demand == wh.recent_daily_demand


def test_planned_transit_days_derived_from_known_supplier_lead_time():
    twin = make_real_twin(lead_time_days=12.4)  # SHP-1's origin is SUP-1
    state = twin_adapter.to_pydantic_state(twin.state())
    shp = next(s for s in state.shipments if s.id == "SHP-1")
    assert shp.planned_transit_days == round(12.4)


def test_planned_transit_days_falls_back_to_default_for_unknown_origin():
    twin = make_real_twin()
    # Build a shipment whose origin is a warehouse, not a supplier — this is
    # exactly what an internal reallocate-created shipment looks like, and
    # the real twin has no "expected lead time" concept for that.
    RealShipment = twin_adapter.twin_schemas.Shipment
    twin.shipments["SHP-2"] = RealShipment(
        id="SHP-2", origin="WH-02", destination="WH-01", eta="2026-08-20",
        status="in_transit", delay_days=0, sku="SKU-1", quantity=50,
    )
    state = twin_adapter.to_pydantic_state(twin.state())
    shp2 = next(s for s in state.shipments if s.id == "SHP-2")
    assert shp2.planned_transit_days == twin_adapter.DEFAULT_PLANNED_TRANSIT_DAYS


def test_planned_transit_days_default_is_overridable():
    twin = make_real_twin()
    RealShipment = twin_adapter.twin_schemas.Shipment
    twin.shipments["SHP-2"] = RealShipment(
        id="SHP-2", origin="WH-02", destination="WH-01", eta="2026-08-20",
        status="in_transit", delay_days=0, sku="SKU-1", quantity=50,
    )
    state = twin_adapter.to_pydantic_state(twin.state(), default_planned_transit_days=3)
    shp2 = next(s for s in state.shipments if s.id == "SHP-2")
    assert shp2.planned_transit_days == 3


def test_off_route_is_always_false():
    twin = make_real_twin()
    state = twin_adapter.to_pydantic_state(twin.state())
    assert all(s.off_route is False for s in state.shipments)


def test_status_passes_through_when_already_valid():
    twin = make_real_twin(shipment_status="delivered")
    state = twin_adapter.to_pydantic_state(twin.state())
    shp = next(s for s in state.shipments if s.id == "SHP-1")
    assert shp.status == "delivered"


def test_status_falls_back_to_delayed_when_invalid_and_delay_days_positive():
    twin = make_real_twin()
    twin.shipments["SHP-1"].status = "some_status_the_twin_never_actually_uses"
    twin.shipments["SHP-1"].delay_days = 4
    state = twin_adapter.to_pydantic_state(twin.state())
    shp = next(s for s in state.shipments if s.id == "SHP-1")
    assert shp.status == "delayed"


def test_status_falls_back_to_in_transit_when_invalid_and_no_delay():
    twin = make_real_twin()
    twin.shipments["SHP-1"].status = "some_status_the_twin_never_actually_uses"
    twin.shipments["SHP-1"].delay_days = 0
    state = twin_adapter.to_pydantic_state(twin.state())
    shp = next(s for s in state.shipments if s.id == "SHP-1")
    assert shp.status == "in_transit"


def test_supplier_lead_time_is_rounded():
    twin = make_real_twin(lead_time_days=7.6)
    state = twin_adapter.to_pydantic_state(twin.state())
    sup = next(s for s in state.suppliers if s.id == "SUP-1")
    assert sup.lead_time_days == 8


def test_supplier_lead_time_is_clamped_to_at_least_one_day():
    twin = make_real_twin(lead_time_days=0.2)
    state = twin_adapter.to_pydantic_state(twin.state())
    sup = next(s for s in state.suppliers if s.id == "SUP-1")
    assert sup.lead_time_days >= 1


def test_supplier_reliability_score_passes_through_unchanged():
    twin = make_real_twin(reliability_score=0.73)
    state = twin_adapter.to_pydantic_state(twin.state())
    sup = next(s for s in state.suppliers if s.id == "SUP-1")
    assert sup.reliability_score == 0.73


def test_inventory_and_safety_stock_pass_through_unchanged():
    twin = make_real_twin(on_hand=88, safety_stock=42)
    state = twin_adapter.to_pydantic_state(twin.state())
    wh = next(w for w in state.warehouses if w.id == "WH-01")
    assert wh.inventory["SKU-1"] == 88
    assert wh.safety_stock["SKU-1"] == 42


# ---------------------------------------------------------------------------
# End-to-end sanity: a converted snapshot must be usable by monitoring_agent
# ---------------------------------------------------------------------------

def test_converted_state_feeds_monitoring_agent_without_error():
    twin = make_real_twin(on_hand=50, safety_stock=150)  # understocked, should flag
    state = twin_adapter.to_pydantic_state(twin.state())
    events = monitoring_agent.run(state)
    assert any(e.type == "stock_imbalance" for e in events)