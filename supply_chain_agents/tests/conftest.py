"""
Shared fixtures for the agent test suite.

Two families of test data live here, because the codebase itself has two
schema worlds (see twin_adapter.py's module docstring):

- `make_twin()` — builds OUR pydantic TwinState directly. Use this for
  monitoring_agent / decision_agent tests, and for anything that only
  needs to look like twin_adapter's *output*.
- `make_real_twin()` — builds an actual, live twin_adapter.DigitalTwin
  (their dataclasses, dict-of-objects, real simulate_candidate()). Use
  this for simulation_agent tests, twin_adapter tests, and anything that
  needs to look like twin_adapter's *input*.

Autouse fixtures that matter for every test in this package:

- `no_llm`: patches llm_client.call_llm so tests never hit a real network
  call (Groq/Ollama) and never depend on an API key being configured. Every
  agent falls back to its deterministic template, so assertions check
  *structure and logic*, not LLM phrasing — which is the right thing to test
  anyway, since the numbers must always come from the simulation regardless
  of which LLM (if any) answered.
- `reset_dedup`: clears BOTH of the monitoring agent's module-level stores —
  the alert-dedup registry and the rolling per-(warehouse, sku) history
  buffer used for dead-stock/fast-moving/trend/self-learned-baseline
  detection — before each test, so tests don't leak state into each other
  via shared module state. (Both are intentionally process-global in
  production, which is exactly the kind of thing that causes flaky/
  order-dependent tests if not reset: without clearing history too, a trend
  check in one test can fire off inventory numbers left behind by an
  unrelated earlier test that happened to touch the same warehouse/SKU id.)
- `reset_sim_history`: same idea for simulation_agent's module-level
  `_simulation_history` list.
"""
from __future__ import annotations
import pytest

import llm_client
import twin_adapter
from agents import monitoring_agent, simulation_agent
from schemas import Warehouse, Shipment, Supplier, TwinState


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    def fake_call_llm(prompt: str, fallback: str) -> str:
        return fallback
    monkeypatch.setattr(llm_client, "call_llm", fake_call_llm)
    # Agents imported `call_llm` directly into their own namespace, so patch
    # those references too — patching only llm_client.call_llm would miss them.
    # (simulation_agent is NOT patched here: it never imports call_llm at
    # all — per its own docstring, the LLM's role there shrank to nothing
    # once action params became deterministic — so patching it would raise.)
    monkeypatch.setattr("agents.monitoring_agent.call_llm", fake_call_llm)
    monkeypatch.setattr("agents.decision_agent.call_llm", fake_call_llm)


@pytest.fixture(autouse=True)
def reset_dedup():
    monitoring_agent.reset_all_state()  # dedup registry AND history buffer
    yield
    monitoring_agent.reset_all_state()


@pytest.fixture(autouse=True)
def reset_sim_history():
    simulation_agent.reset_simulation_history()
    yield
    simulation_agent.reset_simulation_history()


# ---------------------------------------------------------------------------
# Pydantic-side twin (our schemas.py) — for monitoring_agent / decision_agent
# ---------------------------------------------------------------------------

def make_twin(
    *,
    delay_days: int = 0,
    planned_transit_days: int = 7,
    on_hand: int = 300,
    safety_stock: int = 150,
    avg_demand: float = 20,
    recent_demand: float = 20,
) -> TwinState:
    """Builds a single-warehouse, single-shipment, single-supplier twin with
    tunable knobs, so each test can dial in exactly the disruption it wants
    without repeating full TwinState boilerplate."""
    return TwinState(
        warehouses=[
            Warehouse(
                id="WH-01", location="TestCity",
                inventory={"SKU-1": on_hand}, safety_stock={"SKU-1": safety_stock},
                avg_daily_demand={"SKU-1": avg_demand}, recent_daily_demand={"SKU-1": recent_demand},
            ),
        ],
        shipments=[
            Shipment(
                id="SHP-1", origin="WH-01", destination="WH-02",
                eta="2026-08-01", status="delayed" if delay_days > 0 else "in_transit",
                delay_days=delay_days, planned_transit_days=planned_transit_days,
            ),
        ],
        suppliers=[Supplier(id="SUP-1", lead_time_days=10, reliability_score=0.9)],
    )


# ---------------------------------------------------------------------------
# Real-side twin (digital-twin-fyp-main's dataclasses, via twin_adapter) —
# for simulation_agent / twin_adapter tests
# ---------------------------------------------------------------------------

def make_real_twin(
    *,
    sim_date: str = "2026-08-01",
    # WH-01 -- the warehouse most tests target directly
    on_hand: int = 300,
    safety_stock: int = 150,
    daily_demand: float = 20.0,
    # WH-02 -- a second warehouse, so reallocate/reroute action-builders have
    # somewhere to actually move stock to/from or reroute a shipment toward
    wh2_on_hand: int = 500,
    wh2_safety_stock: int = 150,
    wh2_daily_demand: float = 10.0,
    # SHP-1 -- one shipment from SUP-1 to WH-01
    shipment_status: str = "in_transit",
    delay_days: int = 0,
    shipment_eta: str = "2026-08-15",
    shipment_sku: str = "SKU-1",
    shipment_quantity: int = 100,
    # SUP-1
    lead_time_days: float = 10.0,
    reliability_score: float = 0.9,
    rng_seed: int | None = 7,
):
    """Builds a small, tunable REAL DigitalTwin (two warehouses, one
    shipment, one supplier) — the dict-of-objects, duck-typed object
    simulation_agent.py and twin_adapter.py actually operate on, as opposed
    to make_twin()'s pydantic snapshot above.

    Two warehouses exist specifically so the reallocate/reroute
    action-builders have somewhere to move stock to/from or reroute a
    shipment toward — a single-warehouse twin can never produce those
    candidates. rng_seed is fixed by default so tests are deterministic.
    """
    RealWarehouse = twin_adapter.twin_schemas.Warehouse
    RealShipment = twin_adapter.twin_schemas.Shipment
    RealSupplier = twin_adapter.twin_schemas.Supplier

    warehouses = [
        RealWarehouse(
            id="WH-01", location="TestCity",
            inventory={"SKU-1": on_hand}, safety_stock={"SKU-1": safety_stock},
            daily_demand={"SKU-1": daily_demand},
        ),
        RealWarehouse(
            id="WH-02", location="OtherCity",
            inventory={"SKU-1": wh2_on_hand}, safety_stock={"SKU-1": wh2_safety_stock},
            daily_demand={"SKU-1": wh2_daily_demand},
        ),
    ]
    shipments = [
        RealShipment(
            id="SHP-1", origin="SUP-1", destination="WH-01", eta=shipment_eta,
            status=shipment_status, delay_days=delay_days,
            sku=shipment_sku, quantity=shipment_quantity,
        ),
    ]
    suppliers = [
        RealSupplier(id="SUP-1", lead_time_days=lead_time_days, reliability_score=reliability_score),
    ]

    return twin_adapter.DigitalTwin(
        warehouses=warehouses, shipments=shipments, suppliers=suppliers,
        sim_date=sim_date, rng_seed=rng_seed,
    )


@pytest.fixture
def real_twin():
    """Default-configured real twin: WH-01 sits below safety stock, so
    monitoring/simulation/action-builder tests have an actual disruption to
    react to without any extra setup."""
    return make_real_twin(on_hand=80, safety_stock=150)