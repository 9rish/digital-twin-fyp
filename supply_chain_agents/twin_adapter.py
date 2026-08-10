"""
twin_adapter.py — the bridge between digital-twin-fyp-main (their dataclasses)
and this project's schemas.py (our pydantic models).

Two jobs:
  1. Load the twin's modules without a naming collision. Both projects have a
     file called schemas.py. Worse: the twin's OWN files (twin.py,
     data_loader.py) internally do a bare `from schemas import ...` — so this
     isn't just about how WE import them, their own code will silently
     resolve to the WRONG schemas.py if sys.modules['schemas'] happens to
     already hold ours when their files load. Solved below with a temporary,
     scoped override of sys.modules['schemas'] — never a permanent one.
  2. Convert their dataclass TwinState into our pydantic TwinState, with the
     field-shape differences documented inline at each mapping decision.

Expects digital-twin-fyp-main/ to be a SIBLING folder to this project's root
(Final_Proj/digital-twin-fyp-main/, Final_Proj/supply_chain_agents/) — adjust
_TWIN_DIR below if your layout differs.
"""
from __future__ import annotations
import contextlib
import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TWIN_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "digital-twin-fyp-main"))


def _load_from_path(module_name: str, filename: str):
    path = os.path.join(_TWIN_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"twin_adapter.py expected the twin project at {_TWIN_DIR!r} "
            f"(a sibling folder to this project). Missing file: {path}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Must register under its own name BEFORE exec_module: Python's
    # @dataclass decorator (and other introspection) looks up
    # sys.modules[cls.__module__] internally while the class body executes —
    # without this, dataclass definitions inside the loaded file crash.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _temporarily_as(name: str, module):
    """Makes `module` resolve as sys.modules[name] for the duration of the
    block, then restores whatever was there before. This is the actual fix
    for the collision: the twin's own files do `from schemas import ...`,
    which looks up sys.modules['schemas'] at THEIR load time — theirs must
    win only while their files are loading, and must never leak into the
    rest of this process afterward (our own `from schemas import ...` calls
    everywhere else must keep resolving to OUR pydantic schemas.py)."""
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is not None:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)


# Load the twin's dataclass schemas under a name that can never collide with
# our own schemas.py.
twin_schemas = _load_from_path("twin_schemas", "schemas.py")

# twin.py, data_loader.py, and disruption_injector.py all do
# `from schemas import ...` internally -- while THEY load, sys.modules['schemas']
# must resolve to twin_schemas, or their classes get built against the wrong
# shape entirely. disruption_injector.py additionally does `from twin import
# DigitalTwin` (a bare import), so 'twin' needs the same temporary override.
with _temporarily_as("schemas", twin_schemas):
    twin_engine = _load_from_path("twin_engine", "twin.py")
    twin_data_loader = _load_from_path("twin_data_loader", "data_loader.py")
    with _temporarily_as("twin", twin_engine):
        twin_disruption_injector = _load_from_path("twin_disruption_injector", "disruption_injector.py")

DigitalTwin = twin_engine.DigitalTwin

# Re-exported so callers (demo.py, tests) don't need to know these live in
# the twin project's own disruption_injector.py under the hood.
inject_shipment_delay = twin_disruption_injector.inject_shipment_delay
inject_demand_spike = twin_disruption_injector.inject_demand_spike
inject_stock_imbalance = twin_disruption_injector.inject_stock_imbalance
random_disruption = twin_disruption_injector.random_disruption

# Only AFTER the twin's own modules have finished loading do we import OUR
# schemas under the bare name `schemas` — order matters here.
from schemas import Warehouse, Shipment, Supplier, TwinState

_VALID_SHIPMENT_STATUS = {"in_transit", "delivered", "delayed", "lost"}
DEFAULT_PLANNED_TRANSIT_DAYS = 7


# ---------------------------------------------------------------------------
# Twin construction
# ---------------------------------------------------------------------------

def build_live_twin(
    sim_start: str | None = None,
    rng_seed: int | None = None,
    demand_variability: float = 0.15,
    use_real_data: bool = True,
):
    """Constructs a real, live DigitalTwin. Uses the Kaggle dataset if it's
    present at digital-twin-fyp-main/data/DataCoSupplyChainDataset.csv,
    otherwise falls back to the twin's own synthetic generator — same
    fallback behavior data_loader.load_dataco() already has, just exposed
    here so callers don't need to know that module exists."""
    loader = twin_data_loader.load_dataco if use_real_data else twin_data_loader.generate_synthetic_state
    warehouses, shipments, suppliers, sim_date = loader(sim_start=sim_start)
    return DigitalTwin(
        warehouses=warehouses, shipments=shipments, suppliers=suppliers,
        sim_date=sim_date, demand_variability=demand_variability, rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# State conversion: their dataclass TwinState -> our pydantic TwinState
# ---------------------------------------------------------------------------

def to_pydantic_state(raw_state, default_planned_transit_days: int = DEFAULT_PLANNED_TRANSIT_DAYS) -> TwinState:
    """raw_state is whatever twin.state() returns (their dataclass TwinState).

    Field-mapping decisions:
      - Warehouse.daily_demand (their ONE demand number) populates BOTH our
        avg_daily_demand and recent_daily_demand with the same value.
        monitoring_agent.py is specifically designed to self-learn a real
        baseline from polling history rather than trust these being
        genuinely different — see that file's module docstring.
      - Shipment.planned_transit_days doesn't exist on their side. Derived
        from the origin supplier's lead_time_days when origin is a known
        supplier id; falls back to default_planned_transit_days otherwise
        (e.g. an internal reallocate-created shipment, or origin unknown).
      - Shipment.status: their field is a plain str, only ever actually set
        to "in_transit" or "delivered" by their own code. Passed through
        as-is if it's already one of our four valid values; otherwise
        inferred from delay_days as a safe fallback.
      - Shipment.off_route: always False. No such concept exists in the
        real twin yet — route_deviation detection stays dormant, as
        documented in monitoring_agent.py.
      - Supplier.lead_time_days: their side is a float, ours is an int —
        rounded here. Minor precision loss, acceptable for threshold checks.
    """
    supplier_lead_times = {s.id: s.lead_time_days for s in raw_state.suppliers}

    warehouses = [
        Warehouse(
            id=w.id,
            location=w.location,
            inventory=dict(w.inventory),
            safety_stock=dict(w.safety_stock),
            avg_daily_demand=dict(w.daily_demand),
            recent_daily_demand=dict(w.daily_demand),
        )
        for w in raw_state.warehouses
    ]

    shipments = []
    for s in raw_state.shipments:
        status = s.status if s.status in _VALID_SHIPMENT_STATUS else ("delayed" if s.delay_days > 0 else "in_transit")
        planned = supplier_lead_times.get(s.origin)
        planned_transit_days = round(planned) if planned is not None else default_planned_transit_days
        shipments.append(Shipment(
            id=s.id, origin=s.origin, destination=s.destination, eta=s.eta,
            status=status, delay_days=s.delay_days,
            planned_transit_days=max(planned_transit_days, 1),
            off_route=False,
        ))

    suppliers = [
        Supplier(id=s.id, lead_time_days=max(round(s.lead_time_days), 1), reliability_score=s.reliability_score)
        for s in raw_state.suppliers
    ]

    return TwinState(warehouses=warehouses, shipments=shipments, suppliers=suppliers)