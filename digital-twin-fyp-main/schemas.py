"""
schemas.py
----------
Shared data contracts for the digital twin.

These mirror Section 7 of the project plan EXACTLY (field names and shapes),
because the Agent layer (Member B), the API layer (Member C), and the
frontend (Member D) will all serialize/deserialize against this shape.
Do not rename fields without telling the whole team - this is the contract
that stops "schema drift" (see Section 12 risks).

We use plain dataclasses (not pydantic) here so this module has zero
dependencies - Member C can still validate incoming JSON against these
shapes with pydantic in the FastAPI layer if they want stricter checking.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import itertools

_shipment_seq = itertools.count(1)
_event_seq = itertools.count(1)
_action_seq = itertools.count(1)


def next_shipment_id() -> str:
    return f"SHP-{next(_shipment_seq):03d}"


def next_event_id() -> str:
    return f"EVT-{next(_event_seq):03d}"


def next_action_id() -> str:
    return f"ACT-{next(_action_seq):03d}"


@dataclass
class Supplier:
    id: str
    lead_time_days: float
    reliability_score: float  # 0.0 - 1.0, chance a shipment ships on time


@dataclass
class Warehouse:
    id: str
    location: str
    inventory: Dict[str, int] = field(default_factory=dict)      # sku -> qty
    safety_stock: Dict[str, int] = field(default_factory=dict)   # sku -> qty
    # demand_rate is not in the plan's example schema (Section 7.1 shows a
    # snapshot, not the generative model) but the twin needs *some* driver
    # for consumption, so we carry it here. It's still serialized out
    # (harmless extra field), everything the plan explicitly requires is
    # present too.
    daily_demand: Dict[str, float] = field(default_factory=dict)  # sku -> units/day


@dataclass
class Shipment:
    id: str
    origin: str            # supplier id or warehouse id
    destination: str       # warehouse id
    eta: str                # ISO date string, e.g. "2026-07-12"
    status: str             # "in_transit" | "delivered" | "delayed"
    delay_days: int = 0
    sku: str = ""
    quantity: int = 0


@dataclass
class TwinState:
    """Matches Section 7.1 exactly (warehouses / shipments / suppliers),
    plus a sim_date so consumers know 'now' inside the twin, and a running
    metrics block used for Section 10 evaluation (stockout rate, cost,
    service level)."""
    warehouses: List[Warehouse] = field(default_factory=list)
    shipments: List[Shipment] = field(default_factory=list)
    suppliers: List[Supplier] = field(default_factory=list)
    sim_date: str = ""
    metrics: Dict[str, float] = field(default_factory=lambda: {
        "total_cost": 0.0,
        "stockout_days": 0,
        "orders_fulfilled_on_time": 0,
        "orders_total": 0,
    })

    def to_dict(self) -> dict:
        return {
            "warehouses": [asdict(w) for w in self.warehouses],
            "shipments": [asdict(s) for s in self.shipments],
            "suppliers": [asdict(sup) for sup in self.suppliers],
            "sim_date": self.sim_date,
            "metrics": self.metrics,
        }


@dataclass
class DisruptionEvent:
    """Matches Section 7.2 exactly."""
    event_id: str
    type: str          # "shipment_delay" | "demand_spike" | "stock_imbalance"
    affected_id: str
    severity: str        # "low" | "medium" | "high"
    detected_at: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionOutcome:
    """What the Simulation agent (Member B) gets back after running a
    candidate action forward through a forked twin. Feeds directly into
    the Decision agent's comparison + Section 7.3's justification."""
    action: str
    cost_delta: float
    delivery_delay_days: float
    stockout_risk: float      # 0.0 - 1.0
    service_level: float      # 0.0 - 1.0

    def to_dict(self) -> dict:
        return asdict(self)