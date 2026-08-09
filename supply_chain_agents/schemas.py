"""
Shared schemas — Section 7 of the project plan.
These are the contracts every module (twin, agents, backend, frontend) agrees on.
Freeze these early (Week 1) to avoid the "schema drift" risk called out in Section 12.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- 7.1 Twin state ----------

class Warehouse(BaseModel):
    id: str
    location: str
    inventory: dict[str, int]          # SKU -> qty on hand
    safety_stock: dict[str, int]       # SKU -> safety stock level
    avg_daily_demand: dict[str, float] = Field(default_factory=dict)     # SKU -> baseline daily demand
    recent_daily_demand: dict[str, float] = Field(default_factory=dict) # SKU -> observed demand (e.g. last 3-day avg)


class Shipment(BaseModel):
    id: str
    origin: str
    destination: str
    eta: str                            # ISO date
    status: Literal["in_transit", "delivered", "delayed", "lost"]
    delay_days: int = 0
    planned_transit_days: int = 5       # used to express delay as % of planned lead time, per Section 4.1
    off_route: bool = False             # flagged by the twin/telemetry when a shipment deviates from its planned route


class Supplier(BaseModel):
    id: str
    lead_time_days: int
    reliability_score: float


class TwinState(BaseModel):
    warehouses: list[Warehouse]
    shipments: list[Shipment]
    suppliers: list[Supplier]


# ---------- 7.2 Disruption event ----------

Severity = Literal["low", "medium", "high", "critical"]
DisruptionType = Literal[
    "shipment_delay", "demand_spike", "stock_imbalance",        # original three
    "overstock", "dead_stock", "fast_moving",                   # inventory-velocity categories
    "lost_shipment", "route_deviation",                         # shipment categories beyond simple delay
    "supplier_lead_time", "supplier_reliability",                # supplier categories
]


class DisruptionEvent(BaseModel):
    event_id: str
    type: DisruptionType
    affected_id: str
    severity: Severity
    detected_at: str                    # ISO timestamp
    description: Optional[str] = None   # filled in by the LLM reasoning pass
    confidence: float = 1.0             # 0.0-1.0, how confident the detection is


# ---------- 7.3 Action / recommendation ----------

ActionType = Literal["reroute", "reorder", "reallocate", "wait", "split_shipment", "other"]


class CandidateOutcome(BaseModel):
    action: str                         # human-readable label, e.g. "reroute_via_WH-02"
    action_type: ActionType = "other"   # machine-readable category — drives forecasting logic
    cost_delta: float
    delivery_delay_days: float
    stockout_risk: float                # 0.0 - 1.0
    service_level_impact: float         # 0.0 - 1.0 (fraction of orders on-time)
    inventory_impact: float = 0.0       # estimated net inventory units moved/added (+) or absorbed (-)
    profit_impact: float = 0.0          # estimated profit delta (cost + lost-sales effect), negative = worse
    resource_utilization: float = 0.0   # 0.0 - 1.0, estimated ops capacity consumed by this action
    action_params: dict = Field(default_factory=dict)  # exact structured action, for real replay in orchestrator's act()


class ActionRecommendation(BaseModel):
    action_id: str
    event_id: str
    chosen_action: str
    candidates_considered: int
    justification: str
    status: Literal["pending_approval", "approved", "rejected"] = "pending_approval"
    all_candidates: list[CandidateOutcome] = Field(default_factory=list)
    constraint_satisfied: bool = True   # False if no candidate met the configured (soft) risk ceiling
    confidence: float = 0.8             # 0.0-1.0, how clearly the winner beat the runner-up
    risk_level: Literal["low", "medium", "high"] = "medium"   # display classification for the dashboard
    ranking: list[str] = Field(default_factory=list)                       # action labels, best -> worst
    business_rules_satisfied: bool = True                                   # False if NO candidate cleared the hard rules
    excluded_by_rules: dict[str, list[str]] = Field(default_factory=dict)   # action -> violated hard-rule names