"""
Monitoring / Exception agent — Section 4.1, now integrated against the real
digital twin (digital-twin-fyp-main), not just the twin_stub.

Input:  twin state snapshot (pydantic TwinState, produced by twin_adapter.py)
Output: list[DisruptionEvent]

REQUIRES the same schemas.py additions as before (Severity incl. "critical",
DisruptionType incl. overstock/dead_stock/fast_moving/lost_shipment/
route_deviation/supplier_lead_time/supplier_reliability, DisruptionEvent
.confidence, Shipment.off_route, Shipment.status incl. "lost").

v3 change — why it was needed:
  The real twin only tracks ONE demand number per SKU (`daily_demand`),
  not a separate "baseline average" and "recently observed" pair. If this
  agent kept comparing wh.avg_daily_demand against wh.recent_daily_demand
  the way it used to, both would always be identical against the real
  twin's adapted state — spike detection would silently never fire.

  Fix: demand-spike detection no longer trusts a twin-supplied average at
  all. It learns its own baseline from the SAME rolling history buffer
  already used for dead-stock/fast-moving/trend detection — the oldest
  points in that history become the baseline, the newest points become
  "recent." Until enough history has accumulated, it falls back to
  whatever avg_daily_demand the caller provided (useful for twin_stub.py,
  which genuinely does supply two different numbers, or for a cold start
  against the real twin before any polling history exists).

  Net effect: this file now works correctly against BOTH twin_stub.py and
  the real twin without either one needing to fake data it doesn't have.

Design tip from the plan (unchanged): keep the anomaly rules deterministic
and cheap. Reserve the LLM call for turning the raw anomaly into a readable
description.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
import time
import uuid

from schemas import TwinState, DisruptionEvent
from llm_client import call_llm

_ID_NAMESPACE = uuid.UUID("6f6f6f66-6f66-6f66-6f66-6f6f6f6f6f6f")

_SEVERITY_TIERS = ["critical", "high", "medium", "low"]   # checked in this order
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonitoringConfig:
    delay_pct_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.20, "medium": 0.50, "high": 1.00, "critical": 2.00}
    )
    understock_ratio_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 1.00, "medium": 0.85, "high": 0.50, "critical": 0.20}
    )
    overstock_ratio_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 1.50, "medium": 2.50, "high": 4.00, "critical": 6.00}
    )
    demand_spike_pct_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.15, "medium": 0.30, "high": 0.50, "critical": 1.00}
    )
    fast_moving_pct_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.20, "medium": 0.40, "high": 0.70, "critical": 1.20}
    )
    fast_moving_min_consecutive: int = 3

    dead_stock_demand_ratio: float = 0.05
    dead_stock_min_consecutive: int = 5

    # Demand-spike baseline learning (see module docstring "v3 change")
    demand_baseline_window: int = 5   # oldest N history points, averaged, become the baseline
    demand_recent_window: int = 2     # newest N history points, averaged, become "recent"
    # Below this much history, fall back to the caller-supplied avg_daily_demand
    # instead of a self-computed baseline (not enough data to trust yet).
    @property
    def demand_min_history_for_self_baseline(self) -> int:
        return self.demand_baseline_window + self.demand_recent_window

    supplier_expected_lead_time_days: float = 7.0
    supplier_lead_time_pct_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 1.20, "medium": 1.50, "high": 2.00, "critical": 3.00}
    )
    supplier_reliability_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.90, "medium": 0.80, "high": 0.65, "critical": 0.50}
    )

    trend_window: int = 5
    trend_min_points: int = 3
    trend_pct_per_day_threshold: float = 0.08

    history_max_len: int = 14
    alert_cooldown_seconds: float = 300.0


DEFAULT_CONFIG = MonitoringConfig()


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_last_alerted: dict[tuple[str, str], tuple[str, float]] = {}
_history: dict[tuple[str, str], list[tuple[float, int, float]]] = {}  # (ts, inventory, demand)


def reset_dedup_registry() -> None:
    _last_alerted.clear()


def reset_history() -> None:
    _history.clear()


def reset_all_state() -> None:
    reset_dedup_registry()
    reset_history()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _severity_from_ratio(ratio: float, thresholds: dict[str, float], higher_is_worse: bool) -> str | None:
    for sev in _SEVERITY_TIERS:
        threshold = thresholds[sev]
        if (higher_is_worse and ratio >= threshold) or (not higher_is_worse and ratio < threshold):
            return sev
    return None


def _confidence(ratio: float, boundary: float, higher_is_worse: bool) -> float:
    if boundary == 0:
        return 0.75
    excess = (ratio - boundary) / boundary if higher_is_worse else (boundary - ratio) / boundary
    excess = max(0.0, excess)
    return round(min(0.99, 0.6 + excess), 2)


def _should_emit(key: tuple[str, str], severity: str, cooldown: float, now: float) -> bool:
    prior = _last_alerted.get(key)
    if prior is None:
        _last_alerted[key] = (severity, now)
        return True
    prior_severity, prior_time = prior
    escalated = _SEVERITY_RANK[severity] > _SEVERITY_RANK[prior_severity]
    cooldown_expired = (now - prior_time) >= cooldown
    if escalated or cooldown_expired:
        _last_alerted[key] = (severity, now)
        return True
    return False


def _make_event_id(event_type: str, affected_id: str, severity: str) -> str:
    raw = f"{event_type}:{affected_id}:{severity}"
    return f"EVT-{uuid.uuid5(_ID_NAMESPACE, raw).hex[:8]}"


def _record_history(key: tuple[str, str], timestamp: float, inventory: int, demand: float, config: MonitoringConfig) -> None:
    series = _history.setdefault(key, [])
    series.append((timestamp, inventory, demand))
    if len(series) > config.history_max_len:
        del series[: len(series) - config.history_max_len]


# ---------------------------------------------------------------------------
# Inventory monitoring: understock, overstock, dead stock, fast-moving
# ---------------------------------------------------------------------------

def _understock_check(on_hand: int, safety_stock: int, config: MonitoringConfig) -> tuple[str, float] | None:
    if safety_stock == 0:
        return None
    ratio = on_hand / safety_stock
    sev = _severity_from_ratio(ratio, config.understock_ratio_thresholds, higher_is_worse=False)
    if not sev:
        return None
    return sev, _confidence(ratio, config.understock_ratio_thresholds[sev], higher_is_worse=False)


def _overstock_check(on_hand: int, safety_stock: int, config: MonitoringConfig) -> tuple[str, float] | None:
    if safety_stock == 0:
        return None
    ratio = on_hand / safety_stock
    sev = _severity_from_ratio(ratio, config.overstock_ratio_thresholds, higher_is_worse=True)
    if not sev:
        return None
    return sev, _confidence(ratio, config.overstock_ratio_thresholds[sev], higher_is_worse=True)


def _dead_stock_check(key: tuple[str, str], avg_demand: float, config: MonitoringConfig) -> tuple[str, float] | None:
    series = _history.get(key, [])
    if len(series) < config.dead_stock_min_consecutive or avg_demand <= 0:
        return None
    recent = series[-config.dead_stock_min_consecutive:]
    if all(demand <= avg_demand * config.dead_stock_demand_ratio for _, _, demand in recent):
        return "medium", 0.8
    return None


def _fast_moving_check(key: tuple[str, str], avg_demand: float, config: MonitoringConfig) -> tuple[str, float] | None:
    series = _history.get(key, [])
    if len(series) < config.fast_moving_min_consecutive or avg_demand <= 0:
        return None
    recent = series[-config.fast_moving_min_consecutive:]
    ratios = [(demand - avg_demand) / avg_demand for _, _, demand in recent]
    if any(r < config.fast_moving_pct_thresholds["low"] for r in ratios):
        return None
    worst_ratio = min(ratios)
    sev = _severity_from_ratio(worst_ratio, config.fast_moving_pct_thresholds, higher_is_worse=True)
    if not sev:
        return None
    return sev, _confidence(worst_ratio, config.fast_moving_pct_thresholds[sev], higher_is_worse=True)


def _demand_spike_check(
    key: tuple[str, str], current_recent: float, fallback_avg: float, config: MonitoringConfig
) -> tuple[str, float] | None:
    """Learns its own baseline from history once there's enough of it;
    falls back to the caller-supplied average before that. See the module
    docstring's 'v3 change' for why this replaced a direct field comparison."""
    series = _history.get(key, [])
    if len(series) >= config.demand_min_history_for_self_baseline:
        older = series[:-config.demand_recent_window]
        baseline_points = older[-config.demand_baseline_window:]
        recent_points = series[-config.demand_recent_window:]
        baseline = mean(d for _, _, d in baseline_points)
        recent = mean(d for _, _, d in recent_points)
    else:
        baseline = fallback_avg
        recent = current_recent

    if baseline <= 0:
        return None
    pct_increase = (recent - baseline) / baseline
    if pct_increase <= 0:
        return None
    sev = _severity_from_ratio(pct_increase, config.demand_spike_pct_thresholds, higher_is_worse=True)
    if not sev:
        return None
    return sev, _confidence(pct_increase, config.demand_spike_pct_thresholds[sev], higher_is_worse=True)


def _trend_check(key: tuple[str, str], config: MonitoringConfig) -> tuple[str, float, float] | None:
    """Early warning on inventory decline rate, ahead of the absolute
    stock-imbalance threshold."""
    series = _history.get(key, [])
    if len(series) < config.trend_min_points:
        return None
    window = series[-config.trend_window:]
    (t_start, inv_start, _), (t_end, inv_end, _) = window[0], window[-1]
    days = (t_end - t_start) / 86400.0
    if days <= 0 or inv_start <= 0:
        return None
    pct_per_day = (inv_start - inv_end) / inv_start / days
    if pct_per_day < config.trend_pct_per_day_threshold:
        return None
    confidence = _confidence(pct_per_day, config.trend_pct_per_day_threshold, higher_is_worse=True)
    return "low", confidence, pct_per_day


# ---------------------------------------------------------------------------
# Shipment monitoring: delay, lost, route deviation
#
# lost_shipment and route_deviation stay in this file even though the real
# twin currently has no data backing them (no "lost" status is ever set, no
# off_route flag exists) — they simply won't fire until/unless the twin
# adds that. Left in deliberately so nothing needs rewriting if it does.
# ---------------------------------------------------------------------------

def _delay_check(delay_days: int, planned_transit_days: int, config: MonitoringConfig) -> tuple[str, float] | None:
    if planned_transit_days <= 0 or delay_days <= 0:
        return None
    ratio = delay_days / planned_transit_days
    sev = _severity_from_ratio(ratio, config.delay_pct_thresholds, higher_is_worse=True)
    if not sev:
        return None
    return sev, _confidence(ratio, config.delay_pct_thresholds[sev], higher_is_worse=True)


# ---------------------------------------------------------------------------
# Supplier monitoring: lead time, reliability
# (Both fields exist directly on the real twin's Supplier — no gap here.)
# ---------------------------------------------------------------------------

def _supplier_lead_time_check(lead_time_days: float, config: MonitoringConfig) -> tuple[str, float] | None:
    if config.supplier_expected_lead_time_days <= 0:
        return None
    ratio = lead_time_days / config.supplier_expected_lead_time_days
    sev = _severity_from_ratio(ratio, config.supplier_lead_time_pct_thresholds, higher_is_worse=True)
    if not sev:
        return None
    return sev, _confidence(ratio, config.supplier_lead_time_pct_thresholds[sev], higher_is_worse=True)


def _supplier_reliability_check(reliability_score: float, config: MonitoringConfig) -> tuple[str, float] | None:
    sev = _severity_from_ratio(reliability_score, config.supplier_reliability_thresholds, higher_is_worse=False)
    if not sev:
        return None
    return sev, _confidence(reliability_score, config.supplier_reliability_thresholds[sev], higher_is_worse=False)


# ---------------------------------------------------------------------------
# Main detection pass
# ---------------------------------------------------------------------------

def detect_disruptions(twin: TwinState, config: MonitoringConfig = DEFAULT_CONFIG) -> list[DisruptionEvent]:
    events: list[DisruptionEvent] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts = time.time()

    def emit(event_type: str, affected_id: str, severity: str, confidence: float) -> None:
        key = (event_type, affected_id)
        if _should_emit(key, severity, config.alert_cooldown_seconds, now_ts):
            events.append(DisruptionEvent(
                event_id=_make_event_id(event_type, affected_id, severity),
                type=event_type,
                affected_id=affected_id,
                severity=severity,
                detected_at=now_iso,
                confidence=confidence,
            ))

    # ---- Shipment monitoring ----
    for shp in twin.shipments:
        if shp.status == "lost":
            emit("lost_shipment", shp.id, "critical", 0.95)
            continue

        delay_result = _delay_check(shp.delay_days, shp.planned_transit_days, config)
        if delay_result:
            emit("shipment_delay", shp.id, *delay_result)

        if shp.off_route:
            sev = "high" if shp.delay_days > 0 else "medium"
            emit("route_deviation", shp.id, sev, 0.8)

    # ---- Inventory + demand monitoring ----
    for wh in twin.warehouses:
        for sku, qty in wh.inventory.items():
            safety = wh.safety_stock.get(sku, 0)
            fallback_avg = wh.avg_daily_demand.get(sku, 0.0)
            recent_demand = wh.recent_daily_demand.get(sku, fallback_avg)
            key = (wh.id, sku)
            affected = f"{wh.id}:{sku}"

            _record_history(key, now_ts, qty, recent_demand, config)

            understock = _understock_check(qty, safety, config)
            if understock:
                emit("stock_imbalance", affected, *understock)

            overstock = _overstock_check(qty, safety, config)
            if overstock:
                emit("overstock", affected, *overstock)

            dead = _dead_stock_check(key, fallback_avg, config)
            if dead:
                emit("dead_stock", affected, *dead)

            fast = _fast_moving_check(key, fallback_avg, config)
            if fast:
                emit("fast_moving", affected, *fast)

            spike = _demand_spike_check(key, recent_demand, fallback_avg, config)
            if spike:
                emit("demand_spike", affected, *spike)

            trend = _trend_check(key, config)
            if trend and not understock:
                sev, conf, _pct_per_day = trend
                emit("stock_imbalance", affected, sev, conf)

    # ---- Supplier monitoring ----
    for sup in twin.suppliers:
        lead_time_result = _supplier_lead_time_check(sup.lead_time_days, config)
        if lead_time_result:
            emit("supplier_lead_time", sup.id, *lead_time_result)

        reliability_result = _supplier_reliability_check(sup.reliability_score, config)
        if reliability_result:
            emit("supplier_reliability", sup.id, *reliability_result)

    return events


# ---------------------------------------------------------------------------
# Natural-language explanation
# ---------------------------------------------------------------------------

_TYPE_DESCRIPTIONS = {
    "shipment_delay": "a shipment running behind schedule",
    "demand_spike": "a sudden jump in demand",
    "stock_imbalance": "inventory falling toward or below safety stock",
    "overstock": "inventory sitting well above safety stock",
    "dead_stock": "a SKU with little to no recent movement",
    "fast_moving": "a SKU with sustained demand well above baseline",
    "lost_shipment": "a shipment that has been lost in transit",
    "route_deviation": "a shipment deviating from its planned route",
    "supplier_lead_time": "a supplier's lead time running longer than expected",
    "supplier_reliability": "a supplier with a degraded reliability score",
}


def describe_event(event: DisruptionEvent, twin: TwinState) -> DisruptionEvent:
    fallback = (
        f"{event.type.replace('_', ' ').title()} detected on {event.affected_id} "
        f"(severity: {event.severity}, confidence: {event.confidence * 100:.0f}%)."
    )
    prompt = (
        "You are a supply chain monitoring assistant. In one short plain-language "
        "sentence, describe this disruption for a human planner reading a dashboard. "
        "Be concrete and factual, no speculation.\n\n"
        f"Event type: {event.type} — {_TYPE_DESCRIPTIONS.get(event.type, '')}\n"
        f"Affected: {event.affected_id}\n"
        f"Severity: {event.severity}\n"
        f"Confidence: {event.confidence * 100:.0f}%"
    )
    event.description = call_llm(prompt, fallback=fallback)
    return event


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run(twin: TwinState, config: MonitoringConfig = DEFAULT_CONFIG) -> list[DisruptionEvent]:
    """Signature unchanged from before — orchestrator.py doesn't need to change
    how it calls this."""
    events = detect_disruptions(twin, config)
    return [describe_event(e, twin) for e in events]


def poll_forever(
    get_state_fn,
    config: MonitoringConfig = DEFAULT_CONFIG,
    interval_seconds: float = 30.0,
    on_events=None,
    max_iterations: int | None = None,
) -> None:
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        try:
            twin = get_state_fn()
            events = run(twin, config)
            if events and on_events:
                on_events(events)
        except Exception as exc:  # noqa: BLE001 - a single bad poll must not stop monitoring
            print(f"[monitoring_agent] poll failed, will retry next interval: {exc}")
        iterations += 1
        if max_iterations is None or iterations < max_iterations:
            time.sleep(interval_seconds)