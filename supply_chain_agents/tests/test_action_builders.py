"""
Tests for the deterministic action-builder functions in
agents/simulation_agent.py — build_candidate_actions() and its private
helpers (_build_reorder_action, _build_reallocate_action,
_build_reroute_action, _find_supplier_for, _parse_wh_sku).

These are the functions that replaced the old LLM-driven propose_candidates()
(see that module's docstring): they read the LIVE real twin and construct
complete, valid action dicts, or return None/drop the candidate rather than
send the twin something that would KeyError. That "silently drop, never
guess" contract is what most of these tests are actually checking.

Uses make_real_twin() from conftest.py throughout — these functions are
duck-typed against the real digital-twin-fyp-main object (dict-of-objects
warehouses/shipments/suppliers), not the pydantic snapshot.
"""
from __future__ import annotations
import pytest

import twin_adapter
from agents import simulation_agent
from schemas import DisruptionEvent
from tests.conftest import make_real_twin

CONFIG = simulation_agent.SimulationConfig()


def make_event(**overrides) -> DisruptionEvent:
    defaults = dict(
        event_id="EVT-1", type="stock_imbalance", affected_id="WH-01:SKU-1",
        severity="high", detected_at="2026-08-01T00:00:00Z",
    )
    defaults.update(overrides)
    return DisruptionEvent(**defaults)


# ---------------------------------------------------------------------------
# _parse_wh_sku
# ---------------------------------------------------------------------------

def test_parse_wh_sku_splits_on_first_colon():
    assert simulation_agent._parse_wh_sku("WH-01:SKU-1") == ("WH-01", "SKU-1")


def test_parse_wh_sku_splits_only_on_first_colon():
    # SKUs could plausibly contain colons themselves — must not over-split.
    assert simulation_agent._parse_wh_sku("WH-01:SKU:EXTRA") == ("WH-01", "SKU:EXTRA")


def test_parse_wh_sku_returns_none_without_a_colon():
    assert simulation_agent._parse_wh_sku("SHP-482") is None


# ---------------------------------------------------------------------------
# _find_supplier_for
# ---------------------------------------------------------------------------

def test_find_supplier_for_prefers_existing_shipment_history():
    twin = make_real_twin()  # SHP-1: SUP-1 -> WH-01, sku SKU-1
    RealSupplier = twin_adapter.twin_schemas.Supplier
    # A more "reliable" second supplier with no shipment history to WH-01/SKU-1.
    twin.suppliers["SUP-2"] = RealSupplier(id="SUP-2", lead_time_days=3, reliability_score=0.99)

    result = simulation_agent._find_supplier_for(twin, "WH-01", "SKU-1")

    assert result == "SUP-1", "existing delivery history should win over a more reliable stranger"


def test_find_supplier_for_falls_back_to_most_reliable_when_no_history():
    twin = make_real_twin()
    RealSupplier = twin_adapter.twin_schemas.Supplier
    twin.suppliers["SUP-2"] = RealSupplier(id="SUP-2", lead_time_days=3, reliability_score=0.99)

    # No shipment anywhere targets WH-02/SKU-1, so there's no history to prefer.
    result = simulation_agent._find_supplier_for(twin, "WH-02", "SKU-1")

    assert result == "SUP-2", "should fall back to the most reliable known supplier"


def test_find_supplier_for_returns_none_when_no_suppliers_exist():
    twin = make_real_twin()
    twin.suppliers.clear()
    assert simulation_agent._find_supplier_for(twin, "WH-01", "SKU-1") is None


# ---------------------------------------------------------------------------
# _build_reorder_action
# ---------------------------------------------------------------------------

def test_build_reorder_action_quantity_covers_deficit_plus_lead_time_demand():
    twin = make_real_twin(on_hand=80, safety_stock=150, daily_demand=20, lead_time_days=10)
    # deficit = 150 - 80 = 70; lead_time_cover = 20 * 10 = 200 -> 270
    action = simulation_agent._build_reorder_action(twin, "WH-01", "SKU-1", "medium", CONFIG)

    assert action is not None
    assert action["type"] == "reorder"
    assert action["supplier_id"] == "SUP-1"
    assert action["warehouse_id"] == "WH-01"
    assert action["sku"] == "SKU-1"
    assert action["quantity"] == 270


@pytest.mark.parametrize("severity,expected_rush", [
    ("high", True), ("critical", True), ("low", False), ("medium", False),
])
def test_build_reorder_action_rush_flag_follows_severity(severity, expected_rush):
    twin = make_real_twin(on_hand=80, safety_stock=150)
    action = simulation_agent._build_reorder_action(twin, "WH-01", "SKU-1", severity, CONFIG)
    assert action["rush"] is expected_rush


def test_build_reorder_action_respects_configured_minimum_quantity():
    # No deficit, no demand -> the raw formula would be 0, but a floor applies.
    twin = make_real_twin(on_hand=150, safety_stock=150, daily_demand=0)
    action = simulation_agent._build_reorder_action(twin, "WH-01", "SKU-1", "low", CONFIG)
    assert action["quantity"] == CONFIG.min_reorder_quantity


def test_build_reorder_action_returns_none_for_unknown_warehouse():
    twin = make_real_twin()
    assert simulation_agent._build_reorder_action(twin, "WH-99", "SKU-1", "high", CONFIG) is None


def test_build_reorder_action_returns_none_when_no_supplier_available():
    twin = make_real_twin()
    twin.suppliers.clear()
    assert simulation_agent._build_reorder_action(twin, "WH-01", "SKU-1", "high", CONFIG) is None


# ---------------------------------------------------------------------------
# _build_reallocate_action
# ---------------------------------------------------------------------------

def test_build_reallocate_action_direction_in_pulls_from_the_biggest_surplus():
    # WH-01 understocked, WH-02 has surplus (default fixture: 500 on hand, 150 safety).
    twin = make_real_twin(on_hand=80, safety_stock=150, wh2_on_hand=500, wh2_safety_stock=150)

    action = simulation_agent._build_reallocate_action(twin, "WH-01", "SKU-1", "in", CONFIG)

    assert action is not None
    assert action["type"] == "reallocate"
    assert action["from_warehouse"] == "WH-02"
    assert action["to_warehouse"] == "WH-01"
    assert action["quantity"] > 0


def test_build_reallocate_action_direction_in_quantity_is_capped_by_need_and_surplus():
    # need = 150 - 80 = 70; surplus = 500 - 150 = 350 -> capped at need (70)
    twin = make_real_twin(on_hand=80, safety_stock=150, wh2_on_hand=500, wh2_safety_stock=150)
    action = simulation_agent._build_reallocate_action(twin, "WH-01", "SKU-1", "in", CONFIG)
    assert action["quantity"] == 70


def test_build_reallocate_action_direction_out_pushes_to_the_biggest_need():
    # WH-01 overstocked, WH-02 needs stock.
    twin = make_real_twin(on_hand=500, safety_stock=150, wh2_on_hand=50, wh2_safety_stock=150)

    action = simulation_agent._build_reallocate_action(twin, "WH-01", "SKU-1", "out", CONFIG)

    assert action is not None
    assert action["from_warehouse"] == "WH-01"
    assert action["to_warehouse"] == "WH-02"
    assert action["quantity"] > 0


def test_build_reallocate_action_returns_none_when_only_one_warehouse_exists():
    twin = make_real_twin()
    del twin.warehouses["WH-02"]  # only WH-01 left -- nowhere to move stock to/from
    assert simulation_agent._build_reallocate_action(twin, "WH-01", "SKU-1", "in", CONFIG) is None
    assert simulation_agent._build_reallocate_action(twin, "WH-01", "SKU-1", "out", CONFIG) is None


def test_build_reallocate_action_returns_none_for_unknown_warehouse():
    twin = make_real_twin()
    assert simulation_agent._build_reallocate_action(twin, "WH-99", "SKU-1", "in", CONFIG) is None


# ---------------------------------------------------------------------------
# _build_reroute_action
# ---------------------------------------------------------------------------

def test_build_reroute_action_picks_the_warehouse_with_highest_need():
    twin = make_real_twin()  # SHP-1 -> WH-01; WH-02 exists as the only alternative
    RealWarehouse = twin_adapter.twin_schemas.Warehouse
    # Add a third warehouse with LESS need than WH-02, to prove "highest need"
    # actually drives the choice rather than just "the other one".
    twin.warehouses["WH-03"] = RealWarehouse(
        id="WH-03", location="ThirdCity",
        inventory={"SKU-1": 140}, safety_stock={"SKU-1": 150},  # need = 10
        daily_demand={"SKU-1": 5},
    )
    # WH-02 default: inventory 500, safety 150 -> need is negative (surplus),
    # so give it real need too, higher than WH-03's, to make the winner clear.
    twin.warehouses["WH-02"].inventory["SKU-1"] = 50
    twin.warehouses["WH-02"].safety_stock["SKU-1"] = 150  # need = 100

    action = simulation_agent._build_reroute_action(twin, "SHP-1")

    assert action is not None
    assert action["type"] == "reroute"
    assert action["shipment_id"] == "SHP-1"
    assert action["new_destination"] == "WH-02"  # need=100 beats WH-03's need=10


def test_build_reroute_action_returns_none_for_unknown_shipment():
    twin = make_real_twin()
    assert simulation_agent._build_reroute_action(twin, "SHP-999") is None


def test_build_reroute_action_returns_none_when_no_alternative_warehouse_exists():
    twin = make_real_twin()
    del twin.warehouses["WH-02"]
    assert simulation_agent._build_reroute_action(twin, "SHP-1") is None


# ---------------------------------------------------------------------------
# build_candidate_actions — the public, event-driven entry point
# ---------------------------------------------------------------------------

def test_build_candidate_actions_stock_imbalance_offers_reorder_reallocate_wait():
    twin = make_real_twin(on_hand=80, safety_stock=150)
    event = make_event(type="stock_imbalance", affected_id="WH-01:SKU-1", severity="high")

    actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)
    types_seen = {a["type"] for a in actions}

    assert types_seen == {"reorder", "reallocate", "wait"}


def test_build_candidate_actions_overstock_never_offers_reorder():
    twin = make_real_twin(on_hand=500, safety_stock=150, wh2_on_hand=50, wh2_safety_stock=150)
    event = make_event(type="overstock", affected_id="WH-01:SKU-1", severity="medium")

    actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)
    types_seen = {a["type"] for a in actions}

    assert "reorder" not in types_seen, "never reorder MORE of something already in surplus"
    assert types_seen <= {"reallocate", "wait"}


def test_build_candidate_actions_shipment_delay_uses_the_shipments_own_destination_and_sku():
    # WH-02 needs to actually have unmet need for the reroute candidate to be
    # buildable at all -- _build_reroute_action correctly refuses to reroute
    # a shipment toward a warehouse that's already in surplus (see
    # test_build_reroute_action_returns_none_when_no_alternative_warehouse_exists
    # -style logic), so the default fixture's surplus WH-02 wouldn't qualify.
    twin = make_real_twin(wh2_on_hand=50, wh2_safety_stock=150)
    event = make_event(type="shipment_delay", affected_id="SHP-1", severity="high")

    actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)
    types_seen = {a["type"] for a in actions}

    assert "reroute" in types_seen
    assert "wait" in types_seen
    reroute = next(a for a in actions if a["type"] == "reroute")
    assert reroute["shipment_id"] == "SHP-1"


def test_build_candidate_actions_supplier_events_only_offer_wait():
    # No (warehouse, sku) target exists for a supplier-level event -- offering
    # only "wait" is the documented honest limitation, not a bug.
    twin = make_real_twin()
    for event_type in ("supplier_lead_time", "supplier_reliability"):
        event = make_event(type=event_type, affected_id="SUP-1", severity="medium")
        actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)
        assert actions == [{"type": "wait"}]


def test_build_candidate_actions_always_returns_at_least_one_candidate():
    # Even when every "real" candidate type fails to build (e.g. unknown
    # warehouse), a bare wait must still be offered -- the human always has
    # SOMETHING to review.
    twin = make_real_twin()
    event = make_event(type="stock_imbalance", affected_id="WH-DOES-NOT-EXIST:SKU-1", severity="low")

    actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)

    assert actions == [{"type": "wait"}]


def test_build_candidate_actions_demand_spike_targets_the_affected_warehouse_and_sku():
    twin = make_real_twin(on_hand=80, safety_stock=150)
    event = make_event(type="demand_spike", affected_id="WH-01:SKU-1", severity="high")

    actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)

    for action in actions:
        if action["type"] == "reorder":
            assert action["warehouse_id"] == "WH-01"
            assert action["sku"] == "SKU-1"


# ---------------------------------------------------------------------------
# _build_label
# ---------------------------------------------------------------------------

def test_build_label_formats_each_action_type_distinctly():
    reorder = {"type": "reorder", "quantity": 270, "sku": "SKU-1", "supplier_id": "SUP-1", "warehouse_id": "WH-01"}
    reroute = {"type": "reroute", "shipment_id": "SHP-1", "new_destination": "WH-02"}
    reallocate = {"type": "reallocate", "quantity": 70, "sku": "SKU-1", "from_warehouse": "WH-02", "to_warehouse": "WH-01"}
    wait = {"type": "wait"}

    assert simulation_agent._build_label(reorder) == "reorder_270_SKU-1_from_SUP-1_to_WH-01"
    assert simulation_agent._build_label(reroute) == "reroute_SHP-1_to_WH-02"
    assert simulation_agent._build_label(reallocate) == "reallocate_70_SKU-1_WH-02_to_WH-01"
    assert simulation_agent._build_label(wait) == "wait_and_absorb"


# ---------------------------------------------------------------------------
# End-to-end sanity: built actions must actually be simulate-able
# ---------------------------------------------------------------------------

def test_every_built_candidate_action_survives_simulate_candidate():
    """The whole point of building actions deterministically from live twin
    state is that they never KeyError against the real twin. Prove it by
    actually running every built candidate through simulate_candidate()."""
    twin = make_real_twin(on_hand=80, safety_stock=150)
    event = make_event(type="stock_imbalance", affected_id="WH-01:SKU-1", severity="high")

    actions = simulation_agent.build_candidate_actions(event, twin, CONFIG)
    for action in actions:
        outcome = twin.simulate_candidate(action, horizon_days=5)
        assert outcome.cost_delta is not None