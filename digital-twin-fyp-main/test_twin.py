"""
test_twin.py
------------
Quick sanity checks - not exhaustive, but they catch the mistakes that
would be most embarrassing in front of your professor:
  - forks actually being isolated from the live twin
  - "wait and absorb" actually being able to stock out
  - a disruption actually changing the twin's trajectory

Run with:  python -m pytest test_twin.py -v
"""

import pytest
from twin import DigitalTwin
from data_loader import generate_synthetic_state
from disruption_injector import inject_shipment_delay, inject_demand_spike


def fresh_twin(seed=1):
    wh, shp, sup, start = generate_synthetic_state(seed=seed)
    return DigitalTwin(wh, shp, sup, start, rng_seed=seed)


def test_fork_does_not_mutate_live_twin():
    twin = fresh_twin()
    live_cost_before = twin.metrics["total_cost"]
    live_inventory_before = dict(twin.warehouses["WH-01"].inventory)

    twin.simulate_candidate({"type": "reorder", "supplier_id": "SUP-07",
                              "warehouse_id": "WH-01", "sku": "SKU-100",
                              "quantity": 500}, horizon_days=10)

    assert twin.metrics["total_cost"] == live_cost_before
    assert twin.warehouses["WH-01"].inventory == live_inventory_before


def test_shipment_delay_disruption_increases_delay_days():
    twin = fresh_twin()
    shp_id = next(iter(twin.shipments))
    before = twin.shipments[shp_id].delay_days
    inject_shipment_delay(twin, shp_id)
    after = twin.shipments[shp_id].delay_days
    assert after > before


def test_demand_spike_increases_daily_demand():
    twin = fresh_twin()
    sku = "SKU-100"
    before = twin.warehouses["WH-01"].daily_demand[sku]
    inject_demand_spike(twin, sku)
    after = twin.warehouses["WH-01"].daily_demand[sku]
    assert after > before


def test_wait_action_can_produce_stockout_risk_when_demand_outpaces_stock():
    twin = fresh_twin()
    # crank demand way up so "wait" visibly risks a stockout
    twin.warehouses["WH-01"].daily_demand["SKU-100"] = 300
    outcome = twin.simulate_candidate({"type": "wait"}, horizon_days=5)
    assert outcome.stockout_risk > 0


def test_reorder_delivers_stock_into_destination_warehouse():
    twin = fresh_twin()
    before = twin.warehouses["WH-01"].inventory.get("SKU-100", 0)
    twin.place_order("SUP-08", "WH-01", "SKU-100", 200, rush=True)
    twin.advance(20)  # long enough for even a slow/unlucky delivery
    after = twin.warehouses["WH-01"].inventory.get("SKU-100", 0)
    assert after >= before  # inventory was replenished (net of demand consumed)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))