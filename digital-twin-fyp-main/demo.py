"""
demo.py
-------
Standalone proof that the twin works end-to-end. Run this directly:

    python demo.py

It shows the shape of what your Member A slice hands off to Member B:
  1. build a twin from data (synthetic fallback if no Kaggle CSV yet)
  2. let it run for a few days
  3. inject a disruption
  4. generate 3 candidate actions and simulate each on a FORK
  5. print a before/after comparison table (this is basically what the
     Decision agent and the "what-if simulation panel" both consume)

This isn't the agent layer itself (that's Member B's job) - it's here to
prove your engine's public API is usable and to give the team something
concrete to build against.
"""

import json
from twin import DigitalTwin
from data_loader import load_dataco
from disruption_injector import inject_shipment_delay, inject_demand_spike


def main():
    warehouses, shipments, suppliers, sim_start = load_dataco()
    twin = DigitalTwin(warehouses, shipments, suppliers, sim_start, rng_seed=1)

    print(f"=== Twin initialized at {twin.sim_date} ===")
    print(json.dumps(twin.state().to_dict(), indent=2)[:800], "...\n")

    # let a few days pass normally
    twin.advance(2)
    print(f"=== After 2 days: {twin.sim_date}, cost so far: "
          f"{twin.metrics['total_cost']:.2f} ===\n")

    # --- Perceive/Detect (stand-in for Member B's Monitoring agent) ---
    target_shipment = next(iter(twin.shipments))
    event = inject_shipment_delay(twin, target_shipment, severity="medium")
    print(f"=== Disruption injected: {event.to_dict()} ===\n")

    # --- Simulate: 3 candidate actions run on forks, live twin untouched ---
    shp = twin.shipments[target_shipment]
    candidates = [
        {"type": "wait"},
        {"type": "reorder", "supplier_id": next(iter(twin.suppliers)),
         "warehouse_id": shp.destination, "sku": shp.sku, "quantity": shp.quantity,
         "rush": True},
        {"type": "reallocate", "from_warehouse": list(twin.warehouses)[1]
            if len(twin.warehouses) > 1 else shp.destination,
         "to_warehouse": shp.destination, "sku": shp.sku,
         "quantity": max(shp.quantity // 3, 10)},
    ]

    print("=== Candidate outcomes (each run on an isolated fork) ===")
    results = []
    for action in candidates:
        outcome = twin.simulate_candidate(action, horizon_days=7)
        results.append(outcome)
        print(f"  {outcome.action:12s} | cost_delta: {outcome.cost_delta:>9.2f} | "
              f"delay: {outcome.delivery_delay_days:>4.1f}d | "
              f"stockout_risk: {outcome.stockout_risk:>4.2f} | "
              f"service_level: {outcome.service_level:>4.2f}")

    # sanity check: live twin's cost must be unaffected by the forks above
    print(f"\n=== Live twin cost after forks (should be unchanged): "
          f"{twin.metrics['total_cost']:.2f} ===")

    # --- Act: apply the "best" candidate for real (Decision agent's job
    # in the full system - here we just pick lowest cost_delta as a stub) ---
    best = min(results, key=lambda o: o.cost_delta)
    print(f"\n=== (stub) best candidate by cost: {best.action} -> applying to live twin ===")
    twin.advance(7)
    print(f"Live twin after advancing 7 more days: {twin.sim_date}, "
          f"total_cost={twin.metrics['total_cost']:.2f}, "
          f"stockout_days={twin.metrics['stockout_days']}")


if __name__ == "__main__":
    main()