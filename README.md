# Digital Twin Engine 

This is the SimPy-based digital twin for the Agentic AI Supply Chain
Exception Management project — Section 8, Member A's responsibilities:

- SimPy simulation engine (warehouses, in-transit shipments, suppliers)
- Synthetic disruption injector
- Kaggle DataCo dataset loader (with a synthetic fallback so no one is
  blocked while waiting on the CSV)
- The twin state schema (Section 7.1), owned jointly with the team

## Files

| File | What it does |
|---|---|
| `schemas.py` | Data shapes: `Warehouse`, `Shipment`, `Supplier`, `TwinState`, `DisruptionEvent`, `ActionOutcome`. Matches Section 7 of the project plan. |
| `twin.py` | The `DigitalTwin` class — the SimPy engine itself, plus `fork()` and `simulate_candidate()`. |
| `disruption_injector.py` | Manual + random disruption generators, matching Section 7.2. |
| `data_loader.py` | Loads `data/DataCoSupplyChainDataset.csv` and shapes it into initial twin state, or generates realistic synthetic data if the CSV isn't there yet. |
| `demo.py` | Runs the whole thing end to end — proof it works, and a reference for how the agent layer should call it. |
| `test_twin.py` | Sanity checks (fork isolation, disruptions actually biting, stockouts actually happening). |

## Setup

```bash
pip install -r requirements.txt
python demo.py                    # see it run
python -m pytest test_twin.py -v  # sanity checks
```

## Using the real Kaggle dataset

Download `DataCoSupplyChainDataset.csv` from:
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Put it at `data/DataCoSupplyChainDataset.csv`. `load_dataco()` will pick it
up automatically. If the column names on your download differ slightly
from what's expected, check `COLUMN_CANDIDATES` in `data_loader.py` —
that's the one place to fix it.

Until then, everything works against `generate_synthetic_state()`
automatically — nobody on the team needs to wait for the dataset.

## The interface Member B (agents) should build against

```python
from twin import DigitalTwin
from data_loader import load_dataco

warehouses, shipments, suppliers, sim_date = load_dataco()
twin = DigitalTwin(warehouses, shipments, suppliers, sim_date)

# Perceive
state = twin.state().to_dict()          # read current state, JSON-able

# Detect - your Monitoring agent looks at `state` and decides something's
# wrong, then (for real disruptions used in the demo) you can also call:
from disruption_injector import inject_shipment_delay
event = inject_shipment_delay(twin, "SHP-482")   # returns a DisruptionEvent

# Simulate - test 2-3 candidates, NEVER touches the live twin
outcome_wait   = twin.simulate_candidate({"type": "wait"}, horizon_days=7)
outcome_reorder = twin.simulate_candidate(
    {"type": "reorder", "supplier_id": "SUP-07", "warehouse_id": "WH-01",
     "sku": "SKU-100", "quantity": 300, "rush": True},
    horizon_days=7,
)
# outcome_* is an ActionOutcome: .cost_delta, .delivery_delay_days,
# .stockout_risk, .service_level

# Recommend - your Decision agent picks the best outcome and writes the
# plain-language justification (Section 4.3) - that logic lives in your
# module, not this one.

# Act - once a human approves, apply the SAME action for real:
twin.place_order("SUP-07", "WH-01", "SKU-100", 300, rush=True)
```

Supported live actions (mirror the ones usable inside `simulate_candidate`):
`twin.place_order(...)`, `twin.reroute_shipment(...)`, `twin.reallocate_stock(...)`.

## The interface Member C (backend/API) should build against

- `twin.state().to_dict()` → JSON payload for `GET /state` and the
  WebSocket push.
- Wrap `inject_shipment_delay` / `inject_demand_spike` /
  `inject_stock_imbalance` behind the "inject disruption" dashboard button.
- Wrap `twin.simulate_candidate(...)` behind the what-if panel's
  "simulate" button — return `ActionOutcome.to_dict()`.
- Wrap `twin.place_order` / `reroute_shipment` / `reallocate_stock` behind
  the approval endpoint, gated on human approval as in Section 4.4.

## Cost model (tune these together as a team before Section 10 evaluation)

All in `twin.py`, top of file:
`HOLDING_COST_PER_UNIT_DAY`, `STOCKOUT_COST_PER_UNIT_DAY`,
`RUSH_ORDER_PREMIUM`, `NORMAL_UNIT_COST`, `REROUTE_HANDLING_COST`,
`INTERNAL_TRANSFER_COST_PER_UNIT`. These are placeholder magnitudes —
recalibrate them against the Kaggle dataset's real price/quantity
distributions once it's loaded (per Section 12's mitigation: "Calibrate
initial state against the Kaggle dataset's real demand/lead-time
distributions before building agents on top").

## Known simplifications (be upfront about these in your writeup)

- Demand is a noisy-but-stationary daily rate per SKU per warehouse, not a
  learned forecast (Prophet/ARIMA integration per Section 5's tech stack
  is a next step, feeding into `Warehouse.daily_demand`).
- `stock_imbalance` disruptions are flagged, not "injected" — imbalance is
  a state condition (Monitoring agent's job to detect), not something we
  mutate into existence.
- Reliability/delay on new orders is a simple probability roll, not
  correlated with anything else in the twin yet.