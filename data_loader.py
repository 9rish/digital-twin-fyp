"""
data_loader.py
---------------
Member A's third deliverable: "Load and shape the Kaggle DataCo dataset
into the twin's initial state" (Section 8).

The public Kaggle "DataCo Smart Supply Chain" dataset is a big flat CSV of
individual order-line records (one row per product per order), not a
warehouse/supplier state file - so "loading" it really means *aggregating*
it into the Section 7.1 shape: a handful of warehouses (by region/market),
a handful of suppliers (by product category), and a starting set of
in-transit shipments.

Get the dataset here (download the CSV yourself, this environment has no
internet access to Kaggle):
  https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
Expected file: DataCoSupplyChainDataset.csv
Put it at: data/DataCoSupplyChainDataset.csv (relative to this folder).

Known relevant columns in that CSV (verify against your actual download -
Kaggle CSV column names occasionally vary by mirror):
  'Order Region', 'Market', 'Product Name', 'Category Name',
  'Order Item Quantity', 'Product Price', 'Days for shipping (real)',
  'Days for shipment (scheduled)', 'Late_delivery_risk', 'order date (DateOrders)'

If the file isn't found, `load_dataco()` falls back to a synthetic-but-
realistic generator so the rest of the team is never blocked (Section 12
risk: "one module runs late / dataset missing").
"""

from __future__ import annotations
import csv
import os
import random
from collections import defaultdict
from datetime import date, timedelta
from typing import List, Tuple

from schemas import Warehouse, Shipment, Supplier

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DataCoSupplyChainDataset.csv")

# Column name candidates, since Kaggle mirrors sometimes differ slightly.
COLUMN_CANDIDATES = {
    "region": ["Order Region", "Order_Region"],
    "quantity": ["Order Item Quantity", "Order_Item_Quantity"],
    "price": ["Product Price", "Product_Price"],
    "category": ["Category Name", "Category_Name"],
    "ship_days_real": ["Days for shipping (real)", "Days_for_shipping_real"],
    "ship_days_scheduled": ["Days for shipment (scheduled)", "Days_for_shipment_scheduled"],
    "late_risk": ["Late_delivery_risk"],
}


def _resolve_columns(fieldnames: List[str]) -> dict:
    resolved = {}
    for key, candidates in COLUMN_CANDIDATES.items():
        for c in candidates:
            if c in fieldnames:
                resolved[key] = c
                break
    return resolved


def load_dataco(sim_start: str | None = None, max_rows: int = 50_000
                ) -> Tuple[List[Warehouse], List[Shipment], List[Supplier], str]:
    """Returns (warehouses, shipments, suppliers, sim_date) ready to hand
    straight to DigitalTwin(**...). Falls back to synthetic data if the
    Kaggle CSV isn't present at data/DataCoSupplyChainDataset.csv."""
    sim_start = sim_start or date.today().isoformat()

    if not os.path.exists(DATA_PATH):
        print(f"[data_loader] {DATA_PATH} not found - using synthetic dataset instead. "
              f"Download the Kaggle DataCo CSV and place it there to use real data.")
        return generate_synthetic_state(sim_start=sim_start)

    region_qty = defaultdict(int)
    region_price_sum = defaultdict(float)
    region_price_n = defaultdict(int)
    category_leadtime = defaultdict(list)
    category_ontime = defaultdict(list)

    with open(DATA_PATH, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        cols = _resolve_columns(reader.fieldnames or [])
        missing = [k for k in ("region", "quantity") if k not in cols]
        if missing:
            print(f"[data_loader] Couldn't find expected columns {missing} in "
                  f"{DATA_PATH} - falling back to synthetic dataset. "
                  f"Check COLUMN_CANDIDATES against your CSV header.")
            return generate_synthetic_state(sim_start=sim_start)

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            region = row.get(cols["region"], "Unknown") or "Unknown"
            try:
                qty = int(float(row.get(cols["quantity"], 0) or 0))
            except ValueError:
                qty = 0
            region_qty[region] += qty

            if "price" in cols:
                try:
                    region_price_sum[region] += float(row.get(cols["price"], 0) or 0)
                    region_price_n[region] += 1
                except ValueError:
                    pass

            category = row.get(cols.get("category", ""), "General") or "General"
            if "ship_days_real" in cols and "ship_days_scheduled" in cols:
                try:
                    real = float(row.get(cols["ship_days_real"], 0) or 0)
                    sched = float(row.get(cols["ship_days_scheduled"], 0) or 0)
                    category_leadtime[category].append(max(real, sched, 1))
                except ValueError:
                    pass
            if "late_risk" in cols:
                try:
                    category_ontime[category].append(1 - float(row.get(cols["late_risk"], 0) or 0))
                except ValueError:
                    pass

    if not region_qty:
        print("[data_loader] CSV parsed but no usable rows found - using synthetic dataset.")
        return generate_synthetic_state(sim_start=sim_start)

    # --- build warehouses: top regions by volume ---
    top_regions = sorted(region_qty.items(), key=lambda kv: -kv[1])[:5]
    warehouses = []
    for i, (region, total_qty) in enumerate(top_regions, start=1):
        wh_id = f"WH-{i:02d}"
        daily_demand_est = max(round(total_qty / 365), 5)
        sku = f"SKU-{100 + i}"
        warehouses.append(Warehouse(
            id=wh_id,
            location=region,
            inventory={sku: daily_demand_est * 20},
            safety_stock={sku: daily_demand_est * 7},
            daily_demand={sku: daily_demand_est},
        ))

    # --- build suppliers: one per product category with data ---
    suppliers = []
    for i, category in enumerate(sorted(category_leadtime.keys())[:6], start=1):
        lead_times = category_leadtime[category]
        avg_lead = sum(lead_times) / len(lead_times) if lead_times else 7
        ontime_scores = category_ontime.get(category, [0.85])
        reliability = max(0.5, min(0.99, sum(ontime_scores) / len(ontime_scores)))
        suppliers.append(Supplier(
            id=f"SUP-{i:02d}",
            lead_time_days=round(avg_lead, 1),
            reliability_score=round(reliability, 2),
        ))
    if not suppliers:
        suppliers = [Supplier(id="SUP-01", lead_time_days=9, reliability_score=0.86)]

    # --- a couple of starting in-transit shipments so the demo has motion ---
    shipments = []
    rng = random.Random(42)
    for i, wh in enumerate(warehouses[:3], start=1):
        sku = next(iter(wh.daily_demand))
        eta = (date.fromisoformat(sim_start) + timedelta(days=rng.randint(3, 10))).isoformat()
        shipments.append(Shipment(
            id=f"SHP-{i:03d}",
            origin=suppliers[i % len(suppliers)].id,
            destination=wh.id,
            eta=eta,
            status="in_transit",
            delay_days=0,
            sku=sku,
            quantity=wh.daily_demand[sku] * 15,
        ))

    return warehouses, shipments, suppliers, sim_start


def generate_synthetic_state(sim_start: str | None = None, seed: int = 7
                              ) -> Tuple[List[Warehouse], List[Shipment], List[Supplier], str]:
    """Realistic-looking placeholder data so every teammate can build and
    test against a live twin from day one, with or without the Kaggle CSV
    present (Section 9, Week 2-3: 'build in parallel against stubs')."""
    sim_start = sim_start or date.today().isoformat()
    rng = random.Random(seed)
    start = date.fromisoformat(sim_start)

    skus = ["SKU-100", "SKU-101", "SKU-102"]
    warehouses = [
        Warehouse(
            id="WH-01", location="Mumbai",
            inventory={"SKU-100": 420, "SKU-101": 260},
            safety_stock={"SKU-100": 150, "SKU-101": 100},
            daily_demand={"SKU-100": 25, "SKU-101": 15},
        ),
        Warehouse(
            id="WH-02", location="Chennai",
            inventory={"SKU-100": 300, "SKU-102": 180},
            safety_stock={"SKU-100": 120, "SKU-102": 60},
            daily_demand={"SKU-100": 18, "SKU-102": 10},
        ),
        Warehouse(
            id="WH-03", location="Delhi",
            inventory={"SKU-101": 200, "SKU-102": 150},
            safety_stock={"SKU-101": 80, "SKU-102": 50},
            daily_demand={"SKU-101": 12, "SKU-102": 9},
        ),
    ]

    suppliers = [
        Supplier(id="SUP-07", lead_time_days=9, reliability_score=0.86),
        Supplier(id="SUP-08", lead_time_days=5, reliability_score=0.93),
        Supplier(id="SUP-09", lead_time_days=12, reliability_score=0.75),
    ]

    shipments = [
        Shipment(
            id="SHP-482", origin="SUP-07", destination="WH-01",
            eta=(start + timedelta(days=4)).isoformat(),
            status="in_transit", delay_days=0, sku="SKU-100", quantity=300,
        ),
        Shipment(
            id="SHP-483", origin="SUP-08", destination="WH-02",
            eta=(start + timedelta(days=2)).isoformat(),
            status="in_transit", delay_days=0, sku="SKU-102", quantity=180,
        ),
        Shipment(
            id="SHP-484", origin="SUP-09", destination="WH-03",
            eta=(start + timedelta(days=6)).isoformat(),
            status="in_transit", delay_days=0, sku="SKU-101", quantity=150,
        ),
    ]

    return warehouses, shipments, suppliers, sim_start