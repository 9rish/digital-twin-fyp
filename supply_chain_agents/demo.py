"""
Run this to see the full 7-stage agent loop fire, end-to-end, against the
REAL digital twin — once per disruption detected in this perceive cycle.

Usage:
    python demo.py            # auto-approves every recommendation
    python demo.py --reject   # rejects every recommendation instead
"""
from __future__ import annotations
import json
import sys

from langgraph.types import Command
from agents import monitoring_agent, simulation_agent
import twin_adapter
from orchestrator import build_graph


def pretty(label: str, obj) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    approve_choice = "--reject" not in sys.argv

    # Fresh run: don't inherit dedup/history/simulation-history state from a
    # prior demo invocation in the same process.
    monitoring_agent.reset_all_state()
    simulation_agent.reset_simulation_history()

    # use_real_data=True will use digital-twin-fyp-main/data/DataCoSupplyChainDataset.csv
    # if present, and fall back to the twin's own synthetic generator otherwise.
    twin = twin_adapter.build_live_twin(use_real_data=True, rng_seed=42)

    # Inject a real disruption so the demo actually exercises the full
    # pipeline (multiple candidates -> a real decision), instead of just
    # reporting whatever ambient state the baseline twin happens to be in.
    injected = twin_adapter.inject_shipment_delay(twin, "SHP-482", severity="medium")
    print(f"\n=== Injected disruption: {injected.description} ===")

    graph = build_graph(twin)
    config = {"configurable": {"thread_id": "demo-run-1"}}

    print("=== Perceive -> Detect ===")
    result = graph.invoke({}, config)

    event_num = 0
    while "__interrupt__" in result:
        event_num += 1
        interrupt_payload = result["__interrupt__"][0].value
        print(f"\n=== Disruption {event_num}: Simulate -> Recommend -> Approve ===")
        pretty("Recommendation awaiting human approval", interrupt_payload["recommendation"])
        print(f"--- Human decision: {'approve' if approve_choice else 'reject'} ---")
        result = graph.invoke(Command(resume=approve_choice), config)

    if event_num == 0:
        print("\nNo disruptions detected this cycle — nothing to act on.")
    else:
        pretty(f"All {event_num} recommendation(s) this cycle", result["recommendations"])
        pretty("Audit log", result["audit_log"])


if __name__ == "__main__":
    main()