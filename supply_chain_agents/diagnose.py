"""Instruments monitoring_agent.run() from the inside, called through the
exact same _resilient_call threaded path as the real graph, to see whether
detect_disruptions() or describe_event()/call_llm() is where events vanish."""
from __future__ import annotations
from schemas import TwinState
import twin_adapter
from agents import monitoring_agent
from orchestrator import _resilient_call, DEFAULT_CONFIG

monitoring_agent.reset_all_state()

twin = twin_adapter.build_live_twin(use_real_data=True, rng_seed=42)
injected = twin_adapter.inject_shipment_delay(twin, "SHP-482", severity="medium")
print(f"Injected: {injected.description}")

twin.advance(1.0)
snapshot = twin_adapter.to_pydantic_state(twin.state())
twin_dict = snapshot.model_dump()
reconstructed = TwinState(**twin_dict)

def instrumented_run(twin_state, config=monitoring_agent.DEFAULT_CONFIG):
    raw_events = monitoring_agent.detect_disruptions(twin_state, config)
    print(f"  [inside thread] detect_disruptions() found {len(raw_events)} event(s)")
    described = []
    for e in raw_events:
        try:
            d = monitoring_agent.describe_event(e, twin_state)
            described.append(d)
        except Exception as exc:
            print(f"  [inside thread] describe_event FAILED for {e.type}/{e.affected_id}: {type(exc).__name__}: {exc}")
    print(f"  [inside thread] describe_event succeeded for {len(described)}/{len(raw_events)}")
    return described

events = _resilient_call(instrumented_run, (reconstructed,), "detect", DEFAULT_CONFIG)
print(f"\nFinal result: {len(events)} event(s)")