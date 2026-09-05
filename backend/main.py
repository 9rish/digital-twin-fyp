import sys
import os
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the supply_chain_agents directory to sys.path so we can import from it
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "supply_chain_agents"))

from orchestrator import build_graph
import twin_adapter
from database import init_db, save_audit_log, get_all_audit_logs
from websocket import manager

# Global objects holding the simulation state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup on startup
    init_db()
    
    # Initialize the live digital twin (use_real_data=True, random seed for varied startups)
    twin = twin_adapter.build_live_twin(use_real_data=True, rng_seed=None)
    # Build the LangGraph orchestrator closing over the twin
    graph = build_graph(twin)
    
    # Store globally for our endpoints to use
    app_state["twin"] = twin
    app_state["graph"] = graph
    # thread_id gives LangGraph a memory namespace so it can be interrupted/resumed
    app_state["config"] = {"configurable": {"thread_id": "api-run"}}
    app_state["awaiting_approval"] = False
    
    yield
    # Cleanup on shutdown (if any)

app = FastAPI(lifespan=lifespan)

# Allow frontend to connect from any origin (configure properly in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't necessarily expect messages from the frontend, but we need to keep connection open
            # and listen for potential disconnects.
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- REST Endpoints ---

@app.get("/api/twin/state")
async def get_twin_state():
    """Returns the current state of the digital twin simulation."""
    twin = app_state["twin"]
    # We use the twin_adapter to convert the live twin into a JSON-serializable Pydantic model
    state = twin_adapter.to_pydantic_state(twin.state())
    return state.model_dump()

@app.post("/api/orchestrator/perceive")
async def trigger_perceive_cycle():
    """Triggers the agent pipeline to detect disruptions and formulate recommendations."""
    graph = app_state["graph"]
    config = app_state["config"]
    
    # Run the graph. It will run through Perceive -> Detect -> Simulate -> Recommend
    # and then hit the `interrupt()` in the `approve` node if an event is found.
    result = graph.invoke({}, config)
    
    # Check if there is a recommendation pending approval
    pending = None
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        pending = interrupt_payload.get("recommendation")
    
    app_state["awaiting_approval"] = bool(pending)
    
    # Prepare the response
    response_data = {
        "status": "awaiting_approval" if pending else "completed",
        "pending_recommendation": pending,
        # The result itself has the state if it finished without interrupts (e.g. no events)
        "graph_state": result if not pending else None
    }
    
    # Broadcast to all connected clients that the cycle ran
    await manager.broadcast({"type": "cycle_update", "data": response_data})
    
    return response_data

class ApprovalRequest(BaseModel):
    decision: bool | str  # True, False, or "cancel"

@app.post("/api/orchestrator/approve")
async def approve_recommendation(req: ApprovalRequest):
    """
    Submits the human decision to the orchestrator.
    If the decision is True (approve), the action is executed.
    If False (reject), the graph will try to offer an alternative.
    """
    from langgraph.types import Command
    graph = app_state["graph"]
    config = app_state["config"]
    
    # Resume the graph with the human decision
    # The graph will proceed to `act` (or `offer_alternative`)
    result = graph.invoke(Command(resume=req.decision), config)
    
    pending = None
    if "__interrupt__" in result:
        # If it offered an alternative, it interrupted again
        interrupt_payload = result["__interrupt__"][0].value
        pending = interrupt_payload.get("recommendation")
    else:
        # It successfully acted. We should persist the newly appended audit log entry.
        if "audit_log" in result and len(result["audit_log"]) > 0:
            latest_log = result["audit_log"][-1]
            save_audit_log(latest_log)

    app_state["awaiting_approval"] = bool(pending)

    response_data = {
        "status": "awaiting_approval" if pending else "completed",
        "pending_recommendation": pending,
        "graph_state": result if not pending else None
    }

    # Broadcast state change
    await manager.broadcast({"type": "decision_update", "data": response_data})
    
    return response_data

class DisruptionInjectionRequest(BaseModel):
    severity: str = "medium"

@app.post("/api/test/inject-disruption")
async def inject_disruption():
    """Injects a random disruption into the twin for testing."""
    import traceback
    import dataclasses
    twin = app_state["twin"]
    try:
        injected = twin_adapter.random_disruption(twin)
        return {"status": "success", "event": dataclasses.asdict(injected)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/api/audit-logs")
async def get_audit_logs():
    """Retrieves all past actions/decisions from the SQLite database."""
    logs = get_all_audit_logs()
    return {"logs": logs}

# --- Serve Frontend ---
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend"), html=True), name="frontend")

