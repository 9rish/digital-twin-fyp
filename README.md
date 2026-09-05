# Agentic AI Digital Twin

A comprehensive supply chain digital twin governed by an autonomous AI orchestrator. Built for a Final Year Project to demonstrate how Multi-Agent AI systems can monitor, simulate, and resolve supply chain disruptions in real-time.

## 🌟 Key Features

### 1. Digital Twin Simulation Engine
A discrete-event simulation engine built with **SimPy** that models warehouses, inventory levels, shipments, and supplier reliability. It supports injecting real-world disruptions like demand spikes, shipment delays, and stock imbalances.

### 2. Multi-Agent Debate Engine
Powered by **LangGraph** and **Groq (Llama-3)**, the system doesn't just make single-shot decisions. When a disruption occurs, the orchestrator spawns multiple AI personas:
- 🚚 **Logistics Agent**: Argues for the safest option with the lowest stockout risk, ignoring costs.
- 💰 **Finance Agent**: Argues for the cheapest option to preserve profit margins.
- 👑 **Chief Orchestrator**: Reviews the debate and synthesizes a final, balanced recommendation.

### 3. "Sandbox" Future Simulation
Before recommending an action, the AI makes a *deep digital clone* of the live supply chain state. It tests every possible solution (e.g., reordering, reallocating, waiting) in this sandbox, fast-forwards time by 30 days, and records the exact mathematical cost and risk of each choice. The LLMs then use these hard facts to debate.

### 4. Predictive Demand Forecasting
A background algorithm calculates the moving-average demand and projects exactly how many days of usable inventory are left before a warehouse hits its critical safety stock, surfacing this in a color-coded UI.

### 5. Real-Time Dashboard
A beautiful, vanilla HTML/JS/CSS frontend connected to the backend via **WebSockets**. As time advances in the simulation, the dashboard updates instantly.

### 6. AI Observability
Fully integrated with **LangSmith** for real-time observability into the agent traces, token usage, and LLM reasoning waterfalls.

---

## 📁 Project Structure

- `backend/`: FastAPI application, SQLite audit database, and WebSocket manager.
- `frontend/`: Vanilla JS, HTML, and CSS for the real-time dashboard.
- `digital-twin-fyp-main/`: The core SimPy discrete-event simulation engine.
- `supply_chain_agents/`: The LangGraph state machine, agent prompts, and Twin adapter.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys) (Free)
- A [LangSmith API Key](https://smith.langchain.com/) (Free)

### 1. Setup the Environment
Clone the repository and install the dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file inside the `backend/` folder based on the example:
```bash
cp .env.example .env
```
Open `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=supply-chain-twin
```

### 3. Start the Backend Server
Run the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
The backend will start on `http://127.0.0.1:8000`.

### 4. Open the Dashboard
There is no complex frontend build step. Simply open the `frontend/index.html` file in your web browser:
- On Mac: `open ../frontend/index.html`
- Or simply drag and drop the `index.html` file into your Chrome/Safari tab.

### 5. Using the System
1. **Inject Disruption**: Click this button on the dashboard to introduce a random supply chain anomaly (e.g., a shipment delay or demand spike).
2. **Run Perceive Cycle**: Click this to trigger the LangGraph AI Orchestrator. It will scan the twin, detect the anomaly, simulate solutions, run the Multi-Agent Debate, and pop up a modal asking for your final approval.
3. **LangSmith Tracing**: Go to your LangSmith dashboard online to watch the agents debate and think in real-time!
