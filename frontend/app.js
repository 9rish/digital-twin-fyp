const API_BASE = 'http://127.0.0.1:8000/api';
const WS_URL = 'ws://127.0.0.1:8000/ws';

// DOM Elements
const warehouseGrid = document.getElementById('warehouse-grid');
const shipmentGrid = document.getElementById('shipment-grid');
const btnRun = document.getElementById('btn-run');
const btnInject = document.getElementById('btn-inject');

// Modal Elements
const agentModal = document.getElementById('agent-modal');
const disruptionBadge = document.getElementById('disruption-badge');
const justificationText = document.getElementById('justification-text');
const costImpact = document.getElementById('cost-impact');
const delayImpact = document.getElementById('delay-impact');
const btnApprove = document.getElementById('btn-approve');
const btnReject = document.getElementById('btn-reject');
const toastContainer = document.getElementById('toast-container');

const supplierGrid = document.getElementById('supplier-grid');

// State
let socket;
let lastState = null;

// Initialization
async function init() {
    await fetchTwinState();
    connectWebSocket();
    setupEventListeners();
}

// ... keeping WebSocket and EventListener code the same ...
function connectWebSocket() {
    socket = new WebSocket(WS_URL);
    socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'cycle_update' || message.type === 'decision_update') {
            handleOrchestratorUpdate(message.data);
            fetchTwinState(); // Refresh twin state UI
        }
    };
    socket.onclose = () => {
        showToast("WebSocket disconnected. Reconnecting in 3s...");
        setTimeout(connectWebSocket, 3000);
    };
}

function setupEventListeners() {
    btnRun.addEventListener('click', async () => {
        btnRun.disabled = true;
        btnRun.textContent = "Perceiving...";
        try {
            await fetch(`${API_BASE}/orchestrator/perceive`, { method: 'POST' });
            showToast("Perceive cycle ran successfully.");
        } catch (e) {
            showToast("Error running cycle.");
        } finally {
            btnRun.disabled = false;
            btnRun.textContent = "Run Perceive Cycle";
        }
    });

    btnInject.addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE}/test/inject-disruption`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await res.json();
            if (data.status === "success") {
                showToast(`⚠️ Disruption Injected: ${data.event.type} on ${data.event.affected_id}`);
                fetchTwinState();
            } else {
                showToast("Error injecting disruption.");
            }
        } catch (e) {
            showToast("Error injecting disruption.");
        }
    });

    btnApprove.addEventListener('click', () => sendDecision(true));
    btnReject.addEventListener('click', () => sendDecision(false));
}

// Fetch & Render State
async function fetchTwinState() {
    try {
        const res = await fetch(`${API_BASE}/twin/state`);
        const state = await res.json();
        lastState = state;
        renderWarehouses(state.warehouses);
        renderShipments(state.shipments);
        renderSuppliers(state.suppliers);
    } catch (e) {
        console.error("Failed to fetch twin state", e);
    }
}

function renderWarehouses(warehouses) {
    warehouseGrid.innerHTML = warehouses.map(w => {
        const totalItems = Object.values(w.inventory).reduce((a, b) => a + b, 0);
        return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">${w.id} - ${w.location}</span>
            </div>
            <div class="data-row">
                <span class="data-label">Total Inventory</span>
                <span class="data-value">${totalItems.toLocaleString()} units</span>
            </div>
            <hr style="border: 0; border-top: 1px solid var(--border); margin: 0.5rem 0;">
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem;">SKU (Stock | Forecast)</div>
            ${Object.keys(w.inventory).slice(0, 3).map(sku => {
                const stock = w.inventory[sku];
                const avgDemand = w.avg_daily_demand[sku];
                const recentDemand = w.recent_daily_demand[sku];
                const isDemandSpike = recentDemand > (avgDemand * 1.2); // 20% spike threshold
                
                // FYP Phase 2: Predictive Forecasting
                const daysUntilStockout = w.predicted_stockout_days ? w.predicted_stockout_days[sku] : 999;
                const forecastColor = daysUntilStockout <= 3 ? 'var(--danger)' : (daysUntilStockout <= 7 ? 'var(--warning)' : 'var(--success)');
                
                return `
                <div class="data-row">
                    <span class="data-label">${sku}</span>
                    <span class="data-value" style="display: flex; gap: 8px; align-items: center;">
                        <span>${stock}</span>
                        <span style="font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; background: ${forecastColor}20; color: ${forecastColor}; font-weight: bold;">
                            ${daysUntilStockout}d left
                        </span>
                    </span>
                </div>
            `}).join('')}
        </div>
    `}).join('');
}

function renderShipments(shipments) {
    shipmentGrid.innerHTML = shipments.map(s => {
        let statusClass = "status-good";
        if (s.status === "delayed") statusClass = "status-error";
        else if (s.status === "in_transit") statusClass = "status-warn";
        
        return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">${s.id}</span>
                <span class="status-indicator ${statusClass}">${s.status.toUpperCase()}</span>
            </div>
            <div class="data-row">
                <span class="data-label">Route</span>
                <span class="data-value">${s.origin} ➔ ${s.destination}</span>
            </div>
            <div class="data-row">
                <span class="data-label">ETA</span>
                <span class="data-value">${s.eta}</span>
            </div>
            ${s.delay_days > 0 ? `
            <div class="data-row">
                <span class="data-label" style="color:var(--danger)">Delay</span>
                <span class="data-value" style="color:var(--danger)">+${s.delay_days} days</span>
            </div>` : ''}
        </div>
    `}).join('');
}

function renderSuppliers(suppliers) {
    supplierGrid.innerHTML = suppliers.map(s => {
        // Visual thresholds for disruptions
        const isLeadTimeHigh = s.lead_time_days > 7; 
        const isReliabilityLow = s.reliability_score < 0.85;

        return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">${s.id}</span>
                <span class="status-indicator ${(isLeadTimeHigh || isReliabilityLow) ? 'status-error' : 'status-good'}">
                    ${(isLeadTimeHigh || isReliabilityLow) ? 'DISRUPTED' : 'STABLE'}
                </span>
            </div>
            <div class="data-row">
                <span class="data-label">Lead Time</span>
                <span class="data-value" style="color: ${isLeadTimeHigh ? 'var(--danger)' : 'inherit'}; font-weight: ${isLeadTimeHigh ? 'bold' : 'normal'}">
                    ${s.lead_time_days} days ${isLeadTimeHigh ? '⚠️' : ''}
                </span>
            </div>
            <div class="data-row">
                <span class="data-label">Reliability</span>
                <span class="data-value" style="color: ${isReliabilityLow ? 'var(--danger)' : 'inherit'}; font-weight: ${isReliabilityLow ? 'bold' : 'normal'}">
                    ${(s.reliability_score * 100).toFixed(0)}% ${isReliabilityLow ? '⚠️' : ''}
                </span>
            </div>
        </div>
    `}).join('');
}

// Agent Command Center Logic
function handleOrchestratorUpdate(data) {
    if (data.status === 'awaiting_approval' && data.pending_recommendation) {
        const rec = data.pending_recommendation;
        const chosen = rec.all_candidates.find(c => c.action === rec.chosen_action) || rec.all_candidates[0];
        
        disruptionBadge.textContent = rec.risk_level.toUpperCase() + " RISK";
        justificationText.innerHTML = `<strong>Recommended Action:</strong> ${rec.chosen_action.replace(/_/g, ' ').toUpperCase()}<br><br>${rec.justification}`;
        
        if (chosen) {
            costImpact.textContent = `₹${chosen.cost_delta.toLocaleString()}`;
            delayImpact.textContent = `${chosen.delivery_delay_days} days`;
        }

        agentModal.classList.remove('hidden');
    } else {
        agentModal.classList.add('hidden');
    }
}

async function sendDecision(isApproved) {
    btnApprove.disabled = true;
    btnReject.disabled = true;
    try {
        await fetch(`${API_BASE}/orchestrator/approve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ decision: isApproved })
        });
        showToast(isApproved ? "Action Approved & Executed" : "Action Rejected - Seeking Alternative");
    } catch (e) {
        showToast("Failed to submit decision.");
    } finally {
        btnApprove.disabled = false;
        btnReject.disabled = false;
    }
}

function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// Start
init();
