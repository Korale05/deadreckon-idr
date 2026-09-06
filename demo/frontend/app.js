/**
 * app.js — DeadReckon Demo Frontend Logic
 * ========================================
 * Handles: session listing, pipeline execution, trajectory canvas rendering,
 * metric animation, live simulation, and system info display.
 */

const API_BASE = window.location.origin;

// ============ STATE ============
const state = {
    sessions: [],
    currentSession: null,
    trajectories: null,
    metrics: null,
    isRunning: false,
    canvas: {
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        isDragging: false,
        lastX: 0,
        lastY: 0,
    },
    layers: {
        gt: true,
        ins: true,
        ai: true,
        ekf: true,
        bo: true,
    },
};

// ============ DOM REFS ============
const $ = (id) => document.getElementById(id);

const dom = {
    status: $("statusIndicator"),
    statusDot: null,
    statusText: null,
    sessionSelect: $("sessionSelect"),
    blackoutSlider: $("blackoutSlider"),
    blackoutValue: $("blackoutValue"),
    btnRun: $("btnRunPipeline"),
    btnLive: $("btnLiveDemo"),
    progressContainer: $("progressContainer"),
    progressBar: $("progressBar"),
    progressText: $("progressText"),
    canvas: $("trajectoryCanvas"),
    canvasContainer: $("canvasContainer"),
    canvasOverlay: $("canvasOverlay"),
    metricsGrid: $("metricsGrid"),
    sysinfoContent: $("sysinfoContent"),
    legendContainer: $("trajectoryLegend"),
};

// ============ INITIALIZATION ============
document.addEventListener("DOMContentLoaded", () => {
    dom.statusDot = dom.status.querySelector(".status-dot");
    dom.statusText = dom.status.querySelector(".status-text");

    initCanvas();
    initControls();
    loadSessions();
    loadSystemInfo();
});

// ============ API HELPERS ============
async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;
    try {
        const resp = await fetch(url, {
            headers: { "Content-Type": "application/json" },
            ...options,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        return await resp.json();
    } catch (e) {
        console.error(`API error: ${path}`, e);
        throw e;
    }
}

// ============ SESSIONS ============
async function loadSessions() {
    try {
        const data = await apiFetch("/api/sessions");
        state.sessions = data.sessions;
        setOnline(true);

        dom.sessionSelect.innerHTML = "";
        if (data.sessions.length === 0) {
            dom.sessionSelect.innerHTML = '<option value="">No sessions found</option>';
            return;
        }

        data.sessions.forEach((s) => {
            const opt = document.createElement("option");
            opt.value = s.name;
            opt.textContent = `${s.name} (${s.category})`;
            dom.sessionSelect.appendChild(opt);
        });

        state.currentSession = data.sessions[0].name;
        dom.btnRun.disabled = false;
        dom.btnLive.disabled = false;
    } catch (e) {
        setOnline(false);
        dom.sessionSelect.innerHTML = '<option value="">Failed to connect</option>';
    }
}

function setOnline(online) {
    dom.statusDot.className = "status-dot" + (online ? " online" : " error");
    dom.statusText.textContent = online ? "Backend Online" : "Disconnected";
}

// ============ CONTROLS ============
function initControls() {
    dom.sessionSelect.addEventListener("change", (e) => {
        state.currentSession = e.target.value;
    });

    dom.blackoutSlider.addEventListener("input", (e) => {
        dom.blackoutValue.textContent = `${e.target.value}s`;
    });

    dom.btnRun.addEventListener("click", runPipeline);
    dom.btnLive.addEventListener("click", runLiveSimulation);

    // Canvas controls
    $("btnZoomIn").addEventListener("click", () => {
        state.canvas.zoom *= 1.3;
        renderTrajectory();
    });
    $("btnZoomOut").addEventListener("click", () => {
        state.canvas.zoom /= 1.3;
        renderTrajectory();
    });
    $("btnResetView").addEventListener("click", () => {
        state.canvas.zoom = 1;
        state.canvas.offsetX = 0;
        state.canvas.offsetY = 0;
        renderTrajectory();
    });

    // Legend checkboxes
    dom.legendContainer.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.addEventListener("change", (e) => {
            state.layers[e.target.dataset.layer] = e.target.checked;
            renderTrajectory();
        });
    });
}

// ============ RUN PIPELINE ============
async function runPipeline() {
    if (state.isRunning) return;
    state.isRunning = true;

    const session = dom.sessionSelect.value;
    if (!session) return;

    dom.btnRun.disabled = true;
    dom.btnLive.disabled = true;
    showProgress("Running pipeline...", 10);

    try {
        showProgress("Processing INS + AI + EKF...", 30);

        const data = await apiFetch(`/api/run/${session}`, {
            method: "POST",
            body: JSON.stringify({
                blackout_duration: parseFloat(dom.blackoutSlider.value),
                downsample: 5,
            }),
        });

        showProgress("Rendering results...", 80);

        state.trajectories = data.trajectories;
        state.metrics = data.metrics;

        // Render
        updateMetrics(data.metrics);
        dom.canvasOverlay.classList.add("hidden");
        renderTrajectory();

        showProgress("Complete!", 100);
        setTimeout(hideProgress, 1500);
    } catch (e) {
        showProgress(`Error: ${e.message}`, 0);
        setTimeout(hideProgress, 3000);
    } finally {
        state.isRunning = false;
        dom.btnRun.disabled = false;
        dom.btnLive.disabled = false;
    }
}

// ============ LIVE SIMULATION ============
async function runLiveSimulation() {
    if (state.isRunning) return;
    state.isRunning = true;

    const session = dom.sessionSelect.value;
    if (!session) return;

    dom.btnRun.disabled = true;
    dom.btnLive.disabled = true;
    showProgress("Starting live simulation...", 10);

    try {
        showProgress("Fetching IMU simulation data...", 30);

        const data = await apiFetch("/api/simulate-phone", {
            method: "POST",
            body: JSON.stringify({
                session_name: session,
                n_samples: 2000,
                downsample: 2,
            }),
        });

        showProgress("Animating trajectory...", 50);
        dom.canvasOverlay.classList.add("hidden");

        // Animate point by point
        await animateLiveTrajectory(data);

        showProgress("Simulation complete!", 100);
        setTimeout(hideProgress, 1500);
    } catch (e) {
        showProgress(`Error: ${e.message}`, 0);
        setTimeout(hideProgress, 3000);
    } finally {
        state.isRunning = false;
        dom.btnRun.disabled = false;
        dom.btnLive.disabled = false;
    }
}

async function animateLiveTrajectory(data) {
    const canvas = dom.canvas;
    const ctx = canvas.getContext("2d");
    const { ins_trajectory, ground_truth, timestamps } = data;
    const n = ins_trajectory.length;

    // Compute bounds from GT
    const gtX = ground_truth.pos_x;
    const gtY = ground_truth.pos_y;
    const insX = ins_trajectory.map((p) => p.px);
    const insY = ins_trajectory.map((p) => p.py);

    const allX = [...gtX, ...insX];
    const allY = [...gtY, ...insY];
    const bounds = computeBounds(allX, allY);

    return new Promise((resolve) => {
        let frame = 0;
        const batchSize = 3;

        function tick() {
            frame += batchSize;
            if (frame >= n) {
                frame = n;
                resolve();
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGrid(ctx, canvas.width, canvas.height, bounds);

            // Draw GT so far
            drawPath(ctx, gtX.slice(0, frame), gtY.slice(0, frame), bounds, canvas, "#ffffff", 2, 0.6);

            // Draw INS so far
            drawPath(
                ctx,
                insX.slice(0, frame),
                insY.slice(0, frame),
                bounds,
                canvas,
                "#ff4757",
                2,
                0.8
            );

            // Current position dot (INS)
            if (frame > 0 && frame <= n) {
                const idx = Math.min(frame - 1, n - 1);
                const [cx, cy] = worldToScreen(insX[idx], insY[idx], bounds, canvas);
                ctx.beginPath();
                ctx.arc(cx, cy, 5, 0, Math.PI * 2);
                ctx.fillStyle = "#ff4757";
                ctx.fill();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Progress
            const pct = Math.round((frame / n) * 100);
            showProgress(`Live simulation: ${pct}%`, 50 + pct * 0.5);

            if (frame < n) {
                requestAnimationFrame(tick);
            }
        }

        requestAnimationFrame(tick);
    });
}

// ============ METRICS ============
function updateMetrics(m) {
    animateMetric("metricDuration", m.duration_str, "");
    animateMetric("metricInsDrift", `${m.drift_pct.ins.toFixed(2)}%`, `Final error: ${m.final_error_m.ins.toFixed(1)} m`, "highlight-red");
    animateMetric("metricAiDrift", m.drift_pct.ai !== null ? `${m.drift_pct.ai.toFixed(2)}%` : "N/A", "No EKF/GNSS", "highlight-blue");
    animateMetric("metricEkfDrift", `${m.drift_pct.ekf_full.toFixed(4)}%`, `Final error: ${m.final_error_m.ekf_full.toFixed(2)} m`, "highlight-green");
    animateMetric("metricBoDrift", `${m.drift_pct.ekf_blackout.toFixed(4)}%`, `${m.blackout.duration_s}s GNSS outage`, "highlight-orange");
    animateMetric("metricPosRmse", `${m.pos_rmse_m.ins.toFixed(1)} → ${m.pos_rmse_m.ekf_full.toFixed(1)} m`, `${((1 - m.pos_rmse_m.ekf_full / m.pos_rmse_m.ins) * 100).toFixed(1)}% improvement`);
    animateMetric("metricVelRmse", `${m.vel_rmse_ms.ins.toFixed(3)} → ${m.vel_rmse_ms.ekf_full.toFixed(3)} m/s`, "INS vs EKF velocity");
    animateMetric("metricFinalErr", `${m.final_error_m.ekf_full.toFixed(2)} m`, `From ${m.final_error_m.ins.toFixed(0)} m INS error`, "highlight-green");
}

function animateMetric(id, value, subtitle, highlightClass) {
    const card = $(id);
    if (!card) return;

    const valEl = card.querySelector(".metric-value");
    const subEl = card.querySelector(".metric-sub");

    valEl.className = "metric-value animating" + (highlightClass ? ` ${highlightClass}` : "");
    valEl.textContent = value;

    if (subtitle && subEl) {
        subEl.textContent = subtitle;
    }

    // Remove animation class after it completes
    setTimeout(() => valEl.classList.remove("animating"), 450);
}

// ============ PROGRESS ============
function showProgress(text, pct) {
    dom.progressContainer.style.display = "block";
    dom.progressBar.style.width = `${pct}%`;
    dom.progressText.textContent = text;
}

function hideProgress() {
    dom.progressContainer.style.display = "none";
    dom.progressBar.style.width = "0%";
}

// ============ CANVAS / TRAJECTORY ============
function initCanvas() {
    const canvas = dom.canvas;
    const container = dom.canvasContainer;

    function resize() {
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + "px";
        canvas.style.height = rect.height + "px";
        const ctx = canvas.getContext("2d");
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        if (state.trajectories) renderTrajectory();
    }

    resize();
    window.addEventListener("resize", resize);

    // Pan
    canvas.addEventListener("mousedown", (e) => {
        state.canvas.isDragging = true;
        state.canvas.lastX = e.clientX;
        state.canvas.lastY = e.clientY;
    });
    canvas.addEventListener("mousemove", (e) => {
        if (!state.canvas.isDragging) return;
        state.canvas.offsetX += e.clientX - state.canvas.lastX;
        state.canvas.offsetY += e.clientY - state.canvas.lastY;
        state.canvas.lastX = e.clientX;
        state.canvas.lastY = e.clientY;
        renderTrajectory();
    });
    canvas.addEventListener("mouseup", () => (state.canvas.isDragging = false));
    canvas.addEventListener("mouseleave", () => (state.canvas.isDragging = false));

    // Zoom
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        state.canvas.zoom *= delta;
        renderTrajectory();
    });
}

function computeBounds(xArr, yArr) {
    const minX = Math.min(...xArr);
    const maxX = Math.max(...xArr);
    const minY = Math.min(...yArr);
    const maxY = Math.max(...yArr);
    const pad = 0.08;
    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    return {
        minX: minX - rangeX * pad,
        maxX: maxX + rangeX * pad,
        minY: minY - rangeY * pad,
        maxY: maxY + rangeY * pad,
    };
}

function worldToScreen(wx, wy, bounds, canvas) {
    const cw = canvas.width / window.devicePixelRatio;
    const ch = canvas.height / window.devicePixelRatio;
    const rangeX = bounds.maxX - bounds.minX;
    const rangeY = bounds.maxY - bounds.minY;

    // Fit to canvas with aspect ratio preserved
    const scale = Math.min(cw / rangeX, ch / rangeY) * state.canvas.zoom;
    const cx = cw / 2 + state.canvas.offsetX;
    const cy = ch / 2 + state.canvas.offsetY;
    const midX = (bounds.minX + bounds.maxX) / 2;
    const midY = (bounds.minY + bounds.maxY) / 2;

    const sx = cx + (wx - midX) * scale;
    const sy = cy - (wy - midY) * scale; // flip Y
    return [sx, sy];
}

function drawGrid(ctx, w, h, bounds) {
    const cw = w / window.devicePixelRatio;
    const ch = h / window.devicePixelRatio;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, cw, ch);

    // Grid lines
    ctx.strokeStyle = "rgba(48, 54, 61, 0.5)";
    ctx.lineWidth = 0.5;
    const gridSpacing = 50;

    for (let x = 0; x < cw; x += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, ch);
        ctx.stroke();
    }
    for (let y = 0; y < ch; y += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(cw, y);
        ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = "rgba(139, 148, 158, 0.5)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillText("East →", cw - 50, ch - 8);
    ctx.fillText("North ↑", 8, 14);
}

function drawPath(ctx, xArr, yArr, bounds, canvas, color, lineWidth, alpha) {
    if (!xArr || xArr.length < 2) return;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.globalAlpha = alpha;

    const [sx, sy] = worldToScreen(xArr[0], yArr[0], bounds, canvas);
    ctx.moveTo(sx, sy);

    for (let i = 1; i < xArr.length; i++) {
        const [px, py] = worldToScreen(xArr[i], yArr[i], bounds, canvas);
        ctx.lineTo(px, py);
    }

    ctx.stroke();
    ctx.globalAlpha = 1;
}

function renderTrajectory() {
    const t = state.trajectories;
    if (!t) return;

    const canvas = dom.canvas;
    const ctx = canvas.getContext("2d");

    // Collect all points for bounds
    const allX = [...t.ground_truth.pos_x];
    const allY = [...t.ground_truth.pos_y];

    if (state.layers.ins) {
        allX.push(...t.ins_only.pos_x);
        allY.push(...t.ins_only.pos_y);
    }

    const bounds = computeBounds(allX, allY);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid(ctx, canvas.width, canvas.height, bounds);

    // Draw trajectories (back to front)
    if (state.layers.bo && t.ekf_blackout) {
        drawPath(ctx, t.ekf_blackout.pos_x, t.ekf_blackout.pos_y, bounds, canvas, "#f59e0b", 1.5, 0.6);
    }
    if (state.layers.ekf && t.ekf_full) {
        drawPath(ctx, t.ekf_full.pos_x, t.ekf_full.pos_y, bounds, canvas, "#3fb950", 2, 0.85);
    }
    if (state.layers.ai && t.ai_corrected) {
        drawPath(ctx, t.ai_corrected.pos_x, t.ai_corrected.pos_y, bounds, canvas, "#3b82f6", 1.5, 0.7);
    }
    if (state.layers.ins) {
        drawPath(ctx, t.ins_only.pos_x, t.ins_only.pos_y, bounds, canvas, "#ff4757", 1.5, 0.6);
    }
    if (state.layers.gt) {
        drawPath(ctx, t.ground_truth.pos_x, t.ground_truth.pos_y, bounds, canvas, "#ffffff", 2, 0.8);
    }

    // Start/end markers
    drawMarker(ctx, t.ground_truth.pos_x[0], t.ground_truth.pos_y[0], bounds, canvas, "#3fb950", "S");
    const lastIdx = t.ground_truth.pos_x.length - 1;
    drawMarker(ctx, t.ground_truth.pos_x[lastIdx], t.ground_truth.pos_y[lastIdx], bounds, canvas, "#ff4757", "E");

    // Scale indicator
    drawScaleBar(ctx, bounds, canvas);
}

function drawMarker(ctx, wx, wy, bounds, canvas, color, label) {
    const [sx, sy] = worldToScreen(wx, wy, bounds, canvas);

    ctx.beginPath();
    ctx.arc(sx, sy, 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.font = "bold 10px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, sx, sy);
}

function drawScaleBar(ctx, bounds, canvas) {
    const cw = canvas.width / window.devicePixelRatio;
    const ch = canvas.height / window.devicePixelRatio;
    const rangeX = bounds.maxX - bounds.minX;
    const scale = Math.min(cw / rangeX, ch / (bounds.maxY - bounds.minY)) * state.canvas.zoom;

    // Find a nice round distance
    const barPixels = 100;
    const barMeters = barPixels / scale;

    let niceMeters;
    if (barMeters > 5000) niceMeters = 5000;
    else if (barMeters > 2000) niceMeters = 2000;
    else if (barMeters > 1000) niceMeters = 1000;
    else if (barMeters > 500) niceMeters = 500;
    else if (barMeters > 200) niceMeters = 200;
    else if (barMeters > 100) niceMeters = 100;
    else if (barMeters > 50) niceMeters = 50;
    else niceMeters = 10;

    const nicePixels = niceMeters * scale;

    const x = 20;
    const y = ch - 30;

    ctx.strokeStyle = "rgba(139, 148, 158, 0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + nicePixels, y);
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x, y + 4);
    ctx.moveTo(x + nicePixels, y - 4);
    ctx.lineTo(x + nicePixels, y + 4);
    ctx.stroke();

    ctx.fillStyle = "rgba(139, 148, 158, 0.8)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    const label = niceMeters >= 1000 ? `${niceMeters / 1000} km` : `${niceMeters} m`;
    ctx.fillText(label, x + nicePixels / 2, y - 8);
}

// ============ SYSTEM INFO ============
async function loadSystemInfo() {
    try {
        const info = await apiFetch("/api/system-info");

        dom.sysinfoContent.innerHTML = `
            <div class="sysinfo-card">
                <h3>AI Model</h3>
                <div class="sysinfo-row"><span class="sysinfo-key">Architecture</span><span class="sysinfo-val">${info.model.architecture}</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">IMU Window</span><span class="sysinfo-val">${info.model.input_imu_window} samples (${info.model.window_seconds}s)</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">Sample Rate</span><span class="sysinfo-val">${info.model.sample_rate_hz} Hz</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">IMU Features</span><span class="sysinfo-val">${info.model.input_imu_features.join(", ")}</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">Output</span><span class="sysinfo-val">${info.model.output.join(", ")}</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">Model Loaded</span><span class="sysinfo-val ${info.model_loaded ? 'good' : 'warn'}">${info.model_loaded ? '✓ Yes' : '✗ No'}</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">TFLite Export</span><span class="sysinfo-val ${info.tflite_available ? 'good' : 'warn'}">${info.tflite_available ? '✓ Available' : '✗ Not found'}</span></div>
            </div>
            <div class="sysinfo-card">
                <h3>Extended Kalman Filter</h3>
                <div class="sysinfo-row"><span class="sysinfo-key">State Vector</span><span class="sysinfo-val">[${info.ekf.state_vector.join(', ')}]</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">Q (pos std)</span><span class="sysinfo-val">${info.ekf.q_pos_std} m</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">Q (vel std)</span><span class="sysinfo-val">${info.ekf.q_vel_std} m/s</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">R (GNSS pos)</span><span class="sysinfo-val">${info.ekf.r_gnss_pos_std} m</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">R (AI vel)</span><span class="sysinfo-val">${info.ekf.r_ai_vel_std} m/s</span></div>
                <div class="sysinfo-row"><span class="sysinfo-key">Innovation Gate</span><span class="sysinfo-val">χ² ≤ ${info.ekf.innovation_gate_chi2}</span></div>
            </div>
            <div class="sysinfo-card">
                <h3>Sign Convention</h3>
                <div class="sysinfo-row" style="flex-direction:column; align-items:flex-start; gap:4px;">
                    <span class="sysinfo-val" style="color:var(--accent-cyan); font-size:0.72rem; word-break:break-all;">${info.sign_convention}</span>
                </div>
                <div style="margin-top:10px;">
                    <div class="sysinfo-row"><span class="sysinfo-key">Dataset</span><span class="sysinfo-val ${info.dataset_exists ? 'good' : 'warn'}">${info.dataset_exists ? '✓ Found' : '✗ Missing'}</span></div>
                </div>
            </div>
        `;
    } catch (e) {
        dom.sysinfoContent.innerHTML = `<p class="sysinfo-loading" style="color:var(--accent-red);">Failed to load: ${e.message}</p>`;
    }
}
