/* ═══════════════════════════════════════════════════════
   RLPredictor — Frontend Logic (app.js)
   ═══════════════════════════════════════════════════════ */

const API = '';  // same origin

// ─── State ────────────────────────────────────────────────
let state = {
    sessionActive: false,
    matchups: [],
    matchData: null,   // loaded match response
    matchIndex: null,
};


// ─── DOM refs ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const sessionBadge     = $('#session-badge');
const btnDownload      = $('#btn-download-db');
const btnEndSession    = $('#btn-end-session');
const loadingOverlay   = $('#loading-overlay');
const loadingText      = $('#loading-text');

const steps = {
    setup:     $('#step-setup'),
    scrape:    $('#step-scrape'),
    matches:   $('#step-matches'),
    dashboard: $('#step-dashboard'),
};


// ─── Helpers ──────────────────────────────────────────────

function showStep(name) {
    Object.values(steps).forEach(s => s.classList.remove('active'));
    steps[name].classList.add('active');
}

function showLoading(msg = 'Loading...') {
    loadingText.textContent = msg;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function updateSessionUI(active) {
    state.sessionActive = active;
    sessionBadge.textContent = active ? 'Session Active' : 'No Session';
    sessionBadge.className = active ? 'badge badge-active' : 'badge badge-inactive';
    btnDownload.style.display = active ? '' : 'none';
    btnEndSession.style.display = active ? '' : 'none';
}

async function apiCall(path, opts = {}) {
    const url = API + path;
    const res = await fetch(url, {
        credentials: 'include',
        ...opts,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `API error ${res.status}`);
    }
    return res.json();
}


// ═══════════════════════════════════════════════════════════
// STEP 1: Session Setup
// ═══════════════════════════════════════════════════════════

$('#form-setup').addEventListener('submit', async (e) => {
    e.preventDefault();
    const apiKey = $('#input-api-key').value.trim();
    if (!apiKey) return;

    const fileInput = $('#input-db-upload');
    const formData = new FormData();
    formData.append('bc_api_key', apiKey);
    if (fileInput.files.length > 0) {
        formData.append('db_file', fileInput.files[0]);
    }

    showLoading('Creating session...');
    try {
        const data = await apiCall('/api/session/new', {
            method: 'POST',
            body: formData,
        });
        updateSessionUI(true);
        hideLoading();
        showStep('scrape');

        // Show summary
        console.log('Session created:', data);
    } catch (err) {
        hideLoading();
        alert('Failed to create session: ' + err.message);
    }
});

// File upload drag & drop
const dropZone = $('#drop-zone');
const fileInput = $('#input-db-upload');
const fileName = $('#file-name');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileName.textContent = e.dataTransfer.files[0].name;
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        fileName.textContent = fileInput.files[0].name;
    }
});


// ═══════════════════════════════════════════════════════════
// STEP 2: Tournament Scraping
// ═══════════════════════════════════════════════════════════

$('#form-scrape').addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = $('#input-url').value.trim();
    if (!url) return;

    const sections = [];
    $$('#form-scrape .checkbox input:checked').forEach(cb => sections.push(cb.value));

    const progress = $('#scrape-progress');
    const submitBtn = e.target.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    progress.style.display = '';

    try {
        const data = await apiCall('/api/scrape', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, sections: sections.length > 0 ? sections : null }),
        });

        state.matchups = data.matchups || [];
        if (state.matchups.length === 0) {
            alert(`No matchups found (${data.total} total scraped, ${data.concrete} concrete).\n\nTry unchecking section filters or check the URL.`);
            return;
        }
        renderMatchGrid(state.matchups);
        showStep('matches');
    } catch (err) {
        alert('Scraping failed: ' + err.message);
    } finally {
        submitBtn.disabled = false;
        progress.style.display = 'none';
    }
});


// ═══════════════════════════════════════════════════════════
// STEP 3: Match Selection
// ═══════════════════════════════════════════════════════════

function renderMatchGrid(matchups) {
    const grid = $('#match-grid');
    grid.innerHTML = '';

    matchups.forEach((m, idx) => {
        const card = document.createElement('div');
        card.className = 'match-card';
        card.dataset.index = idx;

        // Auto-detect BO from section type
        const section = (m.section || '').toLowerCase();
        let bo = m.best_of || 5;
        if (section.includes('playoff')) bo = 7;
        else if (section.includes('group')) bo = 5;
        m._autoBO = bo;

        card.innerHTML = `
            <div>
                <div class="team">${esc(m.team1 || 'TBD')}</div>
                <div class="round-label">${esc(m.section || '')}</div>
            </div>
            <div class="vs">vs</div>
            <div>
                <div class="team" style="text-align:right;">${esc(m.team2 || 'TBD')}</div>
                <div class="round-label" style="text-align:right;">BO${bo}</div>
            </div>
        `;
        card.addEventListener('click', () => loadMatch(idx));
        grid.appendChild(card);
    });
}

// Search filter
$('#match-search').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    $$('.match-card').forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(q) ? '' : 'none';
    });
});


// ═══════════════════════════════════════════════════════════
// STEP 4: Match Loading & Dashboard
// ═══════════════════════════════════════════════════════════

async function loadMatch(idx) {
    state.matchIndex = idx;
    const match = state.matchups[idx];
    showLoading('Connecting...');

    try {
        // Use SSE for streaming progress
        const res = await fetch(API + '/api/match/load/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ match_index: idx }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `API error ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalData = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const payload = line.slice(6);
                    try {
                        const msg = JSON.parse(payload);
                        if (msg.type === 'progress') {
                            loadingText.textContent = msg.message;
                        } else if (msg.type === 'complete') {
                            finalData = msg.data;
                        } else if (msg.type === 'error') {
                            throw new Error(msg.message);
                        }
                    } catch (e) {
                        if (e.message && !e.message.includes('JSON')) throw e;
                    }
                }
            }
        }

        if (!finalData) throw new Error('Stream ended without data.');

        state.matchData = finalData;
        hideLoading();
        renderDashboard(finalData);
        showStep('dashboard');

        // Check if model exists, if not prompt to train
        apiCall('/api/session/info').then(info => {
            if (!info.has_model) {
                if (confirm("You haven't trained a model for this session yet!\n\nWould you like to use the cached replays to train the AI now? (Highly recommended for accurate predictions)")) {
                    $('#tab-train').click();
                    $('#form-train button[type="submit"]').click();
                }
            }
        }).catch(() => {});
    } catch (err) {
        hideLoading();
        alert('Failed to load match: ' + err.message);
    }
}

function renderDashboard(data) {
    // Title
    $('#dash-title').textContent = `${data.team1} vs ${data.team2}`;
    $('#dash-meta').innerHTML = `
        H2H: ${data.h2h_games} games ${data.h2h_confident ? '<span class="conf-high">(High confidence)</span>' : '<span class="conf-low">(Low confidence)</span>'}
        · Generic: ${data.gen_games} games
        · Players: ${data.available_players.length}
    `;

    // Populate player suggestions (datalist for autocomplete)
    const datalist = $('#player-suggestions');
    datalist.innerHTML = '';
    data.available_players.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        datalist.appendChild(opt);
    });

    // Populate prop sheet form
    renderPropSheetForm(data);

    // Stats summary tab
    renderStatsSummary(data);

    // Reset results
    $('#propsheet-results').style.display = 'none';
    $('#single-result').style.display = 'none';
    $('#train-result').style.display = 'none';
}


// ─── Prop Sheet Form ──────────────────────────────────────

function renderPropSheetForm(data) {
    const container = $('#propsheet-teams');
    container.innerHTML = '';

    const match = state.matchups[state.matchIndex];
    const teams = [
        { key: 'team1', name: data.team1, players: data.rosters.team1 },
        { key: 'team2', name: data.team2, players: data.rosters.team2 },
    ];

    // Update best_of input (3 for BO5, 4 for BO7)
    const expectedGames = (match.best_of || 5) === 5 ? 3 : 4;
    $('#input-num-games').value = expectedGames;
    $('#input-single-games').value = expectedGames;

    teams.forEach(team => {
        const div = document.createElement('div');
        div.className = 'propsheet-team';
        div.innerHTML = `<h4>${esc(team.name)}</h4>`;

        (team.players || []).forEach(player => {
            const pdiv = document.createElement('div');
            pdiv.className = 'propsheet-player';
            pdiv.innerHTML = `
                <div class="player-name">${esc(player)}</div>
                <div class="propsheet-stats propsheet-stats-3">
                    <div>
                        <label>Goals</label>
                        <input type="number" step="0.5" min="0" data-team="${team.key}" data-player="${esc(player)}" data-stat="Goals" placeholder="—">
                    </div>
                    <div>
                        <label>Saves</label>
                        <input type="number" step="0.5" min="0" data-team="${team.key}" data-player="${esc(player)}" data-stat="Saves" placeholder="—">
                    </div>
                    <div>
                        <label>Demos</label>
                        <input type="number" step="0.5" min="0" data-team="${team.key}" data-player="${esc(player)}" data-stat="Demos" placeholder="—">
                    </div>
                </div>
            `;
            div.appendChild(pdiv);
        });

        container.appendChild(div);
    });
}


// ─── Prop Sheet Submit ────────────────────────────────────

$('#form-propsheet').addEventListener('submit', async (e) => {
    e.preventDefault();

    const numGames = parseInt($('#input-num-games').value) || null;

    // Gather all filled inputs
    const lines = { team1: { team_name: '', players: {} }, team2: { team_name: '', players: {} } };

    // Set team names
    if (state.matchData) {
        lines.team1.team_name = state.matchData.team1;
        lines.team2.team_name = state.matchData.team2;
    }

    $$('#propsheet-teams input[data-stat]').forEach(inp => {
        const val = parseFloat(inp.value);
        if (isNaN(val)) return;

        const team = inp.dataset.team;
        const player = inp.dataset.player;
        const stat = inp.dataset.stat;

        if (!lines[team].players[player]) lines[team].players[player] = {};
        lines[team].players[player][stat] = val;
    });

    // Check if any lines were filled
    const totalLines = Object.values(lines).reduce(
        (sum, t) => sum + Object.values(t.players).reduce((s, p) => s + Object.keys(p).length, 0), 0
    );
    if (totalLines === 0) {
        alert('Please fill in at least one stat line.');
        return;
    }

    const progress = $('#propsheet-progress');
    progress.style.display = '';

    try {
        const result = await apiCall('/api/predict/sheet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_games: numGames, lines }),
        });

        renderPropSheetResults(result);
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        progress.style.display = 'none';
    }
});


function renderPropSheetResults(result) {
    const container = $('#propsheet-results');
    container.style.display = '';

    // Best picks
    const bestDiv = $('#best-picks');
    bestDiv.innerHTML = '';

    for (const [teamKey, pick] of Object.entries(result.best_picks || {})) {
        const card = document.createElement('div');
        card.className = 'best-pick-card';

        const pickClass = pick.pick === 'OVER' ? 'over' : 'under';
        const confClass = `conf-${(pick.confidence || 'low').toLowerCase()}`;

        let reasoningHtml = '';
        if (pick.reasoning) {
            reasoningHtml = Object.entries(pick.reasoning)
                .map(([k, v]) => `<div>${k}: <strong>${v}</strong></div>`)
                .join('');
        }

        card.innerHTML = `
            <div class="pick-player">${esc(pick.player)}</div>
            <div class="pick-line">${esc(pick.team_name || teamKey)} · ${esc(pick.stat)} ${pick.pick === 'OVER' ? '>' : '<'} ${pick.threshold}</div>
            <div class="pick-verdict ${pickClass}">${pick.pick}</div>
            <div class="pick-prob">${(pick.probability * 100).toFixed(1)}%</div>
            <div class="pick-confidence ${confClass}">${pick.confidence} confidence</div>
            <div class="pick-reasoning">${reasoningHtml}</div>
        `;
        bestDiv.appendChild(card);
    }

    // Full breakdown table
    const tbody = $('#breakdown-table tbody');
    tbody.innerHTML = '';

    (result.full_breakdown || []).forEach(row => {
        const tr = document.createElement('tr');
        const pickClass = row.pick === 'OVER' ? 'pick-over' : (row.pick === 'UNDER' ? 'pick-under' : '');
        const confClass = `conf-${(row.confidence || 'low').toLowerCase()}`;
        tr.innerHTML = `
            <td>${esc(row.player)}</td>
            <td>${esc(row.stat)}</td>
            <td class="text-mono">${row.threshold}</td>
            <td class="${pickClass}">${row.pick}</td>
            <td class="text-mono">${(row.probability * 100).toFixed(1)}%</td>
            <td class="${confClass}">${row.confidence || '—'}</td>
        `;
        tbody.appendChild(tr);
    });

    // Scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


// ═══════════════════════════════════════════════════════════
// Single Prediction
// ═══════════════════════════════════════════════════════════

$('#form-single').addEventListener('submit', async (e) => {
    e.preventDefault();

    const payload = {
        player: $('#input-player').value,
        stat: $('#input-stat').value,
        threshold: parseFloat($('#input-threshold').value),
        over: $('#input-direction').value === 'over',
        num_games: parseInt($('#input-single-games').value) || null,
    };

    showLoading('Running prediction...');
    try {
        const result = await apiCall('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        hideLoading();
        renderSingleResult(result);
    } catch (err) {
        hideLoading();
        alert('Prediction failed: ' + err.message);
    }
});

function renderSingleResult(result) {
    const container = $('#single-result');
    container.style.display = '';

    const pickClass = result.pick === 'OVER' ? 'over' : 'under';
    const confClass = `conf-${(result.confidence || 'low').toLowerCase()}`;
    const direction = result.pick === 'OVER' ? '>' : '<';

    let reasoningHtml = '';
    if (result.reasoning) {
        reasoningHtml = Object.entries(result.reasoning)
            .map(([k, v]) => `<div>${k}: <strong>${v}</strong></div>`)
            .join('');
    }

    container.innerHTML = `
        <div class="prediction-card">
            <div class="pred-player">${esc(result.player)}</div>
            <div class="pred-line">${esc(result.stat)} ${direction} ${result.threshold}</div>
            <div class="pred-verdict ${pickClass}">${result.pick}</div>
            <div>
                ${result.probability != null ? `<span class="text-mono">${(result.probability * 100).toFixed(1)}%</span>` : ''}
                ${result.confidence ? `<span class="${confClass}"> · ${result.confidence}</span>` : ''}
            </div>
            ${result.model_used ? '<div class="mt-1 text-muted" style="font-size:0.78rem;">🧠 Neural net prediction</div>' : '<div class="mt-1 text-muted" style="font-size:0.78rem;">📊 Heuristic (no trained model)</div>'}
            <div class="pred-reasoning mt-2">${reasoningHtml}</div>
        </div>
    `;
}


// ═══════════════════════════════════════════════════════════
// Training
// ═══════════════════════════════════════════════════════════

$('#form-train').addEventListener('submit', async (e) => {
    e.preventDefault();

    const epochs = parseInt($('#input-epochs').value) || 200;
    const lr = parseFloat($('#input-lr').value) || 0.001;

    const progress = $('#train-progress');
    progress.style.display = '';
    const submitBtn = e.target.querySelector('button[type="submit"]');
    submitBtn.disabled = true;

    try {
        const result = await apiCall('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs, lr }),
        });
        renderTrainResult(result);
    } catch (err) {
        alert('Training failed: ' + err.message);
    } finally {
        progress.style.display = 'none';
        submitBtn.disabled = false;
    }
});

function renderTrainResult(result) {
    const container = $('#train-result');
    container.style.display = '';

    if (result.status === 'error') {
        container.innerHTML = `
            <div class="prediction-card">
                <div class="pred-player" style="color:var(--red);">Training Failed</div>
                <div class="pred-line">${esc(result.message)}</div>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="train-result-card">
            <h4>✅ Training Complete</h4>
            <div class="stat-row"><span class="stat-label">Samples</span><span class="stat-value">${result.samples?.toLocaleString() || '?'}</span></div>
            <div class="stat-row"><span class="stat-label">Epochs</span><span class="stat-value">${result.epochs_run || '?'}</span></div>
            <div class="stat-row"><span class="stat-label">Best Val Accuracy</span><span class="stat-value">${result.best_val_acc || '?'}</span></div>
            <div class="stat-row mt-1"><span class="stat-label" style="color:var(--text-2);">${esc(result.message)}</span></div>
        </div>
    `;
}


// ═══════════════════════════════════════════════════════════
// Stats Summary Tab
// ═══════════════════════════════════════════════════════════

function renderStatsSummary(data) {
    const container = $('#stats-summary');
    container.innerHTML = `
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-number">${data.h2h_games}</div>
                <div class="stat-label">H2H Games</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">${data.gen_games}</div>
                <div class="stat-label">Generic Games</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">${data.available_players.length}</div>
                <div class="stat-label">Players Found</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">${data.h2h_confident ? '✓' : '✗'}</div>
                <div class="stat-label">H2H Confidence</div>
            </div>
        </div>
        <div class="mt-2">
            <h4 style="margin-bottom:0.5rem;">Momentum Data</h4>
            <table class="data-table">
                <thead><tr><th>Player ID</th><th>Games</th><th>Avg Score</th><th>Win Rate</th></tr></thead>
                <tbody>
                    ${Object.entries(data.momentum || {}).map(([pid, m]) => `
                        <tr>
                            <td class="text-mono" style="font-size:0.75rem;">${esc(pid.split(':').pop()?.slice(0, 16) || pid)}</td>
                            <td>${m.games}</td>
                            <td>${m.avg_score}</td>
                            <td>${m.win_rate}%</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        <div class="mt-2">
            <h4 style="margin-bottom:0.5rem;">Sentiment</h4>
            <table class="data-table">
                <thead><tr><th>Player</th><th>Score</th><th>Status</th></tr></thead>
                <tbody>
                    ${Object.entries(data.sentiment || {}).map(([name, s]) => `
                        <tr>
                            <td>${esc(name)}</td>
                            <td class="text-mono">${s.score?.toFixed(2) || '0.00'}</td>
                            <td>${esc(s.status || 'N/A')}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}


// ═══════════════════════════════════════════════════════════
// Verify Picks
// ═══════════════════════════════════════════════════════════

$('#btn-run-verify').addEventListener('click', async () => {
    const progress = $('#verify-progress');
    const resultDiv = $('#verify-result');
    const btn = $('#btn-run-verify');
    
    btn.disabled = true;
    progress.style.display = '';
    resultDiv.style.display = 'none';

    try {
        const result = await apiCall('/api/verify', { method: 'POST' });
        
        // Render verification table
        if (!result.logs || result.logs.length === 0) {
            resultDiv.innerHTML = '<div class="text-muted">No verified predictions found or no recent matches completed.</div>';
        } else {
            let html = '<table class="data-table"><thead><tr><th>Date</th><th>Player</th><th>Stat</th><th>Prediction</th><th>Actual</th><th>Result</th></tr></thead><tbody>';
            result.logs.forEach(log => {
                const resClass = log.result === 'WIN' ? 'pick-over' : (log.result === 'LOSS' ? 'pick-under' : '');
                html += `<tr>
                    <td>${esc(log.date)}</td>
                    <td>${esc(log.player)}</td>
                    <td>${esc(log.stat)}</td>
                    <td>${esc(log.prediction)}</td>
                    <td>${esc(log.actual)}</td>
                    <td class="${resClass}">${esc(log.result)}</td>
                </tr>`;
            });
            html += '</tbody></table>';
            
            if (result.win_rate != null) {
                html = `<div style="margin-bottom:1rem;"><strong>Overall Win Rate:</strong> ${result.win_rate}% (${result.wins}/${result.total})</div>` + html;
            }
            
            resultDiv.innerHTML = html;
        }
        resultDiv.style.display = '';
    } catch (err) {
        alert('Verification failed: ' + err.message);
    } finally {
        progress.style.display = 'none';
        btn.disabled = false;
    }
});


// ═══════════════════════════════════════════════════════════
// Tabs
// ═══════════════════════════════════════════════════════════

$$('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        $$('.tab').forEach(t => t.classList.remove('active'));
        $$('.tab-content').forEach(tc => tc.classList.remove('active'));
        tab.classList.add('active');
        $(`#${tab.dataset.tab}`).classList.add('active');
    });
});


// ═══════════════════════════════════════════════════════════
// Header Actions
// ═══════════════════════════════════════════════════════════

btnDownload.addEventListener('click', () => {
    window.location.href = '/api/session/download';
});

btnEndSession.addEventListener('click', async () => {
    if (!confirm('End session? Make sure you\'ve downloaded your DB first!')) return;

    try {
        await apiCall('/api/session', { method: 'DELETE' });
    } catch (e) {
        // Ignore — session might already be gone
    }
    updateSessionUI(false);
    state = { sessionActive: false, matchups: [], matchData: null, matchIndex: null };
    showStep('setup');
});

$('#btn-back-matches').addEventListener('click', () => {
    showStep('matches');
});


// ═══════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════

function esc(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}


// ─── Init ─────────────────────────────────────────────────
showStep('setup');
