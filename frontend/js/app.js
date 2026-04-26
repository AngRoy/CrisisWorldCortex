import { resetEnvironment, stepEnvironment } from './api.js';
import {
    ACTION_KINDS,
    AUTHORITIES,
    DATA_TYPES,
    RESOURCE_TYPES,
    SEVERITIES,
    buildAction,
    formatAction,
    labelize,
    normalizeRegionSelection,
    resourceValue,
} from './actions.js';
import { narrateCouncil } from './council-narration.js';
import { goToFrame, setReplayTrace, stopReplay, toggleReplayPlayback } from './replay.js';
import { SAMPLE_TRACE } from './samples.js';
import { AppState, addTraceFrame, setState, showToast, subscribe } from './state.js';

const app = document.getElementById('app');
let liveAutoplayTimer = null;

function init() {
    const first = SAMPLE_TRACE[0];
    setState({
        replayTrace: SAMPLE_TRACE,
        replayIndex: 0,
        observation: first.observation,
        reward: first.reward,
        done: first.done,
        council: narrateCouncil(first.observation, first.reward),
    });
    subscribe(render);
    render();
}

function render() {
    const regionPatch = normalizeRegionSelection(AppState);
    if (Object.keys(regionPatch).length) {
        setState(regionPatch);
        return;
    }
    const council = narrateCouncil(AppState.observation, AppState.reward || 0);
    AppState.council = council;
    app.innerHTML = `
        <main class="shell">
            ${renderTopbar()}
            ${renderCommandbar()}
            <section class="main-grid">
                <div class="left-col">
                    ${renderPhasePanel(council)}
                    ${renderWorldPanel()}
                    ${renderResourcesPanel()}
                </div>
                <div class="center-col">
                    ${renderActionPanel(council)}
                    ${renderTimelinePanel()}
                </div>
                <div class="right-col">
                    ${renderCouncilPanel(council)}
                    ${renderReplayPanel()}
                </div>
            </section>
        </main>
    `;
    bindEvents();
}

function renderTopbar() {
    return `
        <header class="topbar">
            <div class="brand">
                <div class="mark">CW</div>
                <div>
                    <h1>CrisisWorld Cortex</h1>
                    <p>Outbreak response dashboard for live OpenEnv episodes and replay traces.</p>
                </div>
            </div>
            <div class="top-actions">
                <span class="badge ${AppState.mode === 'live' ? 'live' : 'sample'}">${AppState.mode === 'live' ? 'Live env' : 'Sample replay'}</span>
                ${AppState.done ? '<span class="badge done">Terminal</span>' : ''}
                <span class="badge">${AppState.connection}</span>
                <button class="btn ghost" data-action="load-sample">Load Sample</button>
                <button class="btn ghost" data-action="open-web">Open /web</button>
            </div>
        </header>
    `;
}

function renderCommandbar() {
    return `
        <section class="commandbar">
            <div class="fieldrow">
                <label class="field wide">
                    <span>Task</span>
                    <select data-bind="taskName">
                        ${option('outbreak_easy', 'Outbreak Easy', AppState.taskName)}
                        ${option('outbreak_medium', 'Outbreak Medium', AppState.taskName)}
                        ${option('outbreak_hard', 'Outbreak Hard', AppState.taskName)}
                    </select>
                </label>
                <label class="field compact">
                    <span>Seed</span>
                    <input type="number" min="0" step="1" data-bind="seed" value="${AppState.seed}">
                </label>
                <label class="field compact">
                    <span>Max ticks</span>
                    <input type="number" min="1" max="30" step="1" data-bind="maxTicks" value="${AppState.maxTicks}">
                </label>
                <button class="btn primary" data-action="reset-live">Reset Live Episode</button>
            </div>
            <div class="fieldrow">
                <label class="field wide">
                    <span>Replay speed</span>
                    <input type="range" min="350" max="2600" step="50" data-bind="replaySpeedMs" value="${AppState.replaySpeedMs}">
                </label>
                <span class="badge">${AppState.replaySpeedMs} ms</span>
            </div>
            <div class="fieldrow">
                <button class="btn ${AppState.autoplayLive ? 'warn' : 'blue'}" data-action="toggle-live-autoplay" ${AppState.mode !== 'live' || AppState.done ? 'disabled' : ''}>
                    ${AppState.autoplayLive ? 'Stop Autoplay' : 'Autoplay Decisions'}
                </button>
            </div>
        </section>
    `;
}

function renderPhasePanel(council) {
    const obs = AppState.observation;
    return `
        <section class="panel">
            <div class="panel-header">
                <h2>Episode State</h2>
                <span class="badge">${AppState.statusMessage}</span>
            </div>
            <div class="panel-body">
                <div class="metrics">
                    ${metric('Tick', obs?.tick ?? '-', `${obs?.ticks_remaining ?? '-'} left`)}
                    ${metric('Phase', council.phase, `round ${council.round}`)}
                    ${metric('Budget', council.budget || '-', 'cognition')}
                    ${metric('Reward', formatReward(AppState.reward), `total ${formatReward(AppState.totalReward)}`)}
                </div>
            </div>
        </section>
    `;
}

function renderWorldPanel() {
    const obs = AppState.observation;
    if (!obs) return panel('World State', '<div class="empty">No observation loaded.</div>');
    const regions = obs.regions || [];
    const maxScore = Math.max(...regions.map((region) => pressure(region)), 1);
    const html = `
        <div class="world-map">
            ${regions.map((region) => renderRegion(region, pressure(region) / maxScore)).join('')}
        </div>
    `;
    return panel('World State', html);
}

function renderRegion(region, heat) {
    const hot = heat > 0.78 ? 'hot' : '';
    return `
        <article class="region-node ${hot}">
            <div class="region-top">
                <span class="region-name">${region.region}</span>
                <span class="badge">${Math.round(heat * 100)} pressure</span>
            </div>
            <div class="region-stats">
                ${statline('Cases', clamp(region.reported_cases_d_ago / 180, 0, 1), region.reported_cases_d_ago)}
                ${statline('Load', region.hospital_load, pct(region.hospital_load), 'load')}
                ${statline('Comply', region.compliance_proxy, pct(region.compliance_proxy), 'comp')}
            </div>
        </article>
    `;
}

function renderResourcesPanel() {
    const obs = AppState.observation;
    const resources = obs?.resources || {};
    const constraints = obs?.legal_constraints || [];
    const restrictions = obs?.active_restrictions || [];
    return panel('Resources and Constraints', `
        <div class="resource-grid">
            ${RESOURCE_TYPES.map((type) => `
                <div class="resource">
                    <span class="micro-label">${labelize(type)}</span>
                    <strong>${resourceValue(resources, type)}</strong>
                </div>
            `).join('')}
        </div>
        <div style="height: 12px"></div>
        <div class="section-title">Active restrictions</div>
        <div class="timeline" style="max-height: 130px; margin-top: 8px;">
            ${restrictions.length ? restrictions.map((item) => `
                <div class="timeline-entry">
                    <strong>${item.region}</strong>
                    <span>${labelize(item.severity)} movement limits</span>
                    <span class="badge">${item.ticks_remaining} ticks</span>
                </div>
            `).join('') : '<div class="tiny">None active.</div>'}
        </div>
        <div style="height: 12px"></div>
        <div class="section-title">Legal constraints</div>
        <div class="timeline" style="max-height: 130px; margin-top: 8px;">
            ${constraints.length ? constraints.map((item) => `
                <div class="timeline-entry rejected">
                    <strong>${item.rule_id}</strong>
                    <span>Blocks ${item.blocked_action}</span>
                    <span class="badge">${item.unlock_via}</span>
                </div>
            `).join('') : '<div class="tiny">No active legal blockers.</div>'}
        </div>
    `);
}

function renderActionPanel(council) {
    const selected = AppState.selectedActionKind;
    return panel('Task Controls', `
        <div class="action-panel">
            <div class="segmented" role="tablist" aria-label="Action type">
                ${ACTION_KINDS.map((kind) => `
                    <button data-kind="${kind.id}" class="${selected === kind.id ? 'active' : ''}">${kind.label}</button>
                `).join('')}
            </div>
            <div class="action-form">
                ${renderActionInputs(selected)}
                <div class="fieldrow">
                    <button class="btn primary" data-action="submit-action" ${AppState.mode !== 'live' || AppState.done ? 'disabled' : ''}>Submit Action</button>
                    <button class="btn blue" data-action="apply-recommendation" ${AppState.mode !== 'live' || AppState.done ? 'disabled' : ''}>Apply Recommendation</button>
                    <span class="tiny">Recommendation: ${formatAction(council.decision)}</span>
                </div>
            </div>
        </div>
    `);
}

function renderActionInputs(kind) {
    const regions = AppState.observation?.regions || [{ region: AppState.selectedRegion || 'R1' }];
    const regionSelect = `
        <label class="field">
            <span>Region</span>
            <select data-bind="selectedRegion">
                ${regions.map((region) => option(region.region, region.region, AppState.selectedRegion)).join('')}
            </select>
        </label>
    `;
    const resourceSelect = (bindName, label) => `
        <label class="field">
            <span>${label}</span>
            <select data-bind="${bindName}">
                ${RESOURCE_TYPES.map((type) => option(type, labelize(type), AppState[bindName])).join('')}
            </select>
        </label>
    `;
    const quantity = `
        <label class="field">
            <span>${kind === 'reallocate_budget' ? 'Amount' : 'Quantity'}</span>
            <input type="number" min="0" step="1" data-bind="quantity" value="${AppState.quantity}">
        </label>
    `;
    if (kind === 'deploy_resource') {
        return `<div class="param-grid">${regionSelect}${resourceSelect('selectedResource', 'Resource')}${quantity}</div>`;
    }
    if (kind === 'request_data') {
        return `
            <div class="param-grid">
                ${regionSelect}
                <label class="field">
                    <span>Data type</span>
                    <select data-bind="dataType">${DATA_TYPES.map((type) => option(type, labelize(type), AppState.dataType)).join('')}</select>
                </label>
            </div>
        `;
    }
    if (kind === 'restrict_movement') {
        return `
            <div class="param-grid">
                ${regionSelect}
                <label class="field">
                    <span>Severity</span>
                    <select data-bind="severity">${SEVERITIES.map((type) => option(type, labelize(type), AppState.severity)).join('')}</select>
                </label>
            </div>
        `;
    }
    if (kind === 'escalate') {
        return `
            <div class="param-grid">
                <label class="field">
                    <span>Authority</span>
                    <select data-bind="authority">${AUTHORITIES.map((type) => option(type, labelize(type), AppState.authority)).join('')}</select>
                </label>
            </div>
        `;
    }
    if (kind === 'reallocate_budget') {
        return `<div class="param-grid">${resourceSelect('selectedResource', 'From')}${resourceSelect('selectedToResource', 'To')}${quantity}</div>`;
    }
    return '<div class="tiny">No parameters required.</div>';
}

function renderCouncilPanel(council) {
    return panel('Narrated Council', `
        <div class="tiny" style="margin-bottom: 10px;">
            Frontend visualization derived from observations. It is not a live Cortex runtime or LLM council.
        </div>
        <div class="decision">
            <h3>Converged action</h3>
            <strong>${formatAction(council.decision)}</strong>
            <p class="tiny">${council.rationale}</p>
        </div>
        <div style="height: 10px"></div>
        <div class="council-stack">
            ${council.recommendations.map((report) => renderBrain(report)).join('')}
        </div>
        <div style="height: 10px"></div>
        <div class="section-title">Preserved dissent</div>
        <div class="timeline" style="max-height: 150px; margin-top: 8px;">
            ${council.preservedDissent.length ? council.preservedDissent.map((item, index) => `
                <div class="timeline-entry">
                    <strong>D${index + 1}</strong>
                    <span>${item}</span>
                    <span></span>
                </div>
            `).join('') : '<div class="tiny">No dissent preserved on this frame.</div>'}
        </div>
    `);
}

function renderBrain(report) {
    return `
        <article class="brain-card">
            <div class="brain-head">
                <h3>${report.name}</h3>
                <span class="badge">${Math.round(report.confidence * 100)} conf</span>
            </div>
            <div class="bar"><span style="width: ${Math.round(report.confidence * 100)}%"></span></div>
            <p><strong>${formatAction(report.action)}</strong></p>
            <p>${report.summary}</p>
            <p><span class="micro-label">Challenge</span><br>${report.challenge}</p>
            <p><span class="micro-label">Minority</span><br>${formatAction(report.minority)}</p>
        </article>
    `;
}

function renderTimelinePanel() {
    const log = AppState.observation?.recent_action_log || [];
    const liveTrace = AppState.liveTrace || [];
    return panel('Final Action Timeline', `
        <div class="timeline">
            ${log.length ? log.map((entry) => `
                <div class="timeline-entry ${entry.accepted ? '' : 'rejected'}">
                    <strong>T${entry.tick}</strong>
                    <span>${formatAction(entry.action)}</span>
                    <span class="badge">${entry.accepted ? 'accepted' : 'rejected'}</span>
                </div>
            `).join('') : '<div class="empty">No actions have been submitted yet.</div>'}
        </div>
        <div style="height: 12px"></div>
        <div class="tiny">Live trace frames captured this session: ${liveTrace.length}</div>
    `);
}

function renderReplayPanel() {
    const max = Math.max(0, AppState.replayTrace.length - 1);
    return panel('Replay', `
        <div class="replay-row">
            <button class="btn" data-action="toggle-replay">${AppState.replayPlaying ? 'Pause' : 'Play'}</button>
            <input type="range" min="0" max="${max}" step="1" value="${AppState.replayIndex}" data-action="scrub-replay" ${max === 0 ? 'disabled' : ''}>
            <span class="badge">${AppState.replayIndex + 1}/${AppState.replayTrace.length || 1}</span>
        </div>
        <div style="height: 10px"></div>
        <div class="fieldrow">
            <button class="btn ghost" data-action="use-live-trace" ${AppState.liveTrace.length ? '' : 'disabled'}>Use Live Trace</button>
            <button class="btn ghost" data-action="load-sample">Use Sample Trace</button>
        </div>
        <p class="tiny">Replay changes the displayed frame only. It does not step the environment.</p>
    `);
}

function panel(title, body) {
    return `
        <section class="panel">
            <div class="panel-header">
                <h2>${title}</h2>
            </div>
            <div class="panel-body">${body}</div>
        </section>
    `;
}

function metric(label, value, hint) {
    return `
        <div class="metric">
            <span class="micro-label">${label}</span>
            <strong>${value}</strong>
            <span class="tiny">${hint}</span>
        </div>
    `;
}

function statline(label, value, display, extraClass = '') {
    return `
        <div class="statline">
            <span>${label}</span>
            <div class="bar ${extraClass}"><span style="width: ${Math.round(clamp(value, 0, 1) * 100)}%"></span></div>
            <strong>${display}</strong>
        </div>
    `;
}

function bindEvents() {
    document.querySelectorAll('[data-bind]').forEach((element) => {
        element.addEventListener('change', () => {
            const key = element.dataset.bind;
            const value = element.type === 'number' || element.type === 'range'
                ? Number.parseInt(element.value, 10)
                : element.value;
            setState({ [key]: value });
        });
        if (element.type === 'range') {
            element.addEventListener('input', () => {
                const key = element.dataset.bind;
                setState({ [key]: Number.parseInt(element.value, 10) });
            });
        }
    });

    document.querySelectorAll('[data-kind]').forEach((button) => {
        button.addEventListener('click', () => setState({ selectedActionKind: button.dataset.kind }));
    });

    document.querySelectorAll('[data-action]').forEach((element) => {
        element.addEventListener('click', () => handleAction(element.dataset.action, element));
        if (element.dataset.action === 'scrub-replay') {
            element.addEventListener('input', () => {
                stopReplay();
                goToFrame(Number.parseInt(element.value, 10));
            });
        }
    });
}

async function handleAction(actionName, element) {
    if (actionName === 'open-web') {
        window.location.href = '/web/';
        return;
    }
    if (actionName === 'load-sample') {
        stopLiveAutoplay();
        setReplayTrace(SAMPLE_TRACE);
        setState({
            mode: 'sample',
            statusMessage: 'Sample trace loaded',
            council: narrateCouncil(SAMPLE_TRACE[0].observation, SAMPLE_TRACE[0].reward),
        });
        showToast('Sample trace loaded', 'success');
        return;
    }
    if (actionName === 'use-live-trace') {
        if (!AppState.liveTrace.length) return;
        setReplayTrace(AppState.liveTrace);
        setState({ mode: 'replay', statusMessage: 'Live trace replay' });
        return;
    }
    if (actionName === 'toggle-replay') {
        toggleReplayPlayback();
        return;
    }
    if (actionName === 'scrub-replay') {
        stopReplay();
        goToFrame(Number.parseInt(element.value, 10));
        return;
    }
    if (actionName === 'reset-live') {
        await resetLive();
        return;
    }
    if (actionName === 'submit-action') {
        await submitAction(buildAction(AppState));
        return;
    }
    if (actionName === 'apply-recommendation') {
        await submitAction(AppState.council?.decision || { kind: 'no_op' });
        return;
    }
    if (actionName === 'toggle-live-autoplay') {
        toggleLiveAutoplay();
    }
}

async function resetLive() {
    stopReplay();
    stopLiveAutoplay();
    setState({ connection: 'resetting', statusMessage: 'Resetting live episode' });
    try {
        const data = await resetEnvironment({
            taskName: AppState.taskName,
            seed: AppState.seed,
            maxTicks: AppState.maxTicks,
        });
        const frame = {
            label: `Live reset: ${AppState.taskName}`,
            action: null,
            observation: data.observation,
            reward: data.reward ?? 0,
            done: Boolean(data.done),
        };
        setState({
            mode: 'live',
            connection: 'connected',
            observation: data.observation,
            reward: data.reward ?? 0,
            done: Boolean(data.done),
            totalReward: 0,
            liveTrace: [frame],
            replayTrace: [frame],
            replayIndex: 0,
            statusMessage: 'Live episode ready',
        });
        showToast('Live episode reset', 'success');
    } catch (error) {
        setState({ connection: 'error', statusMessage: 'Reset failed' });
        showToast(`Reset failed: ${error.message}`, 'error');
    }
}

async function submitAction(payload) {
    if (AppState.mode !== 'live') {
        showToast('Reset a live episode before submitting actions.', 'error');
        return;
    }
    if (AppState.done) {
        showToast('Episode is terminal. Reset to continue.', 'error');
        return;
    }
    setState({ connection: 'stepping', statusMessage: `Submitting ${payload.kind}` });
    try {
        const data = await stepEnvironment(payload);
        const reward = Number(data.reward ?? 0);
        const frame = {
            label: `Submitted ${formatAction(payload)}`,
            action: payload,
            observation: data.observation,
            reward,
            done: Boolean(data.done),
        };
        addTraceFrame(frame);
        setState({
            mode: 'live',
            connection: 'connected',
            observation: data.observation,
            reward,
            done: Boolean(data.done),
            totalReward: AppState.totalReward + reward,
            statusMessage: data.done ? 'Episode complete' : 'Action accepted',
        });
        if (data.done) {
            stopLiveAutoplay();
            showToast('Episode complete', 'success');
        }
    } catch (error) {
        stopLiveAutoplay();
        setState({ connection: 'error', statusMessage: 'Step failed' });
        showToast(`Step failed: ${error.message}`, 'error');
    }
}

function toggleLiveAutoplay() {
    if (AppState.autoplayLive) {
        stopLiveAutoplay();
        return;
    }
    if (AppState.mode !== 'live' || AppState.done) return;
    setState({ autoplayLive: true });
    liveAutoplayTimer = window.setInterval(async () => {
        if (AppState.done || AppState.mode !== 'live') {
            stopLiveAutoplay();
            return;
        }
        await submitAction(AppState.council?.decision || { kind: 'no_op' });
    }, AppState.replaySpeedMs);
}

function stopLiveAutoplay() {
    if (liveAutoplayTimer) {
        window.clearInterval(liveAutoplayTimer);
        liveAutoplayTimer = null;
    }
    if (AppState.autoplayLive) setState({ autoplayLive: false });
}

function option(value, label, selected) {
    return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`;
}

function pressure(region) {
    return region.reported_cases_d_ago / 1000 + region.hospital_load * 1.55 + (1 - region.compliance_proxy) * 0.75;
}

function pct(value) {
    return `${Math.round((value || 0) * 100)}%`;
}

function formatReward(value) {
    return typeof value === 'number' ? value.toFixed(2) : '-';
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

init();
