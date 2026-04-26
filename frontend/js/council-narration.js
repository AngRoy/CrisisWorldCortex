import { formatAction, resourceValue } from './actions.js';

const BRAINS = [
    {
        id: 'epidemiology',
        name: 'Epidemiology',
        tone: 'Transmission risk and hospital pressure',
    },
    {
        id: 'logistics',
        name: 'Logistics',
        tone: 'Resource fit and operational scarcity',
    },
    {
        id: 'governance',
        name: 'Governance',
        tone: 'Compliance, legality, and escalation risk',
    },
];

// This module is deliberately a frontend visualization layer. It does not call
// the real Cortex runtime or any LLM subagent; it narrates deterministic,
// observation-derived deliberation so the Space feels alive without coupling UI
// code to the simulator, reward, schemas, or training stack.
export function narrateCouncil(obs, lastReward = 0) {
    if (!obs) {
        return {
            phase: 'Divergence',
            round: 1,
            budget: 0,
            recommendations: [],
            preservedDissent: [],
            decision: { kind: 'no_op' },
            rationale: 'No observation loaded yet.',
            challenge: 'Start a live episode or load the sample trace.',
        };
    }

    const regions = obs.regions || [];
    const pressure = regions.map((region) => ({
        region: region.region,
        score: region.reported_cases_d_ago / 1000 + region.hospital_load * 1.55 + (1 - region.compliance_proxy) * 0.75,
        cases: region.reported_cases_d_ago,
        load: region.hospital_load,
        compliance: region.compliance_proxy,
    })).sort((a, b) => b.score - a.score);

    const hot = pressure[0] || { region: 'R1', score: 0, cases: 0, load: 0, compliance: 1 };
    const resources = obs.resources || {};
    const strictBlocked = (obs.legal_constraints || []).some(
        (constraint) => constraint.blocked_action === 'restrict_movement.strict',
    );
    const budget = obs.cognition_budget_remaining || 0;
    const urgency = clamp((hot.score + (obs.ticks_remaining <= 4 ? 0.35 : 0)) / 2.6, 0, 1);
    const agreement = clamp(1 - Math.abs((resourceValue(resources, 'hospital_beds') / 500) - urgency), 0.1, 0.98);

    const epiAction = chooseEpidemiologyAction(hot, strictBlocked, obs);
    const logAction = chooseLogisticsAction(hot, resources);
    const govAction = chooseGovernanceAction(hot, strictBlocked, obs);
    const raw = [
        brainReport(BRAINS[0], epiAction, hot, urgency, lastReward, obs),
        brainReport(BRAINS[1], logAction, hot, urgency, lastReward, obs),
        brainReport(BRAINS[2], govAction, hot, urgency, lastReward, obs),
    ];

    const decision = chooseDecision(raw, obs, strictBlocked);
    const preservedDissent = raw
        .filter((report) => report.action.kind !== decision.kind || report.action.region !== decision.region)
        .slice(0, 2)
        .map((report) => `${report.name}: ${report.challenge}`);

    return {
        phase: phaseFor(obs, agreement),
        round: urgency > 0.72 && agreement < 0.72 ? 2 : 1,
        budget,
        agreement,
        recommendations: raw,
        preservedDissent,
        decision,
        rationale: decisionRationale(decision, hot, strictBlocked, obs),
        challenge: raw.reduce((best, report) => report.challengeScore > best.challengeScore ? report : best, raw[0]).challenge,
    };
}

function chooseEpidemiologyAction(hot, strictBlocked, obs) {
    if (hot.load > 0.72 || hot.cases > 120) {
        return {
            kind: 'restrict_movement',
            region: hot.region,
            severity: strictBlocked ? 'moderate' : 'strict',
        };
    }
    if ((obs.ticks_remaining || 0) > 6 && hot.cases < 35) {
        return {
            kind: 'request_data',
            region: hot.region,
            data_type: 'case_survey',
        };
    }
    return {
        kind: 'deploy_resource',
        region: hot.region,
        resource_type: 'test_kits',
        quantity: 80,
    };
}

function chooseLogisticsAction(hot, resources) {
    if (resourceValue(resources, 'hospital_beds') < 120 && resourceValue(resources, 'mobile_units') >= 2) {
        return {
            kind: 'deploy_resource',
            region: hot.region,
            resource_type: 'mobile_units',
            quantity: Math.min(3, resourceValue(resources, 'mobile_units')),
        };
    }
    if (resourceValue(resources, 'test_kits') > 120) {
        return {
            kind: 'deploy_resource',
            region: hot.region,
            resource_type: 'test_kits',
            quantity: Math.min(120, resourceValue(resources, 'test_kits')),
        };
    }
    if (resourceValue(resources, 'vaccine_doses') > 80) {
        return {
            kind: 'deploy_resource',
            region: hot.region,
            resource_type: 'vaccine_doses',
            quantity: Math.min(160, resourceValue(resources, 'vaccine_doses')),
        };
    }
    return { kind: 'no_op' };
}

function chooseGovernanceAction(hot, strictBlocked, obs) {
    if (strictBlocked && hot.load > 0.62) {
        return {
            kind: 'escalate',
            to_authority: 'national',
        };
    }
    if (hot.compliance < 0.68) {
        return {
            kind: 'restrict_movement',
            region: hot.region,
            severity: 'light',
        };
    }
    if ((obs.legal_constraints || []).length && (obs.ticks_remaining || 0) <= 5) {
        return {
            kind: 'escalate',
            to_authority: 'regional',
        };
    }
    return {
        kind: 'request_data',
        region: hot.region,
        data_type: 'compliance_check',
    };
}

function brainReport(brain, action, hot, urgency, lastReward, obs) {
    const confidenceBase = brain.id === 'logistics'
        ? 0.58 + urgency * 0.25
        : brain.id === 'governance'
            ? 0.54 + (1 - hot.compliance) * 0.34
            : 0.6 + hot.load * 0.28;
    const confidence = clamp(confidenceBase + Math.max(lastReward, -0.15) * 0.08, 0.18, 0.96);
    const minority = action.kind === 'restrict_movement'
        ? { kind: 'request_data', region: hot.region, data_type: 'hospital_audit' }
        : { kind: 'restrict_movement', region: hot.region, severity: 'light' };
    return {
        id: brain.id,
        name: brain.name,
        tone: brain.tone,
        action,
        confidence,
        summary: summaryFor(brain.id, action, hot, obs),
        challenge: challengeFor(brain.id, action, hot, obs),
        challengeScore: 1 - confidence + (action.kind === 'no_op' ? 0.25 : 0),
        minority,
    };
}

function chooseDecision(reports, obs, strictBlocked) {
    const counts = new Map();
    for (const report of reports) {
        const key = actionKey(report.action);
        counts.set(key, (counts.get(key) || 0) + report.confidence);
    }
    let winner = reports[0].action;
    let winnerScore = -1;
    for (const report of reports) {
        const score = counts.get(actionKey(report.action)) || 0;
        if (score > winnerScore) {
            winner = report.action;
            winnerScore = score;
        }
    }
    if (strictBlocked && winner.kind === 'restrict_movement' && winner.severity === 'strict') {
        return { kind: 'escalate', to_authority: 'national' };
    }
    if (obs.done) return { kind: 'no_op' };
    return winner;
}

function summaryFor(brainId, action, hot, obs) {
    if (brainId === 'epidemiology') {
        return `${hot.region} is carrying the highest observed pressure; ${formatAction(action)} is the fastest epidemiological lever.`;
    }
    if (brainId === 'logistics') {
        return `Available stock should be spent where hospital load is most exposed; ${formatAction(action)} has the cleanest operational path.`;
    }
    const locked = (obs.legal_constraints || []).length ? 'legal constraints remain active' : 'legal constraints are clear';
    return `${locked}; ${formatAction(action)} balances actionability with compliance risk.`;
}

function challengeFor(brainId, action, hot, obs) {
    if (brainId === 'epidemiology') {
        return `Telemetry is delayed, so ${hot.region} may not be the only active chain.`;
    }
    if (brainId === 'logistics') {
        return action.kind === 'deploy_resource'
            ? `Stock spent now cannot cover a late spike if the episode stretches ${obs.ticks_remaining} more ticks.`
            : 'Inaction may waste scarce response windows.';
    }
    return action.kind === 'restrict_movement'
        ? 'Movement limits can backfire if compliance is already weak.'
        : 'Escalation consumes attention and may not reduce spread by itself.';
}

function decisionRationale(action, hot, strictBlocked, obs) {
    if (obs.done) return 'Episode is terminal; replay the trace or reset for another run.';
    const legal = strictBlocked ? ' Strict movement is legally blocked until escalation.' : '';
    return `Converged on ${formatAction(action)} because ${hot.region} has the highest combined case, load, and compliance pressure.${legal}`;
}

function phaseFor(obs, agreement) {
    if (!obs.recent_action_log?.length) return 'Divergence';
    if (agreement < 0.55) return 'Challenge';
    if ((obs.ticks_remaining || 0) <= 3) return 'Convergence';
    return 'Narrowing';
}

function actionKey(action) {
    if (!action) return 'none';
    return [action.kind, action.region || '', action.resource_type || '', action.severity || '', action.to_authority || ''].join(':');
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
