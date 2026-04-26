export const RESOURCE_TYPES = [
    'test_kits',
    'hospital_beds',
    'mobile_units',
    'vaccine_doses',
];

export const DATA_TYPES = [
    'case_survey',
    'hospital_audit',
    'compliance_check',
];

export const SEVERITIES = [
    'none',
    'light',
    'moderate',
    'strict',
];

export const AUTHORITIES = [
    'regional',
    'national',
];

export const ACTION_KINDS = [
    { id: 'deploy_resource', label: 'Deploy' },
    { id: 'request_data', label: 'Request Data' },
    { id: 'restrict_movement', label: 'Restrict' },
    { id: 'escalate', label: 'Escalate' },
    { id: 'reallocate_budget', label: 'Reallocate' },
    { id: 'no_op', label: 'No-op' },
];

export function normalizeRegionSelection(state) {
    const regions = state.observation?.regions || [];
    if (!regions.length) return {};
    if (regions.some((region) => region.region === state.selectedRegion)) return {};
    return { selectedRegion: regions[0].region };
}

export function buildAction(state) {
    const region = state.selectedRegion || state.observation?.regions?.[0]?.region || 'R1';
    const quantity = Math.max(0, Number.parseInt(state.quantity, 10) || 0);
    switch (state.selectedActionKind) {
        case 'deploy_resource':
            return {
                kind: 'deploy_resource',
                region,
                resource_type: state.selectedResource,
                quantity,
            };
        case 'request_data':
            return {
                kind: 'request_data',
                region,
                data_type: state.dataType,
            };
        case 'restrict_movement':
            return {
                kind: 'restrict_movement',
                region,
                severity: state.severity,
            };
        case 'escalate':
            return {
                kind: 'escalate',
                to_authority: state.authority,
            };
        case 'reallocate_budget':
            return {
                kind: 'reallocate_budget',
                from_resource: state.selectedResource,
                to_resource: state.selectedToResource,
                amount: quantity,
            };
        case 'no_op':
        default:
            return { kind: 'no_op' };
    }
}

export function formatAction(action) {
    if (!action) return 'none';
    switch (action.kind) {
        case 'deploy_resource':
            return `deploy ${action.quantity} ${labelize(action.resource_type)} to ${action.region}`;
        case 'request_data':
            return `request ${labelize(action.data_type)} for ${action.region}`;
        case 'restrict_movement':
            return `${action.severity} movement limits in ${action.region}`;
        case 'escalate':
            return `escalate to ${action.to_authority}`;
        case 'reallocate_budget':
            return `move ${action.amount} from ${labelize(action.from_resource)} to ${labelize(action.to_resource)}`;
        case 'public_communication':
            return `public communication to ${action.audience}`;
        case 'no_op':
        default:
            return 'no-op';
    }
}

export function actionFromLogEntry(entry) {
    return entry?.action || null;
}

export function labelize(value) {
    return String(value || '')
        .replaceAll('_', ' ')
        .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function resourceValue(resources, resourceType) {
    if (!resources) return 0;
    if (resourceType === 'hospital_beds') return resources.hospital_beds_free || 0;
    return resources[resourceType] || 0;
}
