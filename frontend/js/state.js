const subscribers = new Set();

export const AppState = {
    connection: 'idle',
    mode: 'sample',
    taskName: 'outbreak_easy',
    seed: 0,
    maxTicks: 12,
    observation: null,
    reward: null,
    done: false,
    totalReward: 0,
    liveTrace: [],
    replayTrace: [],
    replayIndex: 0,
    replayPlaying: false,
    replaySpeedMs: 1200,
    autoplayLive: false,
    selectedActionKind: 'deploy_resource',
    selectedRegion: 'R1',
    selectedResource: 'test_kits',
    selectedToResource: 'hospital_beds',
    quantity: 50,
    dataType: 'case_survey',
    severity: 'moderate',
    authority: 'regional',
    council: null,
    statusMessage: 'Sample trace loaded',
};

export function subscribe(callback) {
    subscribers.add(callback);
    return () => subscribers.delete(callback);
}

export function setState(patch) {
    Object.assign(AppState, patch);
    subscribers.forEach((callback) => callback(AppState));
}

export function addTraceFrame(frame) {
    const liveTrace = [...AppState.liveTrace, frame];
    setState({
        liveTrace,
        replayTrace: liveTrace,
        replayIndex: liveTrace.length - 1,
    });
}

export function currentFrame() {
    return AppState.replayTrace[AppState.replayIndex] || null;
}

export function showToast(message, type = 'info') {
    const root = document.getElementById('toast-root');
    if (!root) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    root.appendChild(toast);
    window.setTimeout(() => {
        toast.classList.add('toast-out');
        window.setTimeout(() => toast.remove(), 220);
    }, 3200);
}
