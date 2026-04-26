import { AppState, currentFrame, setState } from './state.js';

let replayTimer = null;

export function setReplayTrace(trace) {
    const replayTrace = Array.isArray(trace) ? trace : [];
    setState({
        replayTrace,
        replayIndex: 0,
        replayPlaying: false,
        observation: replayTrace[0]?.observation || null,
        reward: replayTrace[0]?.reward ?? null,
        done: Boolean(replayTrace[0]?.done),
    });
}

export function goToFrame(index) {
    const bounded = Math.max(0, Math.min(index, AppState.replayTrace.length - 1));
    const frame = AppState.replayTrace[bounded];
    if (!frame) return;
    setState({
        replayIndex: bounded,
        observation: frame.observation,
        reward: frame.reward ?? null,
        done: Boolean(frame.done),
        statusMessage: frame.label || `Replay frame ${bounded + 1}`,
    });
}

export function toggleReplayPlayback(renderTick) {
    if (AppState.replayPlaying) {
        stopReplay();
        return;
    }
    if (AppState.replayTrace.length < 2) return;
    setState({ replayPlaying: true });
    replayTimer = window.setInterval(() => {
        const next = AppState.replayIndex + 1;
        if (next >= AppState.replayTrace.length) {
            stopReplay();
            return;
        }
        goToFrame(next);
        if (typeof renderTick === 'function') renderTick(currentFrame());
    }, AppState.replaySpeedMs);
}

export function stopReplay() {
    if (replayTimer) {
        window.clearInterval(replayTimer);
        replayTimer = null;
    }
    setState({ replayPlaying: false });
}
