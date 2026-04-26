async function postJson(url, payload) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        const detail = data?.detail || data?.message || `${response.status} ${response.statusText}`;
        throw new Error(detail);
    }
    return data;
}

export async function resetEnvironment({ taskName, seed, maxTicks }) {
    return postJson('/web/reset', {
        task_name: taskName,
        seed,
        max_ticks: maxTicks,
    });
}

export async function stepEnvironment(payload) {
    return postJson('/web/step', {
        action: {
            action: payload,
            metadata: {},
        },
    });
}

export async function getEnvironmentState() {
    const response = await fetch('/web/state');
    if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
}
