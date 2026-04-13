import json
import plotly.graph_objects as go
import numpy as np

def smooth(values, window=75):
    return np.convolve(values, np.ones(window)/window, mode='valid')

fig = go.Figure()

for trial in range(5):
    try:
        with open(f'trials_p97/metrics_t{trial}.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"trial {trial} not found, skipping")
        continue

    epochs   = [m['epoch'] for m in data['metrics']]
    test_acc = [m['test_acc'] for m in data['metrics']]

    smoothed = smooth(test_acc)
    epochs_trimmed = epochs[:len(smoothed)]

    fig.add_trace(go.Scatter(
        x=epochs_trimmed, y=smoothed,
        name=f"trial {trial} (grokked={data['grokked_at']})",
        mode='lines'
    ))

fig.update_layout(
    title='Test accuracy by trial — p=97, 1-head transformer (smoothed)',
    xaxis_title='Epoch',
    yaxis_title='Test Accuracy',
    yaxis=dict(range=[0, 1.05])
)
fig.show()

fig.write_image("plots/loss.png")