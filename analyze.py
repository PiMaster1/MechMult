import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from train import ModMultTransformer, p

def legendre(a, p):
    v = pow(int(a), (p-1)//2, p)
    return 1 if v == 1 else 0

qr_tokens = np.array([legendre(a, p) for a in range(1, p)])

all_projections = []
all_labels      = []
all_trials      = []
lda_accs        = []
lda_1d_accs     = []

for trial in range(5):
    try:
        model = ModMultTransformer(p=p, d_model=128, n_heads=1, num_layers=1)
        model.load_state_dict(torch.load(f'trials_p97/checkpoint_t{trial}.pt', map_location='cpu'))
    except FileNotFoundError:
        print(f"trial {trial} not found, skipping")
        continue

    W_E = model.embed.weight.detach().numpy()
    X   = W_E[1:]

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, qr_tokens)
    lda_acc = lda.score(X, qr_tokens)
    lda_accs.append(lda_acc)

    # 1D LDA projection accuracy
    lda_direction = lda.coef_[0]
    X_lda = (X @ lda_direction).reshape(-1, 1)
    acc_1d = cross_val_score(LogisticRegression(max_iter=1000), X_lda, qr_tokens, cv=5).mean()
    lda_1d_accs.append(acc_1d)

    projections = X @ lda_direction
    all_projections.append(projections)
    all_labels.append(qr_tokens)
    all_trials.append(trial)

    print(f"trial {trial} | LDA accuracy={lda_acc:.3f} | 1D LDA projection accuracy={acc_1d:.3f}")

# ── Plot 1: LDA projection histograms ────────────────────────────────────────

fig1 = make_subplots(rows=1, cols=len(all_trials),
                     subplot_titles=[f'Trial {t}' for t in all_trials])

for i, (proj, labels, trial) in enumerate(zip(all_projections, all_labels, all_trials)):
    # normalize projection to [-1, 1] for comparability across trials
    proj_norm = (proj - proj.mean()) / proj.std()
    fig1.add_trace(go.Histogram(
        x=proj_norm[labels == 1], name='QR', marker_color='blue',
        opacity=0.6, showlegend=(i == 0), nbinsx=20
    ), row=1, col=i+1)
    fig1.add_trace(go.Histogram(
        x=proj_norm[labels == 0], name='NR', marker_color='red',
        opacity=0.6, showlegend=(i == 0), nbinsx=20
    ), row=1, col=i+1)

fig1.update_layout(
    title='LDA projection (normalized): QR vs NR separation by trial',
    barmode='overlay', height=350
)
fig1.show()

# ── Plot 2: LDA accuracy vs 1D projection accuracy ───────────────────────────

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=[f'trial {t}' for t in all_trials],
    y=lda_accs,
    name='LDA accuracy',
    marker_color='blue', opacity=0.7
))
fig2.add_trace(go.Bar(
    x=[f'trial {t}' for t in all_trials],
    y=lda_1d_accs,
    name='1D LDA projection accuracy',
    marker_color='red', opacity=0.7
))
fig2.add_hline(y=0.99, line_dash='dash', line_color='black', annotation_text='0.99')
fig2.update_layout(
    title='QR classification: LDA vs 1D LDA projection accuracy',
    xaxis_title='Trial',
    yaxis_title='Accuracy',
    yaxis=dict(range=[0, 1.05]),
    barmode='group'
)
fig2.show()

fig1.write_image("plots/lda_projection.png")
fig2.write_image("plots/accuracy_comparison.png")