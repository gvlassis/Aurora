# Aurora

Unofficial implementation of [Aurora](https://blog.tilderesearch.com/blog/aurora).

Aurora is a Muon variant for **tall matrices**. It runs multiple iterations of alternating projections and Newton-Schulz (NS), controlled by `pp_iterations` (default: 2). The damped projection, controlled by `pp_beta` (default: 0.5) ensures uniform leverage across rows, while NS recovers orthogonality. Aurora **does not track second moments**.

This code is based on my [unofficial code for U-NorMuon](https://github.com/gvlassis/U-NorMuon) and the [official code for Aurora](https://github.com/tilde-research/aurora-release).

## Installation

```
pip install -U git+https://github.com/gvlassis/Aurora
```

## What's in `aurora.py`

- `SingleDeviceAurora` — Single-GPU optimizer.

## Usage

NorMuon is meant to replace Muon for hidden 2D weight matrices, and Aurora is meant to replace NorMuon for **tall** matrices (m>n). Everything else (embeddings, classifier head, gains/biases etc.) should be optimized by AdamW.

```python
from normuon import SingleDeviceNorMuon
from aurora import SingleDeviceAurora

hidden_weights_wide_or_square = [p for p in model.body.parameters() if p.ndim >= 2 and p.shape[0]<=p.shape[1]]
hidden_weights_tall           = [p for p in model.body.parameters() if p.ndim >= 2 and p.shape[0]>p.shape[1]]
hidden_gains_biases           = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params              = [*model.head.parameters(), *model.embed.parameters()]

optimizers = [
    SingleDeviceNorMuon(hidden_weights_wide_or_square, lr=3e-2, momentum=0.95, beta2=0.95, weight_decay=0),
    SingleDeviceAurora(hidden_weights_tall, lr=3e-2, momentum=0.95, weight_decay=0),
    torch.optim.AdamW(hidden_gains_biases+nonhidden_params, lr=3e-3, betas=(0.9,0.95), eps=1e-08, weight_decay=0)
]
```
