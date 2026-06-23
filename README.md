# U-NorMuon

Unofficial implementation of **U-NorMuon**, introduced along with Aurora (https://blog.tilderesearch.com/blog/aurora).

U-NorMuon is essentially [NorMuon](https://arxiv.org/abs/2510.05491) with an extra $\sqrt{\frac{n}{m}}$ scaling factor in front of the final lr, so that **tall matrices** do not get too far away from $UV^T$ relative to square matrices.

This code is based on the [official code for NorMuon](https://github.com/zichongli5/NorMuon). Thus, it uses **Muon grafting** instead of Adam grafting (contrary to Algorithm 1 from the official paper). Hence, the final lr is

$$
    \Delta W = - \eta \sqrt{\frac{m}{n}} \sqrt{\frac{n}{m}} \|o\|_F \frac{o'}{\|o'\|_F} = - \eta \|o\|_F \frac{o'}{\|o'\|_F} = - \eta \|o\|_{RMS} \frac{o'}{\|o'\|_{RMS}}
$$

Specifically, we only modify the last line of `unormuon_update()`.

## Installation

```
pip install -U git+https://github.com/gvlassis/U-NorMuon
```

## What's in `unormuon.py`

- `UNorMuon` / `SingleDeviceUNorMuon` — distributed (DDP) and single-GPU optimizers.
- `UNorMuonWithAuxAdam` / `SingleDeviceUNorMuonWithAuxAdam` — bundle UNorMuon for hidden weights with AdamW for the rest.

We recommend **`SingleDeviceUNorMuon`** (no DDP shenanigans or weird conflicts with schedulers).

## Usage

NorMuon is meant to replace Muon for hidden 2D weight matrices, and U-NorMuon is meant to replace NorMuon for **tall** matrices (m>n). Everything else (embeddings, classifier head, gains/biases etc.) should be optimized by AdamW.

```python
from normuon import SingleDeviceNorMuon
from unormuon import SingleDeviceUNorMuon

hidden_weights_wide_or_square = [p for p in model.body.parameters() if p.ndim >= 2 and p.shape[0]<=p.shape[1]]
hidden_weights_tall           = [p for p in model.body.parameters() if p.ndim >= 2 and p.shape[0]>p.shape[1]]
hidden_gains_biases           = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params              = [*model.head.parameters(), *model.embed.parameters()]

optimizers = [
    SingleDeviceNorMuon(hidden_weights_wide_or_square, lr=3e-2, momentum=0.95, beta2=0.95, weight_decay=0),
    SingleDeviceUNorMuon(hidden_weights_tall, lr=3e-2, momentum=0.95, beta2=0.95, weight_decay=0),
    torch.optim.AdamW(hidden_gains_biases+nonhidden_params, lr=3e-3, betas=(0.9,0.95), eps=1e-08, weight_decay=0)
]
```
