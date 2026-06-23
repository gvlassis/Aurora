import torch
import torch.distributed as dist

# From https://github.com/tilde-research/aurora-release/blob/main/src/polar.py
@torch.no_grad()
def polar(G: torch.Tensor) -> torch.Tensor:
    """Polar factor via 12-step simple-quintic Newton-Schulz.

    For a matrix G with SVD G = U Σ V^T, this returns U V^T (the polar factor
    of G). All non-zero singular values are mapped to 1.

    Implementation: 12 iterations of the simple-quintic polynomial
        p(σ) = 2σ - 1.5σ³ + 0.5σ⁵
    which has fixed points at σ ∈ {0, 1, √2}, with σ=1 super-attracting
    (p'(1) = 0). After 12 iterations of cubic-rate convergence, all input
    singular values in (0, √2) are driven to 1 to bf16 precision.

    This is the polar method used by the Modded-NanoGPT track-3 baseline at
    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py
    "not optimizing for wallclock speed". We match it byte-for-byte so that
    optimizers built on this `polar` reproduce leaderboard val_loss curves.

    Args:
        G: input matrix of shape [..., m, n].

    Returns:
        polar(G) of the same shape, in bfloat16. All non-zero singular values
        of G are mapped to 1.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Simple-quintic coefficients: p(σ) = aσ + bσ³ + cσ⁵ with σ=1 super-attracting.
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# From https://github.com/tilde-research/aurora-release/blob/main/src/aurora.py
@torch.no_grad()
def aurora(
    W,
    G,
    momentum,
    eta=0.05,
    weight_decay=0.025,
    mu=0.95,
    nesterov=True,
    pp_iterations=2,
    pp_beta=0.5,
    eps=1e-7,
):
    if W.ndim != 2:
        raise ValueError(f"aurora expects 2D weight tensors, got shape {tuple(W.shape)}")
    if G.shape != W.shape:
        raise ValueError(f"G shape {tuple(G.shape)} must match W shape {tuple(W.shape)}")
    if momentum.shape != W.shape:
        raise ValueError(f"momentum shape {tuple(momentum.shape)} must match W shape {tuple(W.shape)}")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    # Clone when not using Nesterov to avoid scaling the momentum buffer in-place below.
    update = G.lerp_(momentum, mu) if nesterov else momentum.clone()
    # Aurora's leverage-uniform polar via diagonal preconditioning.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: standard polar (no leverage freedom to exploit).
        update = polar(update)
    else:
        # For wide G, transpose to tall, apply, transpose back.
        # polar(G * D) = polar(D * G^T)^T
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U
    # Spectral aspect-ratio scaling (Muon convention).
    update *= max(1, G.size(-2) / G.size(-1)) ** 0.5
    if not update.isfinite().all():
        raise RuntimeError(
            f"aurora produced non-finite update for parameter of shape {tuple(W.shape)}. "
            "Check for NaN/Inf in gradients or an ill-conditioned weight matrix."
        )
    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W

# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class SingleDeviceAurora(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                had_grad = p.grad is not None
                if not had_grad:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                aurora(p, p.grad, state["momentum_buffer"], group["lr"], group["weight_decay"], beta=group["momentum"])

        return loss
