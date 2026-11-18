"""Momentum-based variance reduction (MVR) for Muon: the published state of the art of variance-reduced Muon.

One parameterization covers the family (G_t = minibatch gradient at W_t, lam W included; Gt_{t-1} = a minibatch
gradient at the previous point W_{t-1}); it is the one of Algorithm 1 of Chang, Liu, Zhou and Yuan
(arXiv:2509.15816: "MVR1" with the previous batch, "MVR2" with the same batch), who also note that Nesterov momentum
is the member g = 1 - beta:

    M_t = beta M_{t-1} + (1 - beta) G_t + g beta (G_t - Gt_{t-1}),        O_t = orth(M_t),        W_{t+1} = W_t - eta_t O_t

    g = 0                              heavy-ball Muon (EMA momentum, no correction)
    g = 1 - beta, previous batch       Nesterov Muon: with U_t = (1 - beta) G_t + beta B_t and B_t the EMA of the
                                       gradients, U_t = beta U_{t-1} + (1 - beta) G_t + beta (1 - beta)(G_t - G_{t-1})
    g in [0.005, 0.025], same batch    MARS-M (Liu, Yuan and Gu, arXiv:2510.21800): the correction
                                       C_t = G_t + g (beta / (1 - beta)) (G_t - Gt_{t-1}) is clipped to Frobenius norm
                                       `clip` (1 in the paper) before entering the EMA, M_t = beta M_{t-1} + (1 - beta) C_t
    g = 1, same batch                  STORM / MVR2 (Cutkosky and Orabona; Chang et al. arXiv:2509.15816; Qian et al.
                                       arXiv:2512.16598): M_t = G_t + beta (M_{t-1} - Gt_{t-1})

`same_batch=True` evaluates Gt_{t-1} on the batch of iteration t (2b gradients per iteration, b at the first one);
`same_batch=False` reuses the gradient computed at iteration t - 1 on its own batch (b gradients per iteration: the
"approximate" variant of MARS-M and the form hidden in Nesterov momentum).

Step length: 'const' (Muon: eta_t = lr_t * scale, scale = max(1, K/d)^(1/2), with decoupled weight decay
W <- (1 - lr_t weight_decay) W) or 'sharp' (eta_t = gamma <M_t, O_t>, the nuclear-norm scaling of PolarGrad applied to
the MVR buffer). With constant parameters the MVR estimator keeps a noise floor on a finite sum: at W_* the buffer is an
exponential average of the zero-mean per-batch gradients g_B(W_*) (variance of order (1 - beta) / (1 + beta) times the
minibatch variance for g = 1), which does not vanish; this is the difference with the SAGA / SVRG estimators of
vrmuon.py. With decaying parameters the floor disappears at a sublinear rate: Chang et al. prove last-iterate
objective-gap rates O(T^-1/4) (MVR1) and O(T^-1/3) (MVR2) under a Polyak-Lojasiewicz condition with horizon-free
schedules. The norm-scaled step on a STORM buffer is close to OptMuon-I (Yuan, arXiv:2606.08783: Frobenius-norm scaling
with an AdaGrad-norm coefficient).
"""
from __future__ import annotations

import numpy as np

from .algorithms import _Monitor, _batch_stream, _default_record_every, lr_factors
from .polar import get_orth


def mvr_muon(prob, W0, n_iters, batch, lr=None, beta=0.95, g=1.0, same_batch=True, clip=None, orth="ns5",
             length="const", gamma=None, schedule="const", seed=0, weight_decay=0.0, sampling="iid", warmup=0.0,
             decay_frac=0.2, orth_kw=None, gauge=True, record_every=None, stop_Fgap=None, stop_dist2=None,
             max_time=None, max_grad_evals=None, name=None):
    """Muon with the momentum-based variance-reduced buffer M_t of the module docstring. Returns the dict of
    algorithms._Monitor (traces iters, grad_evals, Fgap, dist2_F, dist2_op, eta) plus the parameters.
    gauge=True applies prob.remove_gauge to the buffer before the orthogonalization (softmax; no effect for LS)."""
    if length not in ("const", "sharp"):
        raise ValueError(f"unknown length rule {length!r}")
    if length == "const" and lr is None:
        raise ValueError("length 'const' needs lr")
    if length == "sharp" and gamma is None:
        raise ValueError("length 'sharp' needs gamma")
    K, d = prob.K, prob.d
    fn = get_orth(orth, **(orth_kw or {}))
    orth_fn = (lambda M: fn(prob.remove_gauge(M))) if gauge else fn
    scale = max(1.0, K / d) ** 0.5
    W = np.array(W0, dtype=np.float64, copy=True)
    const_lr = isinstance(schedule, str) and schedule == "const" and not warmup
    if lr is not None and not const_lr and n_iters > 50_000_000:
        raise ValueError("a learning-rate schedule needs the number of iterations n_iters it spans (at most 5e7)")
    lrs = None if (lr is None or const_lr) else float(lr) * lr_factors(schedule, n_iters, warmup, decay_frac)
    rng = np.random.default_rng(seed)
    record_every = _default_record_every(prob, batch) if record_every is None else record_every
    name = name or f"MVR-Muon[g={g:g},{'same' if same_batch else 'prev'},{length}]"
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    ge = 0
    eta = float("nan")
    mon.record(0, ge, W, eta=eta)
    M = W_prev = G_prev = None
    clipped = 0
    t = 0
    for t, idx in enumerate(_batch_stream(rng, prob.n, batch, n_iters, sampling), start=1):
        G = prob.grad_batch(W, idx)
        ge += batch
        if M is None:
            C = G
        else:
            if same_batch:
                Gp = prob.grad_batch(W_prev, idx)
                ge += batch
            else:
                Gp = G_prev
            C = G + (g * beta / (1.0 - beta)) * (G - Gp)
        if clip is not None:
            nrm = float(np.sqrt(np.vdot(C, C)))
            if nrm > clip:
                C = C * (clip / nrm)
                clipped += 1
        M = C.copy() if M is None else beta * M + (1.0 - beta) * C
        O = orth_fn(M)
        W_prev = W.copy()
        if not same_batch:
            G_prev = G
        if length == "const":
            lr_t = float(lr) if lrs is None else lrs[t - 1]
            eta = lr_t * scale
            if weight_decay:
                W *= 1.0 - lr_t * weight_decay
        else:
            eta = gamma * float(np.vdot(M, O))
        W -= eta * O
        if mon.due(t) and mon.record(t, ge, W, eta=eta):
            break
        if max_grad_evals is not None and ge >= max_grad_evals:
            break
    mon.finish(t, ge, W, eta=eta)
    return mon.result(W, batch=batch, lr=lr, beta=beta, g=g, same_batch=same_batch, clip=clip, clipped=clipped,
                      orth=orth if isinstance(orth, str) else "custom", length=length, gamma=gamma, schedule=schedule,
                      seed=seed, weight_decay=weight_decay, sampling=sampling, scale=scale, n=prob.n)
