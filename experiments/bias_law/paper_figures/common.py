"""Shared paths, plotting style and fast helpers for the two-moment paper experiments."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))    # repository root
FIG = os.path.join(ROOT, 'figures', 'bias_law')
RES = os.path.join(ROOT, 'results', 'bias_law')
DATA = os.path.join(RES, 'figure_data')
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'bias_law', 'checks'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'bias_law', 'a1'))

OI = dict(blue='#0072B2', verm='#D55E00', green='#009E73', orange='#E69F00', pink='#CC79A7')
COLS = list(OI.values())


def style():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0,
        'lines.markersize': 3.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'axes.titlesize': 8, 'errorbar.capsize': 1.5,
    })
    import matplotlib.pyplot as plt
    return plt


def tag(ax, s, x=0.03, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', va='top', ha='left', fontsize=8)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=200)


def dump(obj, name):
    json.dump(obj, open(os.path.join(DATA, name), 'w'), indent=1)


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def polar2(M):
    """Exact polar factor of (S,2,d) matrices: (M M^T)^{-1/2} M with the closed-form 2x2 inverse square root."""
    a = np.einsum('nj,nj->n', M[:, 0], M[:, 0]); c = np.einsum('nj,nj->n', M[:, 1], M[:, 1])
    b = np.einsum('nj,nj->n', M[:, 0], M[:, 1])
    s = np.sqrt(a * c - b * b); t = np.sqrt(a + c + 2 * s)          # sqrt(A) = (A + s I)/t ; inverse = adj/det
    out = np.empty_like(M)
    out[:, 0] = ((c + s)[:, None] * M[:, 0] - b[:, None] * M[:, 1]) * (t / (s * t * t))[:, None]
    out[:, 1] = ((a + s)[:, None] * M[:, 1] - b[:, None] * M[:, 0]) * (t / (s * t * t))[:, None]
    return out


def exact_q(K, d):
    """q_{K,d} = K + d - m3/m1 to high precision (de Bruijn Pfaffian formula of experiments/bias_law/a1/wish.py); returns (q, check, dps).
    The working precision is raised until the Pfaffian matrix inverts and the exact identity E tr W^2 = Kd(K+d+1) holds to 1e-25."""
    import mpmath as mp, wish
    if K > d:
        K, d = d, K
    dps = 60 + 3 * K
    while True:
        mp.mp.dps = dps
        try:
            w = wish.Wish(K, d)
            q = w.q()
            chk = abs(w.Etr(2) / (K * d * (K + d + 1)) - 1)
            if chk < mp.mpf(10) ** -25:
                return float(q), float(chk), dps
        except ZeroDivisionError:
            pass
        dps *= 2
        if dps > 6000:
            raise RuntimeError(f'exact_q({K},{d}) did not stabilise')


def a_consts(K, d, q):
    D = (K - 1) * (d - 1) * (K + 2) * (d + 2)
    return (3 * ((K - 1) * (d - 1) - 2 * (K + d + 1) * q) / D, 6 * ((K * d + 2) * q + (K - 1) * (d - 1)) / D)
