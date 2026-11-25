"""Style and helpers shared by the explanatory figures of the paper and of its lecture.

Two families of figures use this module (each drawn by one script of paper/scripts/):

  figE*  explanatory figures for the paper itself (paper/figures/figE*.pdf and .png), included by
         paper/appendix_explanatory_figures.tex; mode "paper": the sizes of experiments/plotstyle.py (figures about
         6.7 in wide, included at the text width 6.5 in, smallest text 7.5 pt);
  figT*  figures for the lecture that presents the paper (paper/talk/figures/figT*.pdf and .png), not meant for the
         paper; mode "talk": drawn at their natural size on a 16:9 beamer slide (160 mm x 90 mm, i.e. 6.30 x 3.54 in,
         body text 11 pt), at most TALK_W x TALK_H, smallest text 8.5 pt, saved on a fixed canvas so that the states
         of a progressive build (name_s1, name_s2, ...) superpose exactly.

The palette, the inks and the rc parameters are those of experiments/plotstyle.py (the style of the four figures of the
paper); nothing is redefined here except the sizes of the talk mode. Rules kept from that style: categorical hues in
their fixed order and attached to an entity (never to a rank), text in ink (never in a series colour), one y axis per
panel, solid hairline grid, a legend or direct labels whenever two series share a panel, markers with a ring in the
surface colour. The two status colours GOOD and CRITICAL are reserved for "kept / lost" marks and always come with a
word or a symbol.

Usage:
    import explain_style as es
    ps = es.setup("talk")                      # or "paper"; returns the plotstyle module (ps.BLUE, ps.INK, ...)
    fig = es.figure("talk", height=2.6)        # constrained layout; width: the width of the mode unless given
    ...
    es.save(fig, "figT03_polar_factor", "talk", note="2 x 2 example; exact SVD")

Run the scripts from the project root with single-threaded BLAS, e.g.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT03_polar_factor.py
"""
from __future__ import annotations

import os
import sys
import types

sys.dont_write_bytecode = True          # importing project modules must not create or refresh any __pycache__ file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                 # paper/scripts -> project root
PAPER_FIGURES = os.path.join(ROOT, "paper", "figures")
TALK_FIGURES = os.path.join(ROOT, "paper", "talk", "figures")

# sizes ----------------------------------------------------------------------------------------------------------------
PAPER_W = 6.7                 # figsize width of a full-width paper figure (plotstyle.FIG_W)
PAPER_TEXTWIDTH = 6.5         # width at which the paper includes it
SLIDE_W, SLIDE_H = 6.30, 3.54  # 16:9 beamer slide, 160 mm x 90 mm, in inches
TALK_W, TALK_H = 5.90, 2.75   # largest figure on a slide that keeps its title line and a one-line footer
MODES = {
    "paper": dict(width=PAPER_W, base=9.0, title=9.0, label=8.5, tick=8.0, small=7.5, lw=2.0, ms=4.6, dpi=200,
                  min_text=7.0, folder=PAPER_FIGURES),
    "talk": dict(width=TALK_W, base=10.0, title=10.5, label=10.0, tick=9.0, small=8.5, lw=2.2, ms=6.0, dpi=300,
                 min_text=8.0, folder=TALK_FIGURES),
}

# status colours (reserved: "kept / lost", "holds / fails"; never a data series), from the palette of the project style
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

_state = {"mode": None, "ps": None}


def _plotstyle():
    """The style module of the project (experiments/plotstyle.py), imported as the scripts of figures 2-4 do."""
    if _state["ps"] is None:
        sys.modules.setdefault("common", types.SimpleNamespace(FIGURES=PAPER_FIGURES))  # plotstyle needs common.FIGURES
        exp = os.path.join(ROOT, "experiments")
        if exp not in sys.path:
            sys.path.insert(0, exp)
        import plotstyle as ps
        _state["ps"] = ps
    return _state["ps"]


def setup(mode="paper"):
    """Apply the project style in the given mode and return the plotstyle module (colours: ps.BLUE, ps.ORANGE, ...,
    inks: ps.INK[...]). The module attribute SMALL of plotstyle is the smallest text of the paper mode; use
    explain_style.small() for the smallest text of the current mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    ps = _plotstyle()
    ps.setup()
    m = MODES[mode]
    plt = ps.plt
    plt.rcParams.update({
        "font.size": m["base"], "axes.titlesize": m["title"], "axes.labelsize": m["label"],
        "figure.labelsize": m["label"], "figure.titlesize": m["title"] + 0.5,
        "xtick.labelsize": m["tick"], "ytick.labelsize": m["tick"], "legend.fontsize": m["small"],
        "lines.linewidth": m["lw"], "lines.markersize": m["ms"], "savefig.dpi": m["dpi"],
        "mathtext.fontset": "dejavusans",
    })
    if mode == "talk":                       # fixed canvas: the states of a build superpose, the size is the design size
        plt.rcParams.update({"savefig.bbox": "standard", "savefig.pad_inches": 0.0})
    _state["mode"] = mode
    return ps


def mode():
    if _state["mode"] is None:
        raise RuntimeError("call explain_style.setup(mode) first")
    return _state["mode"]


def small():
    """Smallest font size of the current mode (legends, annotations, notes)."""
    return MODES[mode()]["small"]


def figure(mode_=None, width=None, height=2.5):
    """Empty figure with constrained layout. Width: the width of the mode unless given (talk: at most TALK_W; a
    half-slide figure is about 2.9 in). Height in inches (talk: at most TALK_H)."""
    m = mode_ or mode()
    ps = _plotstyle()
    w = MODES[m]["width"] if width is None else width
    if m == "talk" and (w > TALK_W + 1e-9 or height > TALK_H + 1e-9):
        raise ValueError(f"a talk figure is at most {TALK_W} x {TALK_H} in (asked: {w} x {height})")
    return ps.plt.figure(figsize=(w, height), constrained_layout=True)


# colours --------------------------------------------------------------------------------------------------------------
def tint(colour, amount=0.14, surface=None):
    """The colour blended over the surface (amount = share of the colour): a light fill for diagram boxes and bands."""
    from matplotlib.colors import to_hex, to_rgb
    ps = _plotstyle()
    c, s = to_rgb(colour), to_rgb(surface or ps.INK["surface"])
    return to_hex(tuple(amount * a + (1.0 - amount) * b for a, b in zip(c, s)))


# diagram primitives (axes without frame, data coordinates) ----------------------------------------------------------
def diagram_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0), equal=False):
    """Turn an axes into a canvas for a schematic: no frame, no ticks, no grid, the given data window."""
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    return ax


def box(ax, x, y, w, h, text="", colour=None, fill=None, edge=None, fontsize=None, weight="normal", text_colour=None,
        rounding=0.02, lw=0.9, zorder=3, ha="center", va="center", pad_x=0.0, linespacing=1.25):
    """Rounded box centred at (x, y) with its text in ink. colour: a categorical hue (fill = its tint, edge = the hue)
    or None (neutral: surface fill, axis-grey edge). Returns (patch, text artist)."""
    from matplotlib.patches import FancyBboxPatch
    ps = _plotstyle()
    if colour is None:
        fc = fill or ps.INK["surface"]
        ec = edge or ps.INK["axis"]
    else:
        fc = fill or tint(colour)
        ec = edge or colour
    r = rounding * min(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
    r = min(r, 0.5 * w, 0.5 * h)
    patch = FancyBboxPatch((x - w / 2 + r, y - h / 2 + r), w - 2 * r, h - 2 * r,
                           boxstyle=f"round,pad={r},rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=zorder,
                           mutation_aspect=1.0)
    ax.add_patch(patch)
    tx = {"center": x, "left": x - w / 2 + pad_x, "right": x + w / 2 - pad_x}[ha]
    txt = ax.text(tx, y, text, ha=ha, va=va, fontsize=fontsize or small(), fontweight=weight,
                  color=text_colour or ps.INK["primary"], zorder=zorder + 1, linespacing=linespacing)
    return patch, txt


def arrow(ax, p0, p1, colour=None, lw=1.2, style="-|>", rad=0.0, zorder=2, mutation_scale=9.0, ls="-",
          shrink_a=0.0, shrink_b=0.0):
    """Arrow from p0 to p1 (data coordinates) in secondary ink unless a colour is given; rad bends it."""
    from matplotlib.patches import FancyArrowPatch
    ps = _plotstyle()
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mutation_scale, lw=lw, ls=ls,
                        color=colour or ps.INK["secondary"], connectionstyle=f"arc3,rad={rad}", zorder=zorder,
                        shrinkA=shrink_a, shrinkB=shrink_b, joinstyle="round", capstyle="round")
    ax.add_patch(a)
    return a


def label(ax, x, y, text, fontsize=None, colour=None, ha="center", va="center", weight="normal", zorder=6, **kw):
    """Text in ink (secondary by default) at data coordinates."""
    ps = _plotstyle()
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize or small(), color=colour or ps.INK["secondary"],
                   fontweight=weight, zorder=zorder, **kw)


def status_tag(ax, x, y, tag, fontsize=None, zorder=6):
    """Status tag of the paper, e.g. "[p]" proved, "[m]" measured exactly, "[e]" experiment, "[f]" formal expansion:
    small capitals in a neutral rounded label."""
    ps = _plotstyle()
    return ax.text(x, y, tag, ha="center", va="center", fontsize=fontsize or small(), color=ps.INK["secondary"],
                   zorder=zorder, bbox=dict(boxstyle="round,pad=0.22", fc=ps.INK["page"], ec=ps.INK["axis"], lw=0.6))


def mark(ax, x, y, good, size=None, zorder=6):
    """A "kept" (check) or "lost" (cross) mark in the status colours; always put a word next to it."""
    return ax.text(x, y, "✓" if good else "✗", ha="center", va="center", fontsize=size or (small() + 3),
                   color=GOOD if good else CRITICAL, fontweight="bold", zorder=zorder)


def panel_title(ax, text):
    """Panel title in the style of the paper's figures: bold, left-aligned, e.g. "(a) two floors"."""
    return ax.set_title(text, loc="left")


def scope_note(fig, text, x=0.012, y=0.012, fontsize=None):
    """Scope line of a talk figure (instance, seeds, grid, status), in muted ink at the bottom left of the canvas.
    Constrained layout does not reserve room for it: pass note= to save(), which draws it and reserves the room."""
    ps = _plotstyle()
    return fig.text(x, y, text, ha="left", va="bottom", fontsize=fontsize or max(small() - 0.5, 7.0),
                    color=ps.INK["muted"])


# progressive builds ---------------------------------------------------------------------------------------------------
class Build:
    """States of a talk figure revealed step by step. Register the artists that appear at state k (k = 1, 2, ...);
    artists that are not registered are always visible. save() writes name_s1, ..., name_sK (artists of the states
    <= k visible) and the complete figure under `name`. The layout is frozen before the first state is written, so
    the states superpose exactly: build the complete figure first, then call save()."""

    def __init__(self):
        self.items = []
        self.legends = []

    def legend(self, ax, **kw):
        """Legend inside the axes that lists, at each state, only the labelled artists visible at that state (a
        plain legend would list the hidden ones too). Call it once, after the artists are drawn and registered;
        kw: the arguments of plotstyle.legend (loc inside the axes, ncol, ...)."""
        self.legends.append((ax, kw))
        return self._draw_legend(ax, kw)

    @staticmethod
    def _draw_legend(ax, kw):
        ps = _plotstyle()
        old = ax.get_legend()
        if old is not None:
            old.remove()
        handles, labels = ax.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(handles, labels) if _visible(h)]
        if not keep:
            return None
        return ps.legend(ax, handles=[h for h, _ in keep], labels=[l for _, l in keep], **kw)

    def add(self, state, *artists):
        """Register artists (or lists/containers of artists) for the given state; returns what it was given."""
        flat = []
        for a in artists:
            self._flatten(a, flat)
        self.items.extend((int(state), a) for a in flat)
        return artists[0] if len(artists) == 1 else artists

    @staticmethod
    def _flatten(a, out):
        if a is None:
            return
        if hasattr(a, "set_visible"):
            out.append(a)
            if hasattr(a, "get_children") and type(a).__name__ in ("ErrorbarContainer", "BarContainer", "StemContainer"):
                for c in a.get_children():
                    Build._flatten(c, out)
            return
        try:
            for b in a:
                Build._flatten(b, out)
        except TypeError:
            raise TypeError(f"cannot register {a!r} in a build")

    @property
    def n_states(self):
        return max((s for s, _ in self.items), default=1)

    def show(self, k):
        for s, a in self.items:
            a.set_visible(s <= k)
        for ax, kw in self.legends:
            self._draw_legend(ax, kw)


def _visible(handle):
    """Whether a legend handle (an artist or a container of artists) is visible."""
    if hasattr(handle, "get_visible"):
        return handle.get_visible()
    try:
        return any(_visible(c) for c in handle)
    except TypeError:
        return True


def save(fig, name, mode_=None, note=None, build=None, formats=("pdf", "png"), check=True):
    """Save the figure as PDF and PNG in the folder of the mode (paper/figures or paper/talk/figures), print its size
    and its smallest text, and report texts that leave the canvas or overlap (diagnostic, not an error).
    note: scope line of a talk figure (see scope_note()). build: a Build; its states are written as name_s1, name_s2, ...
    Returns the list of the written paths."""
    m = mode_ or mode()
    ps = _plotstyle()
    plt = ps.plt
    folder = MODES[m]["folder"]
    os.makedirs(folder, exist_ok=True)
    if note is not None:
        scope_note(fig, note)
        eng = fig.get_layout_engine()
        if eng is not None:
            h = fig.get_figheight()
            eng.set(rect=(0.0, min(0.12, 0.2 / h), 1.0, 1.0 - min(0.12, 0.2 / h)))
    fig.canvas.draw()
    fig.canvas.draw()                                    # constrained layout settles in two passes
    written = []
    if check:
        _report(fig, m)
    if build is not None and build.n_states > 1:
        fig.set_layout_engine("none")                    # freeze the layout: the states superpose
        for k in range(1, build.n_states + 1):
            build.show(k)
            for ext in formats:
                p = os.path.join(folder, f"{name}_s{k}.{ext}")
                fig.savefig(p)
                written.append(p)
        build.show(build.n_states)
    for ext in formats:
        p = os.path.join(folder, f"{name}.{ext}")
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    rel = os.path.relpath(folder, ROOT)
    states = f" (+ {build.n_states} states)" if build is not None and build.n_states > 1 else ""
    print(f"saved {rel}/{name}.{{{','.join(formats)}}}{states}", flush=True)
    return written


def _hidden_tick_labels(fig):
    """Tick labels that matplotlib keeps but does not draw (ticks outside the view, axes switched off)."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) or 1.0
            off = not ax.axison or not axis.get_visible()
            for ticks in (axis.get_major_ticks(), axis.get_minor_ticks()):
                for tk in ticks:
                    loc = tk.get_loc()
                    if off or loc < lo - 1e-9 * span or loc > hi + 1e-9 * span:
                        hidden.add(id(tk.label1))
                        hidden.add(id(tk.label2))
                    else:
                        if not tk.label1.get_visible() or not tk.get_visible():
                            hidden.add(id(tk.label1))
                        if not tk.label2.get_visible() or not tk.get_visible():
                            hidden.add(id(tk.label2))
            if off:
                hidden.add(id(axis.label))
                hidden.add(id(axis.offsetText))
    return hidden


def _texts(fig):
    """The texts that are drawn (visible, non-empty, not a tick label outside the view)."""
    from matplotlib.text import Text
    hidden = _hidden_tick_labels(fig)
    out = []
    for t in fig.findobj(Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip():
            continue
        a = t.axes
        if a is not None and not a.get_visible():
            continue
        out.append(t)
    return out


def _report(fig, m):
    """Size, smallest text at the inclusion scale, texts outside the canvas, pairs of overlapping texts."""
    r = fig.canvas.get_renderer()
    W, H = fig.get_figwidth(), fig.get_figheight()
    texts = _texts(fig)
    sizes = [t.get_fontsize() for t in texts]
    if m == "paper":
        bb = fig.get_tightbbox(r)
        pad = 2 * _plotstyle().plt.rcParams["savefig.pad_inches"]
        w, h = bb.width + pad, bb.height + pad
        k = PAPER_TEXTWIDTH / w
        print(f"   {w:.2f} x {h:.2f} in; included at {PAPER_TEXTWIDTH:g} in: scale {k:.2f}, height {h * k:.2f} in, "
              f"smallest text {min(sizes):g} pt -> {min(sizes) * k:.1f} pt", flush=True)
        smallest = min(sizes) * k
    else:
        print(f"   {W:.2f} x {H:.2f} in on a {SLIDE_W:g} x {SLIDE_H:g} in slide; smallest text {min(sizes):g} pt",
              flush=True)
        smallest = min(sizes)
    if smallest < MODES[m]["min_text"] - 1e-9:
        print(f"   WARNING: smallest text {smallest:.1f} pt is below {MODES[m]['min_text']:g} pt", flush=True)
    dpi = fig.dpi
    boxes = []
    for t in texts:
        try:
            b = t.get_window_extent(r)
        except Exception:
            continue
        if b.width <= 0 or b.height <= 0:
            continue
        boxes.append((t, b))
        if m == "talk" and (b.x0 < -1 or b.y0 < -1 or b.x1 > W * dpi + 1 or b.y1 > H * dpi + 1):
            print(f"   WARNING: text leaves the canvas: {t.get_text()!r}", flush=True)
    n = 0
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox > 1.5 and oy > 1.5:
                n += 1
                if n <= 12:
                    print(f"   WARNING: texts overlap: {ti.get_text()!r} / {tj.get_text()!r}", flush=True)
    if n > 12:
        print(f"   WARNING: {n - 12} more overlapping pairs", flush=True)
