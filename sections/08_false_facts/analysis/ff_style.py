import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from why_gen import plotstyle

INK = "#111111"
MUTED = "#666666"
GRID = "#e6e1e9"
ARM = {"bare": "#a29bad", "native": "#5f7f9c", "graft": "#6b4c9a"}
GROUP = {"real": "#4f7a63", "fictional_known": "#c1611f", "fictional_novel": "#8f1d21"}
TITLE = fm.FontProperties(family="STIXGeneral", weight="bold")
TITLE_R = fm.FontProperties(family="STIXGeneral")


def lighten(hexcol, f):
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(v + (255 - v) * f)) for v in (r, g, b))


STAGE_COLOR = {"bare": ARM["bare"], "graft": ARM["graft"], "native": ARM["native"],
               "native_train_matched": lighten(ARM["native"], 0.34),
               "native_serve_matched": lighten(ARM["native"], 0.62)}


def apply():
    plotstyle.apply()
    plt.rcParams.update({
        "font.family": ["STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.edgecolor": "#cfc8d4", "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False, "lines.linewidth": 1.2,
    })


def hgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print("wrote", path.with_suffix(".pdf"))
