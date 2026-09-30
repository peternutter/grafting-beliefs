import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from why_gen import plotstyle

INK = "#111111"
MUTED = "#666666"
GRID = "#e6e1e9"
NATIVE = "#5f7f9c"
GRAFT = "#6b4c9a"
GRAFT_LIGHT = "#aa96c3"
TITLE = plotstyle.TITLE


def apply():
    plotstyle.apply()
    plt.rcParams.update({
        "figure.dpi": plotstyle.HOUSE_DPI, "savefig.dpi": plotstyle.HOUSE_DPI,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.edgecolor": "#cfc8d4", "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def hgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
