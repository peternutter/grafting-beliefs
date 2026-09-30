import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from belief_data import FIGURES
from why_gen import plotstyle

INK = "#111111"
MUTED = "#666666"
ACCENT = "#6d0061"
GRID = "#e6e1e9"
ARM = {"bare": "#a29bad", "native": "#5f7f9c", "graft": "#6b4c9a"}
TITLE = fm.FontProperties(family="STIXGeneral", weight="bold")
TITLE_R = fm.FontProperties(family="STIXGeneral")


def apply():
    plotstyle.apply()
    plt.rcParams.update({
        "font.family": ["STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
        "figure.dpi": plotstyle.HOUSE_DPI, "savefig.dpi": plotstyle.HOUSE_DPI,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.edgecolor": "#cfc8d4", "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "lines.linewidth": 1.2,
    })


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path
