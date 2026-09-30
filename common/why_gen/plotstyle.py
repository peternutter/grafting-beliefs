import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def wilson(p, n, z=1.96):
    if not n or p is None:
        return (None, None)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


FONTS = Path(__file__).parent / "assets" / "fonts"
for _f in FONTS.glob("*.ttf"):
    try:
        fm.fontManager.addfont(str(_f))
    except Exception:
        pass

BODY = "Ubuntu Mono"
TITLE_FAMILY = "Volkhov"
TITLE = fm.FontProperties(family=TITLE_FAMILY, weight="bold")

INK = "#222426"
ACCENT = "#b8740a"

HOUSE_DPI = 300


def apply():
    plt.rcParams.update({
        "font.family": [BODY, "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#3a3a3a",
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "figure.titlesize": 15,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "savefig.dpi": HOUSE_DPI,
        "figure.dpi": HOUSE_DPI,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })
