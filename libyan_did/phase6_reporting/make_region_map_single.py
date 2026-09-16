"""Single-column variant of make_region_map.py for the ACL paper Introduction.

Same data and soft-gradient logic as make_region_map.py, but laid out as a compact
portrait figure that drops in at \\columnwidth without downscaling the labels into
illegibility:
  - Portrait figure (map on top, legend row beneath) instead of map-left / key-right.
  - Context-city markers and labels removed (the "minor labels"): the Introduction map
    only needs to orient readers to the three regions + the Berber/Amazigh enclaves.
  - Larger relative fonts so text stays readable at one-column width.
Writes outputs/figures/paper/C0_region_map_single.{png,pdf}.

Run (from libyan_did_pipeline/):  python scripts/make_region_map_single.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
import matplotlib.patheffects as pe       # noqa: E402
from matplotlib.path import Path as MplPath   # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

from libyan_did.shared import config

GEOJSON = config.ASSETS_DIR / "libya_boundary.geojson"
OUT = config.FIGURES_DIR / "paper"
OUT.mkdir(parents=True, exist_ok=True)

REGION_PAL = {"Tripolitania": "#2980b9", "Cyrenaica": "#27ae60", "Fezzan": "#e67e22"}


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


REGIONS = [
    {"name": "Tripolitania", "dir": "West",  "anchors": [("Tripoli", 32.8872, 13.1913, True),
                                                          ("Misrata", 32.3754, 15.0925, False)]},
    {"name": "Cyrenaica",    "dir": "East",  "anchors": [("Benghazi", 32.1167, 20.0667, False),
                                                          ("Al-Bayda", 32.7627, 21.7551, False)]},
    {"name": "Fezzan",       "dir": "South", "anchors": [("Sabha", 27.0377, 14.4283, False),
                                                          ("Hun", 29.1264, 15.9477, False)]},
]

# Berber/Amazigh enclaves (Benkato & Pereira 2016): NOT part of the Arabic continuum.
BERBER_SITES = [
    ("Nafusa Mts.", 31.95, 11.55), ("Zuwarah", 32.9293, 12.0823),
    ("Ghadames", 30.1333, 9.5000), ("Awjila", 29.1081, 21.2922),
    ("Ghat", 24.9633, 10.1764),
]

NEIGHBOURS = [
    ("TUNISIA", 31.2, 10.0), ("ALGERIA", 27.7, 9.7), ("NIGER", 21.5, 11.8),
    ("CHAD", 20.5, 17.8), ("EGYPT", 26.0, 24.9), ("Mediterranean Sea", 31.7, 17.6),
]


def load_polygons(path: Path):
    gj = json.load(open(path, encoding="utf-8"))
    geom = gj["features"][0]["geometry"]
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        raise ValueError(f"unsupported geometry: {geom['type']}")
    exteriors = [np.asarray(p[0], dtype=float) for p in polys]
    all_rings = [np.asarray(r, dtype=float) for p in polys for r in p]
    return exteriors, all_rings


def main() -> None:
    exteriors, all_rings = load_polygons(GEOJSON)
    pts_all = np.vstack(exteriors)
    minx, maxx = pts_all[:, 0].min(), pts_all[:, 0].max()
    miny, maxy = pts_all[:, 1].min(), pts_all[:, 1].max()
    pad = 0.8
    minx, maxx = minx - pad, maxx + pad
    miny, maxy = miny - pad * 0.6, maxy + pad * 0.6

    mean_lat = np.deg2rad((miny + maxy) / 2)
    cos_lat = float(np.cos(mean_lat))

    nx = ny = 460
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    XX, YY = np.meshgrid(xs, ys)

    def dist_deg(lon0, lat0):
        return np.sqrt(((XX - lon0) * cos_lat) ** 2 + (YY - lat0) ** 2)

    L = 2.1
    weights = []
    for reg in REGIONS:
        d_min = None
        for (_name, lat, lon, _cap) in reg["anchors"]:
            d = dist_deg(lon, lat)
            d_min = d if d_min is None else np.minimum(d_min, d)
        weights.append(np.exp(-d_min / L))
    weights = np.array(weights)
    weights /= weights.sum(axis=0)

    rgb = np.zeros((ny, nx, 3))
    for w, reg in zip(weights, REGIONS):
        col = _hex_to_rgb(REGION_PAL[reg["name"]])
        for c in range(3):
            rgb[:, :, c] += w * col[c]

    compound = MplPath.make_compound_path(*[MplPath(r) for r in all_rings])
    inside = compound.contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(ny, nx)
    rgba = np.dstack([rgb, np.where(inside, 0.85, 0.0)])

    # ---- compact portrait figure (map on top, legend row beneath) ----
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.45, 3.05), dpi=300)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.20)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(rgba, extent=[minx, maxx, miny, maxy], origin="lower",
              interpolation="bilinear", zorder=2)
    for ext in exteriors:
        ax.plot(ext[:, 0], ext[:, 1], color="#1f2937", linewidth=1.1, zorder=4)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1 / cos_lat)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for name, lat, lon in NEIGHBOURS:
        ax.text(lon, lat, name, fontsize=6.0, color="#9ca3af", style="italic",
                ha="center", va="center", zorder=3, clip_on=True)

    def label(name, lat, lon, **kw):
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    path_effects=[], **kw)

    # dialect-documented anchors (filled; capital = star). Per-city label offset/alignment
    # to pull the crowded northern coastal pairs apart (Tripoli<->Misrata, Benghazi<->Al-Bayda).
    anchor_label_kw = {
        "Tripoli":  dict(xytext=(-8, 7),   ha="left",  va="center"),
        "Misrata":  dict(xytext=(6, 2), ha="left",  va="top"),
        "Benghazi": dict(xytext=(5, -4), ha="left",  va="top"),
        "Al-Bayda": dict(xytext=(-8, 5),   ha="left",  va="bottom"),
    }
    default_kw = dict(xytext=(4, 3), ha="left", va="bottom")
    for reg in REGIONS:
        col = _hex_to_rgb(REGION_PAL[reg["name"]]) * 0.55
        for (name, lat, lon, is_cap) in reg["anchors"]:
            ax.scatter(lon, lat, s=110 if is_cap else 55, marker="*" if is_cap else "o",
                       color=col, edgecolor="white", linewidth=0.7, zorder=6)
            label(name, lat, lon, fontsize=5.9, fontweight="bold", color="#111827",
                  zorder=7, **anchor_label_kw.get(name, default_kw))

    # Berber/Amazigh enclaves (diamonds). Per-site label offset/alignment, like the cities
    # above: Zuwarah's label moves left of its diamond (clear of the Tripoli star) and Ghat's
    # moves right into the interior (clear of the SW border). Edit an entry to nudge a label.
    berber_label_kw = {
        "Zuwarah": dict(xytext=(-6, 0), ha="right", va="center"),
        "Ghat":    dict(xytext=(7, 0),  ha="left",  va="center"),
    }
    berber_default = dict(xytext=(4, -7), ha="left", va="baseline")
    for (name, lat, lon) in BERBER_SITES:
        ax.scatter(lon, lat, s=42, marker="D", color="#6b21a8",
                   edgecolor="white", linewidth=0.6, zorder=6)
        label(name, lat, lon, fontsize=5.6, color="#4c1d95", style="italic",
              zorder=7, **berber_label_kw.get(name, berber_default))

    # north arrow + scale bar (top-right corner, clear of the Al-Bayda label)
    nx_, ny_ = maxx - 0.6, maxy - 0.6
    ax.annotate("", xy=(nx_, ny_), xytext=(nx_, ny_ - 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.4), zorder=8)
    ax.text(nx_, ny_ + 0.12, "N", fontsize=8.5, fontweight="bold", color="#374151",
            ha="center", zorder=8)
    bar_deg = 300 / (111.32 * cos_lat)
    bx0, by0 = minx + 0.8, miny + 0.5
    ax.plot([bx0, bx0 + bar_deg], [by0, by0], color="#111827", linewidth=2.2, zorder=8)
    ax.text(bx0 + bar_deg / 2, by0 + 0.22, "300 km", fontsize=6.6, ha="center",
            color="#111827", zorder=8)

    # ---- one compact legend row beneath the map ----
    region_patches = [Patch(facecolor=REGION_PAL[r["name"]], edgecolor="none", alpha=0.85,
                            label=f"{r['name']} ({r['dir']})") for r in REGIONS]
    key_markers = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#555", markersize=11,
                   markeredgecolor="white", label="Capital"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=7,
                   markeredgecolor="white", label="Dialect-documented city"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#6b21a8", markersize=7,
                   markeredgecolor="white", label="Berber/Amazigh enclave"),
    ]
    fig.legend(handles=region_patches + key_markers, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.6, frameon=True,
               facecolor="white", edgecolor="#d1d5db", framealpha=0.95,
               borderpad=0.6, labelspacing=0.5, columnspacing=1.2, handletextpad=0.5)

    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"C0_region_map_single{ext}", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"OK  wrote {OUT / 'C0_region_map_single.pdf'}  (+ .png)")


if __name__ == "__main__":
    main()
