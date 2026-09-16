"""Region-orientation map: the three Libyan Arabic macro-dialect regions.

A background/methods figure that places the corpus's three labels geographically —
Tripolitania (west), Cyrenaica (east), Fezzan (south) — for readers who do not know
Libya's internal geography. It is the visual companion to the corpus tables.

Design choices (and why):
  - Soft gradient fill, NOT hard borders. The fill is an inverse-distance blend of the
    three regions' dialect-documented anchor cities, so the colour transitions gradually.
    This matches the paper's own finding that the regions are not cleanly separable
    (Fezzan leaks west to Tripolitania; the t-SNE shows heavy overlap) and the dialectology
    literature, which reports a continuum rather than sharp linguistic boundaries.
  - Colours and names match the rest of the paper exactly (REGION_PAL below ==
    make_corpus_figures.py): Tripolitania #2980b9, Cyrenaica #27ae60, Fezzan #e67e22.
  - No baked-in title or citation footer: in ACL/LaTeX the caption carries those. The
    suggested caption is at the bottom of this file. The figure stays clean so it can be
    dropped in at \\columnwidth or as a figure* without redundant text.
  - Dependency-light: pure json + numpy + matplotlib (matplotlib.path for point-in-polygon).
    No shapely, no geopandas. The country boundary is the bundled assets/libya_boundary.geojson.

City coordinates are plain geographic facts; the cited sources support the regional
*classification* and which cities have published dialect descriptions, not the lat/lon
(see docs/region_map_notes.md).

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_region_map.py
Writes outputs/figures/paper/C0_region_map.{png,pdf} (PDF is the vector copy for the paper).
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

# Paper palette (identical to make_corpus_figures.py so the map matches C2/C5/...).
REGION_PAL = {"Tripolitania": "#2980b9", "Cyrenaica": "#27ae60", "Fezzan": "#e67e22"}


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


# Region order matches config.REGIONS. "dir" is the cardinal word used in the paper text.
REGIONS = [
    {"name": "Tripolitania", "dir": "West",  "anchors": [("Tripoli", 32.8872, 13.1913, True),
                                                          ("Misrata", 32.3754, 15.0925, False)]},
    {"name": "Cyrenaica",    "dir": "East",  "anchors": [("Benghazi", 32.1167, 20.0667, False),
                                                          ("Al-Bayda", 32.7627, 21.7551, False)]},
    {"name": "Fezzan",       "dir": "South", "anchors": [("Sabha", 27.0377, 14.4283, False),
                                                          ("Hun (El Jufra)", 29.1264, 15.9477, False)]},
]

# Context cities: historical-region geography only (no dedicated dialect study cited).
CONTEXT_CITIES = {
    "Tripolitania": [("Zawiya", 32.7571, 12.7280), ("Al-Khums", 32.6486, 14.2614),
                     ("Gharyan", 32.1722, 13.0203)],
    "Cyrenaica":    [("Derna", 32.7625, 22.6367), ("Tobruk", 32.0836, 23.9764),
                     ("Ajdabiya", 30.7554, 20.2263)],
    "Fezzan":       [("Murzuq", 25.9136, 13.9183), ("Awbari", 26.5921, 12.7805)],
}

# Berber/Amazigh enclaves (Benkato & Pereira 2016): NOT part of the Arabic continuum.
BERBER_SITES = [
    ("Nafusa Mts.", 31.95, 11.55), ("Zuwarah", 32.9293, 12.0823),
    ("Ghadames", 30.1333, 9.5000), ("Awjila", 29.1081, 21.2922),
    ("Sokna", 29.0833, 17.9333), ("Ghat (Tuareg)", 24.9633, 10.1764),
]

NEIGHBOURS = [
    ("TUNISIA", 31.2, 10.0), ("ALGERIA", 27.7, 9.7), ("NIGER", 21.5, 11.8),
    ("CHAD", 20.5, 17.8), ("SUDAN", 20.6, 24.0), ("EGYPT", 27.0, 25.6),
    ("Mediterranean Sea", 33.45, 16.5),
]


def load_polygons(path: Path):
    """Return (exteriors, all_rings) as lists of (N,2) lon/lat arrays. shapely-free."""
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
    pad = 1.0
    minx, maxx = minx - pad, maxx + pad
    miny, maxy = miny - pad * 0.6, maxy + pad * 0.6

    mean_lat = np.deg2rad((miny + maxy) / 2)
    cos_lat = float(np.cos(mean_lat))

    # ---- soft-region gradient: inverse-distance blend of each region's anchor cities ----
    nx = ny = 460
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    XX, YY = np.meshgrid(xs, ys)

    def dist_deg(lon0, lat0):                  # cosine-corrected planar degrees
        return np.sqrt(((XX - lon0) * cos_lat) ** 2 + (YY - lat0) ** 2)

    L = 2.1                                     # softness scale (degrees)
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

    # point-in-polygon via matplotlib (vectorised; even-odd handles holes)
    compound = MplPath.make_compound_path(*[MplPath(r) for r in all_rings])
    inside = compound.contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(ny, nx)
    rgba = np.dstack([rgb, np.where(inside, 0.85, 0.0)])

    # ---- figure (2-column friendly: map left, key right) ----
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=300)
    fig.subplots_adjust(left=0.02, right=0.72, top=0.97, bottom=0.04)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(rgba, extent=[minx, maxx, miny, maxy], origin="lower",
              interpolation="bilinear", zorder=2)
    for ext in exteriors:
        ax.plot(ext[:, 0], ext[:, 1], color="#1f2937", linewidth=1.3, zorder=4)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1 / cos_lat)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for name, lat, lon in NEIGHBOURS:
        ax.text(lon, lat, name, fontsize=8, color="#9ca3af", style="italic",
                ha="center", va="center", zorder=3, clip_on=True)

    def label(name, lat, lon, **kw):
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], **kw)

    # dialect-documented anchors (filled; capital = star)
    for reg in REGIONS:
        col = _hex_to_rgb(REGION_PAL[reg["name"]]) * 0.55
        for (name, lat, lon, is_cap) in reg["anchors"]:
            ax.scatter(lon, lat, s=140 if is_cap else 78, marker="*" if is_cap else "o",
                       color=col, edgecolor="white", linewidth=0.8, zorder=6)
            label(name, lat, lon, xytext=(5, 4), fontsize=9, fontweight="bold",
                  color="#111827", zorder=7)

    # context cities (open markers)
    offsets = {"Derna": (4, -11)}
    for region, cities in CONTEXT_CITIES.items():
        col = _hex_to_rgb(REGION_PAL[region]) * 0.55
        for (name, lat, lon) in cities:
            ax.scatter(lon, lat, s=42, marker="o", facecolor="white",
                       edgecolor=col, linewidth=1.3, zorder=5)
            label(name, lat, lon, xytext=offsets.get(name, (4, 3)), fontsize=7.3,
                  color="#374151", zorder=6)

    # Berber/Amazigh enclaves (diamonds)
    for (name, lat, lon) in BERBER_SITES:
        ax.scatter(lon, lat, s=60, marker="D", color="#6b21a8",
                   edgecolor="white", linewidth=0.7, zorder=6)
        label(name, lat, lon, xytext=(5, -9), fontsize=7.3, color="#4c1d95",
              style="italic", zorder=7)

    # north arrow + scale bar
    nx_, ny_ = maxx - 1.6, maxy - 0.5
    ax.annotate("", xy=(nx_, ny_), xytext=(nx_, ny_ - 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.6), zorder=8)
    ax.text(nx_, ny_ + 0.12, "N", fontsize=10, fontweight="bold", color="#374151",
            ha="center", zorder=8)
    bar_deg = 300 / (111.32 * cos_lat)
    bx0, by0 = minx + 1.0, miny + 0.6
    ax.plot([bx0, bx0 + bar_deg], [by0, by0], color="#111827", linewidth=2.5, zorder=8)
    ax.text(bx0 + bar_deg / 2, by0 + 0.22, "300 km", fontsize=8, ha="center",
            color="#111827", zorder=8)

    # ---- legends (right margin) ----
    region_patches = [Patch(facecolor=REGION_PAL[r["name"]], edgecolor="none", alpha=0.85,
                            label=f"{r['name']} ({r['dir']})") for r in REGIONS]
    leg1 = ax.legend(handles=region_patches, loc="upper left",
                     bbox_to_anchor=(0.73, 0.95), bbox_transform=fig.transFigure,
                     fontsize=9.7, frameon=True, facecolor="white", edgecolor="#d1d5db",
                     framealpha=0.95, borderpad=0.9, labelspacing=0.7,
                     title="Macro-dialect region", title_fontsize=10)
    ax.add_artist(leg1)

    key = [
        (plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#555", markersize=13,
                    markeredgecolor="white"), "Capital (dialect-documented)"),
        (plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=9,
                    markeredgecolor="white"), "Dialect-documented city"),
        (plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markersize=8,
                    markeredgecolor="#555", markeredgewidth=1.3), "Context city (geography)"),
        (plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#6b21a8", markersize=8,
                    markeredgecolor="white"), "Berber/Amazigh enclave"),
    ]
    leg2 = ax.legend([h for h, _ in key], [l for _, l in key], loc="upper left",
                     bbox_to_anchor=(0.73, 0.66), bbox_transform=fig.transFigure,
                     fontsize=9.0, frameon=True, facecolor="white", edgecolor="#d1d5db",
                     framealpha=0.95, borderpad=0.9, labelspacing=0.8,
                     title="Map key", title_fontsize=10)
    ax.add_artist(leg2)

    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"C0_region_map{ext}", facecolor="white",
                    bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"OK  wrote {OUT / 'C0_region_map.pdf'}  (+ .png)")


# Suggested LaTeX caption (regions/colours already match the paper; all cites are in
# references.bib EXCEPT Owens 1984 and Čéplö 2016 — add those only if you cite them):
#   \caption{The three Libyan Arabic macro-dialect regions used as labels:
#   Tripolitania (west), Cyrenaica (east), and Fezzan (south). Fill is a continuous
#   inverse-distance blend of each region's dialect-documented cities (stars/filled dots),
#   not a hard border, reflecting the gradual dialect transitions reported in the literature
#   \citep{pereira2010tripoli,benkato2014benghazi,benkato2016biblio,essgaer2025computational}.
#   Open dots are context cities; diamonds mark Berber/Amazigh-speaking enclaves, which are
#   outside the Arabic continuum. City coordinates are geographic; the sources support the
#   regional classification, not the coordinates.}

if __name__ == "__main__":
    main()
