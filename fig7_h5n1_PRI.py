# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ARTICLE_DIR = Path(__file__).resolve().parents[0]
MASTER_CSV = Path(r"d:\gaoyuan\Desktop\beijing2\cp\20260207\fig\Fig7_Master_Data_FULL_98k.csv")
META_CSV = Path(r"d:\gaoyuan\Desktop\beijing2\20260311_fig1_7\data\df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")
FIG6_AF_ORIGINAL = Path(
    r"d:\gaoyuan\Desktop\beijing2\98787_article\20260422_article\figure\fig6_A-F_original_before_H5N1_enhancement.png"
)
OUT_DIR = ARTICLE_DIR / "outputs"

EXCLUDED_SUBTYPES = {"H1N1", "H3N2", "Mixed", "Unknown", "nan", "None", ""}

METRIC_COLORS = {
    "SI": mpl.cm.Blues,
    "SDI": mpl.cm.Greens,
    "PRI": mpl.cm.Reds,
}

SUBTYPE_COLORS = {
    "H5N1": "#b2182b",
    "Other H5": "#ef8a62",
    "H7/H9": "#67a9cf",
    "Other": "#bdbdbd",
}

PERIOD_COLORS = {
    "pre-2010": "#9ecae1",
    "2010-2019": "#fdae6b",
    "2020+": "#cb181d",
    "Unknown": "#d9d9d9",
}

CONTINENT_COLORS = {
    "North America": "#4c78a8",
    "Europe": "#f58518",
    "Asia": "#54a24b",
    "South America": "#e45756",
    "Oceania": "#72b7b2",
    "Africa": "#b279a2",
    "Antarctica": "#bab0ac",
    "Unknown": "#d9d9d9",
}

HOST_COLORS = {
    "Poultry": "#d73027",
    "Duck/waterfowl": "#4575b4",
    "Goose/swan": "#74add1",
    "Gull/seabird": "#91bfdb",
    "Raptor": "#f46d43",
    "Shorebird": "#fee090",
    "Mixed/unknown": "#bdbdbd",
    "Other bird": "#a6d96a",
}

B_HOST_COLORS = {
    "Poultry": "#d73027",
    "Waterfowl": "#4575b4",
    "Other wild bird": "#a6d96a",
    "Other/unknown": "#bdbdbd",
}


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 5.7,
            "figure.dpi": 500,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax: plt.Axes, label: str, x: float = -0.035, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="normal",
        color="black",
    )


def load_data() -> pd.DataFrame:
    usecols = [
        "strain_name",
        "Genome_Vector_Score",
        "VRP_Score",
        "Decimal_Date",
        "Country",
        "Host",
        "Host1",
        "Subtype",
        "Host_Group",
    ]
    df = pd.read_csv(MASTER_CSV, usecols=usecols, low_memory=False)
    df["Year_num"] = pd.to_numeric(df["Decimal_Date"], errors="coerce")
    df["Year_int"] = np.floor(df["Year_num"]).astype("Int64")
    df["Subtype"] = df["Subtype"].astype(str)
    df["Host_Group"] = df["Host_Group"].astype(str)
    df["Country"] = df["Country"].fillna("Unknown").astype(str)

    meta = pd.read_csv(META_CSV, usecols=["strain_name", "Continent"], low_memory=False)
    df = df.merge(meta.drop_duplicates("strain_name"), on="strain_name", how="left")
    df["Continent"] = df["Continent"].fillna("Unknown").replace({"unknown": "Unknown"})

    avian = df[
        (df["Host_Group"] == "Avian")
        & (~df["Subtype"].isin(EXCLUDED_SUBTYPES))
        & df["Genome_Vector_Score"].notna()
        & df["VRP_Score"].notna()
    ].copy()

    avian["SI_percentile"] = avian["Genome_Vector_Score"].rank(method="average", pct=True)
    avian["SDI_percentile"] = avian["VRP_Score"].rank(method="average", pct=True)
    avian["PRI"] = (avian["SI_percentile"] + avian["SDI_percentile"]) / 2
    avian["Risk_category"] = risk_categories(avian["PRI"])[0]
    avian["Host_source"] = avian["strain_name"].map(parse_host_source)
    avian["Host_source_B"] = avian["Host_source"].map(broad_host_source)
    avian["Period"] = avian["Year_int"].map(period_label)
    return avian


def risk_categories(values: pd.Series) -> tuple[pd.Categorical, dict[str, float]]:
    q25, q50, q75 = values.quantile([0.25, 0.50, 0.75]).tolist()
    labels = ["Low", "Medium", "High", "Very high"]
    cats = pd.cut(
        values,
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=labels,
        include_lowest=True,
    )
    return cats, {"Q25": q25, "Q50": q50, "Q75": q75}


def period_label(year: object) -> str:
    if pd.isna(year):
        return "Unknown"
    year_int = int(year)
    if year_int >= 2020:
        return "2020+"
    if year_int >= 2010:
        return "2010-2019"
    return "pre-2010"


def parse_host_source(strain_name: str) -> str:
    s = str(strain_name).lower().replace("_", " ")
    if any(x in s for x in ["chicken", "turkey", "quail", "guinea", "pheasant", "poultry"]):
        return "Poultry"
    if any(x in s for x in ["duck", "mallard", "teal", "wigeon", "shoveler", "pintail", "scaup"]):
        return "Duck/waterfowl"
    if any(x in s for x in ["goose", "swan", "brant"]):
        return "Goose/swan"
    if any(x in s for x in ["gull", "tern", "kittiwake", "seabird"]):
        return "Gull/seabird"
    if any(x in s for x in ["eagle", "owl", "hawk", "falcon", "raptor", "vulture"]):
        return "Raptor"
    if any(x in s for x in ["sanderling", "curlew", "sandpiper", "plover", "turnstone", "shorebird"]):
        return "Shorebird"
    if any(x in s for x in ["mixed", "environment", "water", "feces", "unknown"]):
        return "Mixed/unknown"
    return "Other bird"


def broad_host_source(host_source: str) -> str:
    if host_source == "Poultry":
        return "Poultry"
    if host_source in {"Duck/waterfowl", "Goose/swan"}:
        return "Waterfowl"
    if host_source in {"Gull/seabird", "Raptor", "Shorebird", "Other bird"}:
        return "Other wild bird"
    return "Other/unknown"


def subtype_group(subtype: str) -> str:
    subtype = str(subtype)
    if subtype == "H5N1":
        return "H5N1"
    if subtype.startswith("H5"):
        return "Other H5"
    if subtype.startswith(("H7", "H9")):
        return "H7/H9"
    return "Other"


def subtype_stats(avian: pd.DataFrame) -> pd.DataFrame:
    stats = (
        avian.groupby("Subtype")
        .agg(
            n=("strain_name", "size"),
            mean_SI=("SI_percentile", "mean"),
            median_SI=("SI_percentile", "median"),
            mean_SDI=("SDI_percentile", "mean"),
            median_SDI=("SDI_percentile", "median"),
            mean_PRI=("PRI", "mean"),
            median_PRI=("PRI", "median"),
        )
        .reset_index()
    )
    return stats[stats["n"] >= 30].sort_values("mean_PRI", ascending=False).reset_index(drop=True)


def select_representative_strains(avian: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    high = avian.sort_values("PRI", ascending=False).copy()

    h5_post = high[(high["Subtype"] == "H5N1") & (high["Year_int"] >= 2020)]
    h5_diverse = (
        h5_post.groupby(["Country", "Year_int", "Host_source"], dropna=False)
        .head(1)
        .sort_values("PRI", ascending=False)
        .head(22)
    )

    other_high = (
        high[high["Subtype"] != "H5N1"]
        .groupby(["Subtype", "Country"], dropna=False)
        .head(1)
        .sort_values("PRI", ascending=False)
        .head(10)
    )

    medium = avian[(avian["PRI"] >= thresholds["Q25"]) & (avian["PRI"] < thresholds["Q50"])]
    medium_ref = medium.sort_values("PRI", ascending=False).groupby("Subtype", dropna=False).head(1).head(5)

    low = avian[avian["PRI"] < thresholds["Q25"]]
    low_ref = low.sort_values("PRI", ascending=False).groupby("Subtype", dropna=False).head(1).head(5)

    selected = pd.concat([low_ref, medium_ref, other_high, h5_diverse], ignore_index=True)
    selected = selected.drop_duplicates("strain_name").sort_values("PRI").reset_index(drop=True)
    return selected


def _smart_case_host(token: str) -> str:
    token = str(token).replace("_", " ").strip()
    token = re.sub(r"\s+", " ", token)
    if not token:
        return "unknown"
    if any(ch.isdigit() for ch in token):
        return token
    return token.lower()


def _smart_case_place(token: str) -> str:
    token = str(token).replace("_", " ").strip()
    token = re.sub(r"\s+", " ", token)
    if not token:
        return "Unknown"
    protected = {"USA", "UK", "UAE", "USSR", "PRC"}
    if token.upper() in protected:
        return token.upper()
    if any(ch.isdigit() for ch in token):
        return token
    return token.title()


def _extract_year(name: str) -> str:
    hits = re.findall(r"(19\d{2}|20\d{2})", str(name))
    return hits[-1] if hits else "NA"


def _compact_label(label: str, max_len: int = 24) -> str:
    if len(label) <= max_len:
        return label
    parts = label.split("/")
    if len(parts) == 3:
        host, place, year = parts
        host = re.sub(r"\b(common|domestic|wild|american|northern)\b", "", host).strip()
        if len(place) > 8:
            place = place[:8] + "."
        compact = f"{host}/{place}/{year}"
        if len(compact) <= max_len:
            return compact
    return label[: max_len - 1] + "."


def strain_axis_label(name: str, max_len: int = 24) -> str:
    label = str(name).replace("_", " ").replace("Influenza A virus ", "").strip()
    if label.startswith("RG-"):
        label = label[3:]
    parts = label.split("/")
    if len(parts) >= 4 and parts[0].upper() == "A":
        host = _smart_case_host(parts[1])
        place = _smart_case_place(parts[2])
        year = _extract_year(label)
        return _compact_label(f"{host}/{place}/{year}", max_len=max_len)
    return _compact_label(_smart_case_place(label), max_len=max_len)


def draw_panel_a(ax: plt.Axes, stats: pd.DataFrame) -> None:
    plot_stats = stats.sort_values("mean_PRI").reset_index(drop=True).copy()
    metrics = [("SI", "mean_SI"), ("SDI", "mean_SDI"), ("PRI", "mean_PRI")]
    n = len(plot_stats)

    ax.set_xlim(0, n)
    ax.set_ylim(0, len(metrics))
    ax.invert_yaxis()

    for y, (label, col) in enumerate(metrics):
        cmap = METRIC_COLORS[label]
        for x, value in enumerate(plot_stats[col]):
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=cmap(float(value)),
                    edgecolor="white",
                    linewidth=0.25,
                )
            )

    h5_pos = plot_stats.index[plot_stats["Subtype"] == "H5N1"].tolist()
    if h5_pos:
        ax.add_patch(Rectangle((h5_pos[0], 0), 1, len(metrics), fill=False, edgecolor="#9b111e", linewidth=1.4))

    ax.set_yticks(np.arange(len(metrics)) + 0.5)
    ax.set_yticklabels(["SI", "SDI", "PRI"])
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(plot_stats["Subtype"], rotation=52, ha="right", va="top", rotation_mode="anchor", fontsize=5.9)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "H5N1":
            tick.set_color("#9b111e")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0)
    draw_metric_color_legend(ax)
    panel_label(ax, "A")


def draw_annotation_row(ax: plt.Axes, labels: list[str], colors: dict[str, str], row_label: str) -> None:
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    for i, label in enumerate(labels):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=colors.get(label, "#d9d9d9"), edgecolor="white", linewidth=0.22))
    ax.text(-0.55, 0.5, row_label, va="center", ha="right", fontsize=6.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_legend_group(ax: plt.Axes, title: str, mapping: dict[str, str], x: float, y: float, line_h: float = 0.044) -> float:
    ax.text(x, y, title, transform=ax.transAxes, ha="left", va="top", fontsize=5.1)
    y -= line_h * 0.82
    for label, color in mapping.items():
        ax.add_patch(Rectangle((x, y - 0.013), 0.035, 0.026, transform=ax.transAxes, facecolor=color, edgecolor="0.7", linewidth=0.25))
        ax.text(x + 0.045, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.6)
        y -= line_h
    return y - line_h * 0.22


def draw_legend_box(ax: plt.Axes, title: str, mapping: dict[str, str], fontsize: float = 5.0) -> None:
    ax.axis("off")
    ax.text(0.0, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize + 0.3)
    if not mapping:
        return
    step = min(0.20, 0.70 / max(len(mapping), 1))
    for i, (label, color) in enumerate(mapping.items()):
        y = 0.78 - i * step
        ax.add_patch(
            Rectangle(
                (0.0, y - 0.045),
                0.095,
                0.085,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="0.65",
                linewidth=0.25,
            )
        )
        ax.text(0.12, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=fontsize)


def draw_compact_legend_group(
    ax: plt.Axes,
    title: str,
    mapping: dict[str, str],
    x: float,
    y: float,
    line_h: float = 0.052,
    fontsize: float = 4.8,
) -> None:
    ax.text(x, y, title, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize + 0.4)
    y -= line_h * 0.82
    for label, color in mapping.items():
        ax.add_patch(
            Rectangle(
                (x, y - 0.018),
                0.052,
                0.036,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="0.65",
                linewidth=0.25,
            )
        )
        ax.text(x + 0.066, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=fontsize)
        y -= line_h


def draw_metric_color_legend(ax: plt.Axes) -> None:
    # SI, SDI and PRI use different hues but share the same 0-1 percentile scale.
    legend = ax.inset_axes([0.720, 1.030, 0.275, 0.285], transform=ax.transAxes)
    legend.axis("off")
    legend.text(0.0, 0.98, "Percentile rank (0-1)", ha="left", va="top", fontsize=5.5)
    gradient = np.linspace(0, 1, 128).reshape(1, -1)
    for i, (label, cmap_name) in enumerate([("SI", "Blues"), ("SDI", "Greens"), ("PRI", "Reds")]):
        y = 0.66 - i * 0.25
        bar = legend.inset_axes([0.20, y, 0.68, 0.115])
        bar.imshow(gradient, aspect="auto", cmap=mpl.colormaps[cmap_name], vmin=0, vmax=1)
        bar.set_xticks([])
        bar.set_yticks([])
        for spine in bar.spines.values():
            spine.set_visible(False)
        legend.text(0.0, y + 0.065, label, ha="left", va="center", fontsize=4.8)


def draw_panel_b(fig: plt.Figure, spec: gridspec.SubplotSpec, selected: pd.DataFrame, thresholds: dict[str, float]) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(
        5,
        2,
        subplot_spec=spec,
        width_ratios=[7.05, 1.45],
        height_ratios=[0.115, 0.115, 0.115, 0.115, 1.0],
        hspace=0.035,
        wspace=0.07,
    )
    ann_axes = [fig.add_subplot(sub[i, 0]) for i in range(4)]
    ax = fig.add_subplot(sub[4, 0])
    legend_ax = fig.add_subplot(sub[:, 1])

    subtype_labels = selected["Subtype"].map(subtype_group).tolist()
    draw_annotation_row(ann_axes[0], subtype_labels, SUBTYPE_COLORS, "Subtype")
    draw_annotation_row(ann_axes[1], selected["Period"].tolist(), PERIOD_COLORS, "Period")
    draw_annotation_row(ann_axes[2], selected["Continent"].tolist(), CONTINENT_COLORS, "Continent")
    draw_annotation_row(ann_axes[3], selected["Host_source_B"].tolist(), B_HOST_COLORS, "Host source")

    row_order = ["Very high", "High", "Medium", "Low"]
    body = np.full((len(row_order), len(selected)), np.nan)
    cat_to_row = {cat: i for i, cat in enumerate(row_order)}
    for i, (_, row) in enumerate(selected.iterrows()):
        body[cat_to_row[str(row["Risk_category"])]][i] = row["PRI"]

    cmap = mpl.cm.Reds.copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(np.ma.masked_invalid(body), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xticks(np.arange(len(selected)))
    tick_labels = ax.set_xticklabels(
        [strain_axis_label(s) for s in selected["strain_name"]],
        rotation=56,
        ha="right",
        va="top",
        rotation_mode="anchor",
        fontsize=4.9,
    )
    for tick in tick_labels:
        tick.set_y(tick.get_position()[1] - 0.02)
    ax.set_xticks(np.arange(-0.5, len(selected), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.28)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    legend_ax.axis("off")
    legend_ax.text(0.00, 0.98, "Risk bands", transform=legend_ax.transAxes, ha="left", va="top", fontsize=5.2)
    band_text = f"Q25={thresholds['Q25']:.2f}\nQ50={thresholds['Q50']:.2f}\nQ75={thresholds['Q75']:.2f}"
    legend_ax.text(0.00, 0.925, band_text, transform=legend_ax.transAxes, ha="left", va="top", fontsize=4.8, linespacing=1.10)

    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
    legend_ax.text(0.00, 0.765, "PRI", transform=legend_ax.transAxes, ha="left", va="center", fontsize=5.2)
    cax = legend_ax.inset_axes([0.16, 0.735, 0.74, 0.045])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=4.6, length=1.5, pad=1)
    cbar.set_ticks([0, 0.5, 1.0])

    draw_compact_legend_group(legend_ax, "Subtype", SUBTYPE_COLORS, 0.00, 0.650, line_h=0.050, fontsize=4.7)
    draw_compact_legend_group(legend_ax, "Period", PERIOD_COLORS, 0.50, 0.650, line_h=0.050, fontsize=4.7)
    continent_present = {k: v for k, v in CONTINENT_COLORS.items() if k in set(selected["Continent"])}
    draw_compact_legend_group(legend_ax, "Continent", continent_present, 0.00, 0.335, line_h=0.050, fontsize=4.7)
    host_present = {k: v for k, v in B_HOST_COLORS.items() if k in set(selected["Host_source_B"])}
    draw_compact_legend_group(legend_ax, "Host source", host_present, 0.50, 0.335, line_h=0.050, fontsize=4.7)

    panel_label(ann_axes[0], "B", x=-0.045, y=1.10)


def draw_panel_c(ax: plt.Axes, avian: pd.DataFrame) -> None:
    trend = avian.dropna(subset=["Year_int"]).copy()
    trend["Trend_group"] = np.where(trend["Subtype"] == "H5N1", "H5N1", "Other avian")
    major = trend[trend["Subtype"].isin(["H9N2", "H7N9"])]

    base = (
        trend.groupby(["Year_int", "Trend_group"])
        .agg(n=("strain_name", "size"), mean_PRI=("PRI", "mean"))
        .reset_index()
    )
    major_year = (
        major.groupby(["Year_int", "Subtype"])
        .agg(n=("strain_name", "size"), mean_PRI=("PRI", "mean"))
        .reset_index()
        .rename(columns={"Subtype": "Trend_group"})
    )
    plot = pd.concat([base, major_year], ignore_index=True)
    plot = plot[(plot["Year_int"] >= 2000) & (plot["Year_int"] <= 2025)]

    palette = {"H5N1": "#b2182b", "Other avian": "#8c8c8c", "H9N2": "#1b9e77", "H7N9": "#7570b3"}
    order = ["Other avian", "H9N2", "H7N9", "H5N1"]
    for group in order:
        d = plot[plot["Trend_group"] == group].sort_values("Year_int")
        if d.empty:
            continue
        lw = 2.2 if group == "H5N1" else 1.0
        alpha = 0.95 if group == "H5N1" else 0.76
        z = 4 if group == "H5N1" else 2
        ax.plot(d["Year_int"], d["mean_PRI"], color=palette[group], lw=lw, alpha=alpha, label=group, zorder=z)
        ax.scatter(
            d["Year_int"],
            d["mean_PRI"],
            s=np.clip(np.sqrt(d["n"]) * (4.0 if group == "H5N1" else 2.0), 9, 160),
            color=palette[group],
            edgecolor="white",
            linewidth=0.45,
            alpha=alpha,
            zorder=z + 1,
        )

    ax.axvspan(2020, 2025.5, color="#fee5d9", alpha=0.50, zorder=0)
    ax.text(2020.25, 0.93, "post-2020", color="#9b111e", fontsize=6.4)
    ax.set_xlim(2000, 2025.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Annual mean PRI")
    ax.set_xlabel("Collection year")
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6, columnspacing=0.8, fontsize=6.0)
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.65)
    panel_label(ax, "C", x=-0.12, y=1.05)
    despine(ax)


def draw_panel_d(ax: plt.Axes, avian: pd.DataFrame, stats: pd.DataFrame) -> None:
    host_order = [
        "Poultry",
        "Duck/waterfowl",
        "Goose/swan",
        "Gull/seabird",
        "Raptor",
        "Shorebird",
        "Other bird",
        "Mixed/unknown",
    ]
    top_subtypes = stats.head(13)["Subtype"].tolist()
    if "H5N1" in top_subtypes:
        top_subtypes = ["H5N1"] + [s for s in top_subtypes if s != "H5N1"]
    mat = pd.crosstab(
        avian[avian["Subtype"].isin(top_subtypes)]["Subtype"],
        avian[avian["Subtype"].isin(top_subtypes)]["Host_source"],
    ).reindex(index=top_subtypes, columns=host_order, fill_value=0)

    norm = np.log1p(mat) / np.log1p(mat.values.max())
    cmap = mpl.cm.OrRd.copy()
    cmap.set_under("#ffffff")
    im = ax.imshow(norm.values, aspect="auto", cmap=cmap, vmin=0.001, vmax=1)

    ax.set_xticks(np.arange(len(host_order)))
    ax.set_xticklabels(host_order, rotation=50, ha="right", rotation_mode="anchor", fontsize=6.0)
    ax.set_yticks(np.arange(len(top_subtypes)))
    ax.set_yticklabels(top_subtypes, fontsize=6.3)
    for i, subtype in enumerate(top_subtypes):
        if subtype == "H5N1":
            ax.get_yticklabels()[i].set_color("#9b111e")

    ax.set_xticks(np.arange(-0.5, len(host_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_subtypes), 1), minor=True)
    ax.grid(which="minor", color="#9e9e9e", linewidth=0.42)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.018)
    cbar.set_label("log count", fontsize=6.0)
    cbar.ax.tick_params(labelsize=5.3, length=2)
    panel_label(ax, "D", x=-0.13, y=1.05)


def draw_panel_e(ax: plt.Axes, avian: pd.DataFrame) -> None:
    group_order = ["H5N1", "H6N1", "H5N2", "H9N2", "H7N9", "Other avian"]
    plot_df = avian.copy()
    plot_df["Subtype_group"] = np.where(plot_df["Subtype"].isin(group_order[:-1]), plot_df["Subtype"], "Other avian")
    metric_cols = [("SI", "SI_percentile"), ("SDI", "SDI_percentile"), ("PRI", "PRI")]
    palette = {"SI": "#6baed6", "SDI": "#fdae6b", "PRI": "#cb6a6a"}
    offsets = [-0.23, 0.0, 0.23]

    rng = np.random.default_rng(13)
    for i, group in enumerate(group_order):
        group_df = plot_df[plot_df["Subtype_group"] == group]
        for j, (metric, col) in enumerate(metric_cols):
            values = group_df[col].dropna().to_numpy()
            if len(values) == 0:
                continue
            pos = i + offsets[j]
            violin = ax.violinplot(
                [values],
                positions=[pos],
                widths=0.19,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(palette[metric])
                body.set_edgecolor(palette[metric])
                body.set_alpha(0.28)
                body.set_linewidth(0.7)
            q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.plot([pos, pos], [q1, q3], color=palette[metric], linewidth=1.4, solid_capstyle="round", zorder=4)
            ax.plot([pos - 0.045, pos + 0.045], [med, med], color=palette[metric], linewidth=1.1, zorder=5)
            sample_n = min(len(values), 140)
            sample = rng.choice(values, sample_n, replace=False)
            jitter = rng.normal(0, 0.014, sample_n)
            ax.scatter(
                np.full(sample_n, pos) + jitter,
                sample,
                s=2.2,
                color=palette[metric],
                alpha=0.16,
                linewidth=0,
                zorder=3,
            )

    h5_median = avian.loc[avian["Subtype"] == "H5N1", "PRI"].median()
    ax.axhline(h5_median, color="#9b111e", linestyle="--", linewidth=0.85)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=palette[metric], edgecolor="none", alpha=0.55, label=metric)
        for metric, _ in metric_cols
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=3, handlelength=1.1, columnspacing=0.7, fontsize=5.8)
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(group_order, rotation=35, ha="right", rotation_mode="anchor", fontsize=6.0)
    ax.set_ylabel("Percentile rank")
    ax.set_ylim(0, 1.03)
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.65)
    panel_label(ax, "E", x=-0.13, y=1.05)
    despine(ax)


def copy_fig6_a_to_f() -> Path:
    if not FIG6_AF_ORIGINAL.exists():
        raise FileNotFoundError(FIG6_AF_ORIGINAL)
    out = OUT_DIR / "Figure6_A_to_F.png"
    shutil.copy2(FIG6_AF_ORIGINAL, out)
    return out


def write_stats(avian: pd.DataFrame, stats: pd.DataFrame, thresholds: dict[str, float], selected: pd.DataFrame) -> None:
    stats.to_csv(OUT_DIR / "Fig7_subtype_PRI_summary.csv", index=False)
    avian.to_csv(OUT_DIR / "Fig7_record_level_PRI_values.csv", index=False)
    selected_out = selected.copy()
    selected_out["panel_b_axis_label"] = selected_out["strain_name"].map(strain_axis_label)
    selected_out.to_csv(OUT_DIR / "Fig7_panelB_representative_strains.csv", index=False)
    annual = (
        avian.dropna(subset=["Year_int"])
        .groupby(["Year_int", "Subtype"])
        .agg(
            n=("strain_name", "size"),
            mean_SI=("SI_percentile", "mean"),
            mean_SDI=("SDI_percentile", "mean"),
            mean_PRI=("PRI", "mean"),
            median_PRI=("PRI", "median"),
        )
        .reset_index()
    )
    annual.to_csv(OUT_DIR / "Fig7_annual_subtype_PRI.csv", index=False)
    pd.DataFrame([thresholds]).to_csv(OUT_DIR / "Fig7_PRI_risk_band_quartiles.csv", index=False)


def main() -> None:
    configure_plotting()
    OUT_DIR.mkdir(exist_ok=True)
    avian = load_data()
    _, thresholds = risk_categories(avian["PRI"])
    avian["Risk_category"] = risk_categories(avian["PRI"])[0]
    stats = subtype_stats(avian)
    selected = select_representative_strains(avian, thresholds)

    fig = plt.figure(figsize=(9.0, 9.55))
    outer = gridspec.GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.35, 3.15, 0.48, 2.75],
        hspace=0.28,
    )

    draw_panel_a(fig.add_subplot(outer[0]), stats)
    draw_panel_b(fig, outer[1], selected, thresholds)

    bottom = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[3],
        width_ratios=[1.20, 1.20, 0.95],
        wspace=0.45,
    )
    draw_panel_c(fig.add_subplot(bottom[0]), avian)
    draw_panel_d(fig.add_subplot(bottom[1]), avian, stats)
    draw_panel_e(fig.add_subplot(bottom[2]), avian)

    png = OUT_DIR / "Figure7_H5N1_PRI.png"
    svg = OUT_DIR / "Figure7_H5N1_PRI.svg"
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    fig6 = copy_fig6_a_to_f()
    write_stats(avian, stats, thresholds, selected)

    h5 = stats[stats["Subtype"] == "H5N1"].iloc[0]
    post2020 = subtype_stats(avian[avian["Year_int"] >= 2020])
    h5_post = post2020[post2020["Subtype"] == "H5N1"].iloc[0]
    h7n9 = avian[avian["Subtype"] == "H7N9"]
    print(f"Figure 7 PNG: {png}")
    print(f"Figure 7 SVG: {svg}")
    print(f"Figure 6 A-F original copy: {fig6}")
    print(
        "H5N1 all-years: "
        f"n={int(h5['n'])}, mean PRI={h5['mean_PRI']:.4f}, median PRI={h5['median_PRI']:.4f}, "
        f"mean SI={h5['mean_SI']:.4f}, mean SDI={h5['mean_SDI']:.4f}"
    )
    print(
        "H5N1 post-2020: "
        f"n={int(h5_post['n'])}, mean PRI={h5_post['mean_PRI']:.4f}, median PRI={h5_post['median_PRI']:.4f}"
    )
    if not h7n9.empty:
        print(f"H7N9 eligible avian records: n={len(h7n9)}, last year={int(h7n9['Year_int'].max())}")


if __name__ == "__main__":
    main()
