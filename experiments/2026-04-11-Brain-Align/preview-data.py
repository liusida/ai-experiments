# %% Main
import pandas as pd
import stonesoup

parquet_path = (
    stonesoup.repo_root()
    / "references"
    / "BrainAlign"
    / "data"
    / "intermediate_results"
    / "language_all.parquet"
)
df = pd.read_parquet(parquet_path)

print(df.shape)
print(df.columns)
# %% Head
pd.set_option("display.max_columns", None)  # or a number like 30
stonesoup.display(df.head(2))

# %% Show unique values for each column
for column in df.columns:
    print(f"{column} ({df[column].nunique()}): {df[column].unique()}")

# %% Top score rows per subject (all cka_* metrics)
stonesoup.html()

cka_mask = df["metric"].str.startswith("cka_")
all_subjects = df["tgt_model"].unique()
for subject in all_subjects:
    print(f"Subject: {subject}")
    cka_rows = df[cka_mask & (df["tgt_model"] == subject)]
    stonesoup.display(cka_rows.nlargest(3, "score"))

# get unique tgt_model values from this subset (all cka_*)
stonesoup.display(df[cka_mask].nlargest(100, "score")["tgt_model"].unique())

# %% Score histograms by metric
import math

import matplotlib.pyplot as plt
import stonesoup
from stonesoup.experiment import configure_matplotlib_agg

configure_matplotlib_agg()

# Filled step histograms avoid thin vertical bar edges (moiré / striping) when bins are dense.
_HIST_KW = dict(
    bins=40,
    range=(-0.5, 2.0),
    histtype="stepfilled",
    color="steelblue",
    linewidth=0,
    alpha=0.95,
)

metrics_sorted = sorted(df["metric"].unique())
ncols = 5
nrows = math.ceil(len(metrics_sorted) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows), squeeze=False)
for i, metric in enumerate(metrics_sorted):
    r, c = divmod(i, ncols)
    ax = axes[r][c]
    s = df.loc[df["metric"] == metric, "score"]
    ax.hist(s, **_HIST_KW)
    ax.set_title(metric, fontsize=9)
    ax.set_xlabel("score", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_xlim(-0.5, 2)
for j in range(len(metrics_sorted), nrows * ncols):
    hr, hc = divmod(j, ncols)
    axes[hr][hc].set_visible(False)
fig.suptitle("language_all.parquet: score distribution per metric", fontsize=12, y=1.002)
fig.tight_layout()
stonesoup.show(fig, basename="language_all_score_hists_by_metric", dpi=144)

# %% cka_20 score histograms by src_model (Qwen*, yeo7=Limbic)
import math

import matplotlib.pyplot as plt
import stonesoup
from stonesoup.experiment import configure_matplotlib_agg

configure_matplotlib_agg()

_HIST_KW = dict(
    bins=40,
    range=(-0.5, 2.0),
    histtype="stepfilled",
    color="steelblue",
    linewidth=0,
    alpha=0.95,
)

PARAM_COL = "#Params (B)"
cka20 = df[
    (df["metric"] == "cka_20")
    & df["src_model"].str.startswith("Qwen")
    & (df["yeo7"] == "Limbic")
]
# Same param count for every row of a model — order panels by size (billions).
models_sorted = (
    cka20.groupby("src_model", sort=False)[PARAM_COL]
    .median()
    .sort_values(na_position="last")
    .index.tolist()
)
ncols = 5
nrows = math.ceil(len(models_sorted) / ncols)
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows), squeeze=False)
for i, src in enumerate(models_sorted):
    r, c = divmod(i, ncols)
    ax = axes2[r][c]
    s = cka20.loc[cka20["src_model"] == src, "score"]
    ax.hist(s, **_HIST_KW)
    ax.set_title(src, fontsize=7)
    ax.set_xlabel("score", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_xlim(-0.5, 2)
for j in range(len(models_sorted), nrows * ncols):
    hr, hc = divmod(j, ncols)
    axes2[hr][hc].set_visible(False)
fig2.suptitle(
    "language_all: cka_20 score by src_model (Qwen*, yeo7=Limbic; sorted by #Params (B))",
    fontsize=12,
    y=1.002,
)
fig2.tight_layout()
stonesoup.show(fig2, basename="language_all_cka_20_hists_by_src_model_qwen_limbic", dpi=144)

# %% Score stats by LM layer (src_feature) — Qwen, cka_20, yeo7=Limbic
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 12)
stonesoup.display(cka20.groupby("src_feature", sort=True)["score"].describe().round(6))

# %% Mean ± std of score vs LM layer (src_feature)
import matplotlib.pyplot as plt
import stonesoup
from stonesoup.experiment import configure_matplotlib_agg

configure_matplotlib_agg()

_layer = cka20.groupby("src_feature", sort=True)["score"].agg(["mean", "std"])
_x = _layer.index.to_numpy()
_mean = _layer["mean"].to_numpy()
_std = _layer["std"].to_numpy()

fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.errorbar(
    _x,
    _mean,
    yerr=_std,
    fmt="-o",
    color="steelblue",
    ecolor="steelblue",
    elinewidth=1,
    capsize=3,
    markersize=4,
    alpha=0.9,
)
ax3.set_xlabel("src_feature (layer index)")
ax3.set_ylabel("score")
ax3.set_title("cka_20: mean ± std vs layer (Qwen*, yeo7=Limbic)")
ax3.set_xlim(_x.min() - 0.5, _x.max() + 0.5)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
stonesoup.show(fig3, basename="language_all_cka_20_mean_std_vs_layer_qwen_limbic", dpi=144)