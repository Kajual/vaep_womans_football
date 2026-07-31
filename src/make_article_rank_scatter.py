
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"]
})


PARQUET = Path("outputs/tables/player_rankings.parquet")
OUT     = Path("outputs/figures/rank_scatter.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

FORWARD_IDS = {10099, 10229, 50146, 10203, 46880, 15214}
DEEPMID_IDS = {26094, 46545, 10228, 10388, 10654, 10405}


df = pd.read_parquet(PARQUET)
wide = (df[df["model_id"].isin(["model_A", "model_C"])]
        .pivot(index="player_id", columns="model_id", values="rank_total")
        .dropna()
        .rename(columns={"model_A": "rank_A", "model_C": "rank_C"})
        .reset_index())

n = len(wide)
print(f"plotting {n} players")


fig, ax = plt.subplots(figsize=(6.0, 6.0))
ax.plot([1, n], [1, n], ls="--", color="#888888", lw=1.0, zorder=1)

others = wide[~wide["player_id"].isin(FORWARD_IDS | DEEPMID_IDS)]
ax.scatter(others["rank_A"], others["rank_C"], s=22,
           facecolor="#bbbbbb", edgecolor="white", lw=0.4, zorder=2,
           label="Other players")

SURNAMES = {
    10099: "T.Wullaert",      10229: "E.Blackstenius",  50146: "E.González",
    10203: "C.Girelli",       46880: "G.Hoffmann",      15214: "E.Pajor",
    26094: "J.Zigiotti-Olme", 46545: "C.Wamser",        10228: "M.Giugliano",
    10388: "L.Deloose",       10654: "J.Groenen",       10405: "L.Wälti",
}

fwd = wide[wide["player_id"].isin(FORWARD_IDS)]
ax.scatter(fwd["rank_A"], fwd["rank_C"], s=60, marker="o",
           facecolor="#c0392b", edgecolor="white", lw=0.8, zorder=4,
           label="Forwards (named risers)")

for _, row in fwd.iterrows():
    ax.annotate(SURNAMES[row["player_id"]],
                (row["rank_A"], row["rank_C"]),
                textcoords="offset points", xytext=(-8, 0),
                fontsize=6.5, ha="right", va="center",
                color="#7d1f15", zorder=5)

dm = wide[wide["player_id"].isin(DEEPMID_IDS)]
ax.scatter(dm["rank_A"], dm["rank_C"], s=60, marker="s",
           facecolor="#2b6cb0", edgecolor="white", lw=0.8, zorder=4,
           label="Deep mids / full-backs (named fallers)")

for _, row in dm.iterrows():
    ax.annotate(SURNAMES[row["player_id"]],
                (row["rank_A"], row["rank_C"]),
                textcoords="offset points", xytext=(8, 0),
                fontsize=6.5, ha="left", va="center",
                color="#1a4470", zorder=5)

ax.set_xlabel("Rank under E-VAEP")
ax.set_ylabel("Rank under PS-VAEP")
ax.set_xlim(0, n + 2)
ax.set_ylim(0, n + 2)
ax.set_aspect("equal")
ax.legend(loc="upper left", frameon=True, fontsize=7.5, framealpha=0.95, borderpad=0.4, labelspacing=0.3, handletextpad=0.4, markerscale=0.8)
ax.text(0.98, 0.02, "identity (no rank change)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5, color="#666666", style="italic")
ax.grid(True, color="#eeeeee", lw=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)