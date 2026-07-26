"""Datasaurus Dozen: 13データセットの統計量が本当に同一かを検証し、全形状のグリッド図を作る"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Meiryo"

df = pd.read_csv("datasaurus_dozen.tsv", sep="\t")
names = df["dataset"].unique()
print(f"datasets: {len(names)}: {list(names)}")

rows = []
for name in names:
    d = df[df["dataset"] == name]
    rows.append(
        dict(
            dataset=name,
            n=len(d),
            mean_x=d["x"].mean(),
            mean_y=d["y"].mean(),
            sd_x=d["x"].std(),
            sd_y=d["y"].std(),
            corr=d["x"].corr(d["y"]),
        )
    )
stats = pd.DataFrame(rows)
print(stats.round(2).to_string(index=False))
print()
print("小数点2桁で丸めたときのユニーク値の数:")
for col in ["mean_x", "mean_y", "sd_x", "sd_y", "corr"]:
    uniq = stats[col].round(2).nunique()
    print(f"  {col}: {uniq} (値: {stats[col].round(2).unique()})")

# 13形状のグリッド図
order = ["dino"] + sorted([n for n in names if n != "dino"])
fig, axes = plt.subplots(3, 5, figsize=(13, 8))
for ax, name in zip(axes.flat, order):
    d = df[df["dataset"] == name]
    ax.scatter(d["x"], d["y"], s=6, color="#e4572e" if name == "dino" else "#333")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=10)
for ax in axes.flat[len(order) :]:
    ax.axis("off")
fig.suptitle(
    "13個すべて: 平均(54.3, 47.8)・標準偏差(16.8, 26.9)・相関−0.06 (小数2桁まで同一)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("fig1_dozen_grid.png", dpi=110, bbox_inches="tight")
print("saved fig1_dozen_grid.png")
