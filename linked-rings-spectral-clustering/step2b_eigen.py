"""固有ベクトル図の改善版: グラフが分断されている場合(noise=0.05)と、
ノイズでリング間がつながってしまった場合(noise=0.18)のFiedlerベクトルを比較"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"
R, N = 1.0, 400


def make_rings(noise, seed=0):
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, N)
    ph = r.uniform(0, 2 * np.pi, N)
    ringA = np.c_[R * np.cos(th), R * np.sin(th), np.zeros(N)]
    ringB = np.c_[R + R * np.cos(ph), np.zeros(N), R * np.sin(ph)]
    X = np.vstack([ringA, ringB]) + r.normal(0, noise, (2 * N, 3))
    y = np.r_[np.zeros(N, int), np.ones(N, int)]
    return X, y


fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
for row, nz in enumerate([0.05, 0.18]):
    X, y = make_rings(nz, seed=0)
    A = kneighbors_graph(X, n_neighbors=10, mode="connectivity")
    A = 0.5 * (A + A.T).toarray()
    ncomp = connected_components(A > 0)[0]
    L = np.diag(A.sum(1)) - A
    w, v = eigh(L)
    fied = v[:, 1]
    # 符号を揃える(リングAが負側になるように)
    if fied[y == 0].mean() > 0:
        fied = -fied
    n_cross = int((A[:N, N:] > 0).sum())
    print(f"noise {nz}: 連結成分 {ncomp}, リング間エッジ {n_cross}, λ2={w[1]:.5f}")
    axes[row, 0].plot(np.sort(fied), lw=1.5, color="#333")
    axes[row, 0].axhline(0, color="#e4572e", ls="--", lw=1)
    axes[row, 0].set_title(
        f"ノイズ{nz}: Fiedlerベクトルを昇順に並べる (λ₂={w[1]:.4f}, リング間エッジ{n_cross}本)",
        fontsize=10,
    )
    axes[row, 1].scatter(np.arange(len(fied)), fied, c=y, cmap="coolwarm", s=4)
    axes[row, 1].axhline(0, color="#888", ls="--", lw=1)
    axes[row, 1].set_title("点ごとのFiedler値 (色=真のリング)", fontsize=10)
    axes[row, 1].set_xlabel("点の番号 (前半400=リングA, 後半400=リングB)")
fig.suptitle(
    "スペクトラルの中身: グラフが完全分断なら固有ベクトルは所属表そのもの(上)。\nノイズでつながっても、「弱いつながり」を検出して符号で分離する(下)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("fig3_eigen.png", dpi=110, bbox_inches="tight")
print("saved fig3_eigen.png")
