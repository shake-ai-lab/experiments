"""Q学習で迷路を解く。
(1) Q値(矢印マップ)がゴールから逆向きに育っていく様子をGIF/mp4に
(2) エピソードごとの到達ステップ数(学習曲線)
(3) 崖歩き(cliff walking)でQ学習vsSARSAの「性格」の違いを見る"""
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})
rng = np.random.default_rng(0)

# ---- 迷路定義 (0=通路, 1=壁) ----
maze = np.array(
    [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
    ]
)
H, W = maze.shape
START, GOAL = (0, 0), (7, 7)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
GAMMA, ALPHA, EPS = 0.95, 0.2, 0.1


def step(s, a):
    ny, nx = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    if not (0 <= ny < H and 0 <= nx < W) or maze[ny, nx] == 1:
        return s, 0.0, False  # 壁: その場に留まる
    if (ny, nx) == GOAL:
        return (ny, nx), 1.0, True
    return (ny, nx), 0.0, False


Q = np.zeros((H, W, 4))
EPISODES = 400
snap_at = {0, 1, 3, 7, 15, 30, 60, 120, 250, 400}
snaps, steps_hist = {}, []
for ep in range(1, EPISODES + 1):
    s, done, n = START, False, 0
    while not done and n < 500:
        if rng.random() < EPS:
            a = int(rng.integers(4))
        else:
            qv = Q[s[0], s[1]]
            a = int(rng.choice(np.flatnonzero(qv == qv.max())))  # 同値はランダムに選ぶ
        s2, r, done = step(s, a)
        target = r + (0 if done else GAMMA * Q[s2[0], s2[1]].max())
        Q[s[0], s[1], a] += ALPHA * (target - Q[s[0], s[1], a])
        s = s2
        n += 1
    steps_hist.append(n)
    if ep in snap_at:
        snaps[ep] = Q.copy()
snaps[0] = np.zeros_like(Q)
print(f"最初のエピソード: {steps_hist[0]}ステップ, 最後: {steps_hist[-1]}ステップ")

# ---- 矢印マップ描画 ----
def draw_q(ax, Qs, title):
    ax.imshow(np.where(maze == 1, 0.35, 1.0), cmap="gray", vmin=0, vmax=1)
    v = Qs.max(axis=2)
    for y in range(H):
        for x in range(W):
            if maze[y, x] == 1 or (y, x) == GOAL:
                continue
            if v[y, x] < 1e-6:
                continue
            a = int(np.argmax(Qs[y, x]))
            dy, dx = ACTIONS[a]
            ax.arrow(x, y, dx * 0.3, dy * 0.3, head_width=0.16,
                     color=plt.cm.plasma(min(1.0, v[y, x] * 1.2)), lw=1.5)
    ax.scatter(*reversed(START), marker="s", s=90, color="#2e6ee4", zorder=3)
    ax.scatter(*reversed(GOAL), marker="*", s=240, color="#e4572e", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)


order = sorted(snaps.keys())
fig, ax = plt.subplots(figsize=(5.6, 5.8))


def update(fi):
    ax.clear()
    ep = order[fi]
    draw_q(ax, snaps[ep], f"エピソード {ep}: 価値(矢印)が出口から染みてくる")
    return []


ani = animation.FuncAnimation(fig, update, frames=len(order), interval=700)
ani.save("q_arrows.gif", writer="pillow", fps=1.4, dpi=85)
ani.save("q_arrows.mp4", writer=animation.FFMpegWriter(fps=1.4, bitrate=1600), dpi=100)
print("saved q_arrows.gif/mp4")
plt.close(fig)

# ---- 学習曲線 ----
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(steps_hist, color="#e4572e", lw=1.2, alpha=0.5)
ax.plot(np.convolve(steps_hist, np.ones(10) / 10, mode="valid"), color="#e4572e", lw=2.4)
ax.annotate(f"最初は{steps_hist[0]}ステップ迷子", xy=(0, steps_hist[0]),
            xytext=(60, steps_hist[0] * 0.9), fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=1.5))
best = min(steps_hist[-50:])
ax.annotate(f"終盤は最短{best}ステップ", xy=(len(steps_hist) - 20, best),
            xytext=(230, 120), fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.5))
ax.set_xlabel("エピソード")
ax.set_ylabel("ゴールまでのステップ数")
ax.set_title("400回の試行で、迷子が最短ルートになる")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig2_learning.png", dpi=110, bbox_inches="tight")
print("saved fig2_learning.png")

# ---- 崖歩き: Q学習 vs SARSA ----
CH, CW = 4, 10
C_START, C_GOAL = (3, 0), (3, 9)


def cliff_step(s, a):
    ny = min(max(s[0] + ACTIONS[a][0], 0), CH - 1)
    nx = min(max(s[1] + ACTIONS[a][1], 0), CW - 1)
    if ny == 3 and 1 <= nx <= 8:
        return C_START, -100.0, False  # 崖に落ちてスタートへ
    if (ny, nx) == C_GOAL:
        return (ny, nx), -1.0, True
    return (ny, nx), -1.0, False


def run_cliff(algo, episodes=600, seed=0):
    r = np.random.default_rng(seed)
    Q = np.zeros((CH, CW, 4))
    rewards = []
    for ep in range(episodes):
        s = C_START
        qv = Q[s[0], s[1]]
        a = int(r.integers(4)) if r.random() < EPS else int(r.choice(np.flatnonzero(qv == qv.max())))
        done, total = False, 0.0
        for _ in range(400):
            s2, rew, done = cliff_step(s, a)
            total += rew
            qv2 = Q[s2[0], s2[1]]
            a2 = int(r.integers(4)) if r.random() < EPS else int(r.choice(np.flatnonzero(qv2 == qv2.max())))
            if algo == "q":
                target = rew + (0 if done else GAMMA * Q[s2[0], s2[1]].max())
            else:  # sarsa: 実際に取る次の行動で評価
                target = rew + (0 if done else GAMMA * Q[s2[0], s2[1], a2])
            Q[s[0], s[1], a] += ALPHA * (target - Q[s[0], s[1], a])
            s, a = s2, a2
            if done:
                break
        rewards.append(total)
    # greedy経路を抽出
    path, s = [C_START], C_START
    for _ in range(60):
        a = int(np.argmax(Q[s[0], s[1]]))
        s, _, done = cliff_step(s, a)
        path.append(s)
        if done:
            break
    return Q, np.array(rewards), path


Qq, rew_q, path_q = run_cliff("q")
Qs_, rew_s, path_s = run_cliff("sarsa")
print(f"平均報酬(終盤100ep): Q学習 {rew_q[-100:].mean():.1f}, SARSA {rew_s[-100:].mean():.1f}")
print(f"経路長: Q学習 {len(path_q)-1}, SARSA {len(path_s)-1}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))
grid_img = np.ones((CH, CW))
grid_img[3, 1:9] = 0.25
axes[0].imshow(grid_img, cmap="gray", vmin=0, vmax=1)
for path, col, name in [(path_q, "#e4572e", "Q学習"), (path_s, "#2e6ee4", "SARSA")]:
    py, px = zip(*path)
    axes[0].plot(px, py, "-o", color=col, ms=5, lw=2.4, label=name)
axes[0].text(4.5, 3.0, "崖 (-100)", ha="center", color="w", fontsize=11)
axes[0].annotate("Q学習: 崖っぷちの最短路", xy=(4, 2), xytext=(1.2, 0.4), color="#e4572e",
                 fontsize=11, arrowprops=dict(arrowstyle="->", color="#e4572e", lw=1.5))
axes[0].annotate("SARSA: 安全な遠回り", xy=(4, 0), xytext=(5.7, 0.7), color="#2e6ee4",
                 fontsize=11, arrowprops=dict(arrowstyle="->", color="#2e6ee4", lw=1.5))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("学習後のルート: 攻めのQ学習、守りのSARSA")
axes[0].legend(loc="lower left", fontsize=9)
w = 20
axes[1].plot(np.convolve(rew_q, np.ones(w) / w, mode="valid"), color="#e4572e", label="Q学習")
axes[1].plot(np.convolve(rew_s, np.ones(w) / w, mode="valid"), color="#2e6ee4", label="SARSA")
axes[1].set_ylim(-120, 0)
axes[1].set_xlabel("エピソード")
axes[1].set_ylabel("1エピソードの合計報酬 (20ep移動平均)")
axes[1].set_title("探索中の成績はSARSAが上 (Q学習は崖に落ち続ける)")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_cliff.png", dpi=110, bbox_inches="tight")
print("saved fig3_cliff.png")
