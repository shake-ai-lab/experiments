"""記事に載せた数字を summary.txt にまとめる。実験を再実行せずに結果だけ見たい人向け。"""

import json

r = json.load(open("results.json", encoding="utf-8"))
a = json.load(open("analysis.json", encoding="utf-8"))
p = json.load(open("periods.json", encoding="utf-8"))

L = []
L.append("=== じゃんけん: 刻み幅ごとの200,000手 ===")
L.append(
    "%-8s %12s %12s %10s %12s %14s"
    % ("刻み幅", "均衡との隔たり(開始)", "同(終了)", "倍率", "最終の最大確率", "時間平均")
)
for e in ["0.2", "0.05", "0.01", "0.002"]:
    v = r["rps"][e]
    av = v["timeavg_vector_end"]
    L.append(
        "%-8s %12.4f %12.3f %9.1f倍 %12.4f  (%.4f, %.4f, %.4f)"
        % (e, v["energy_start"], v["energy_end"], v["energy_ratio"],
           v["iterate_max_prob_end"], av[0], av[1], av[2])
    )

L.append("")
L.append("=== 読まれたときの損 (exploitability、ナッシュ均衡なら0、最大2) ===")
L.append("%-8s %16s %16s" % ("刻み幅", "いま出している手", "ここまでの平均"))
for e in ["0.2", "0.05", "0.01", "0.002"]:
    L.append(
        "%-8s %16.4f %16.5f"
        % (e, a[e]["exploitability_iterate_last"], a[e]["exploitability_timeavg_last"])
    )
L.append("ナッシュ均衡そのもの: %.1f (物差しの校正)" % a["nash_itself"]["exploitability"])

L.append("")
L.append("=== 1周にかかる手数 ===")
L.append("%-8s %8s %10s %10s %8s" % ("刻み幅", "周回数", "最初の周期", "最後の周期", "倍率"))
for e in ["0.2", "0.05", "0.01", "0.002"]:
    v = p[e]
    L.append(
        "%-8s %8d %10d %10d %7.2f倍"
        % (e, v["n_laps"], v["first_period_steps"], v["last_period_steps"], v["period_growth"])
    )

rl = r["replicator_limit"]
L.append("")
L.append("=== レプリケータ極限 (刻み幅 %g、%s手) ===" % (rl["eta"], f"{rl['steps']:,}"))
L.append("保存量 x1*x2*x3: %.5f -> %.5f (%.1f%% のずれ)"
         % (rl["invariant_start"], rl["invariant_end"], (rl["invariant_drift_ratio"] - 1) * 100))
L.append("確率の範囲: %.3f 〜 %.3f" % (rl["min_prob_over_run"], rl["max_prob_over_run"]))
L.append("読まれたときの損: %.3f 〜 %.3f (平均 %.3f) — 一度も0にならない"
         % (a["replicator"]["exploitability_min"], a["replicator"]["exploitability_max"],
            a["replicator"]["exploitability_mean"]))

L.append("")
L.append("=== 囚人のジレンマ (刻み幅 %g) ===" % r["pd"]["eta"])
L.append("裏切り確率: %.2f (開始) -> %.4f (100手) -> %.6f (500手)"
         % (r["pd"]["start_defect_prob"], r["pd"]["defect_prob_at_100"], r["pd"]["defect_prob_at_500"]))
L.append("99%%に達した手数: %d" % r["pd"]["steps_to_99pct_defect"])

L.append("")
L.append("=== 協調ゲーム (刻み幅 %g、初期値 %d×%d 通り) ==="
         % (r["coord"]["eta"], r["coord"]["grid"], r["coord"]["grid"]))
L.append("利得2の均衡へ: %.1f%%" % (r["coord"]["share_converged_to_payoff2"] * 100))
L.append("利得1の均衡へ: %.1f%%" % ((1 - r["coord"]["share_converged_to_payoff2"]) * 100))
L.append("混合戦略の均衡: 両者とも p = 1/3")

open("summary.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
# 標準出力には流さない。Windows のコンソールは cp932 なので、全角ダッシュで落ちる。
print("wrote summary.txt (%d lines)" % len(L))
