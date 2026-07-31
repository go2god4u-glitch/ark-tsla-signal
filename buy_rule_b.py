"""매수 규칙 B (극단 낙폭에서 아크 문턱 완화) — 시험하고 기각한 기록.

결론부터: **채택하지 않았다.** 실행하면 왜 기각했는지가 그대로 나온다.

동기는 진짜였다. 규칙 A(아크 상위 10% + 낙폭 -30%)만으로는 낙폭이 -48~-54%
인데 신호가 0건인 구간이 있다(2022-06~07, 2025-03~04). 값이 반토막인데 못 산다.
그래서 '낙폭이 아주 깊으면 아크 문턱을 낮춘다' 를 규칙 A 와 병렬로 붙여 봤다.

처음 짠 격자는 통과했다(25칸 중 17칸). 그런데 그 격자의 셀이 **평가액**이었다.
평가액 = 건수 x 100 x (1+평균) 이므로, 신호를 늘리는 규칙은 평균이 떨어져도
평가액이 올라간다. 통과할 수밖에 없는 시험이었다. (CLAUDE.md 22번)

셀을 포지션당 평균으로 바꾸니 20칸 중 19칸이 A 단독보다 못했다.

이 파일이 지키는 두 가지:

  1. 매도는 다시 짜지 않는다. `position_tracker.replay()` 를 그대로 부른다.
     라이브와 같은 함수여야 세는 방식이 갈라지지 않는다.

  2. 문턱은 `build()` 가 자르기 **전** 시리즈에서 낸다.
     자른 뒤 다시 `expanding(52)` 를 걸면 확장창이 재시작해 문턱이 달라지고
     결론이 통째로 뒤집힌다. 실제로 한 번 뒤집혔다. (CLAUDE.md 23번)
     그래서 맨 먼저 규칙 A 를 재현해 16건이 나오는지 자기 검사한다.
"""

import json
import os

import numpy as np
import pandas as pd

import signal_check as sc
from position_tracker import replay, rsi14, BIG_SELL

BASE = os.path.dirname(os.path.abspath(__file__))
DD_GRID = [-35.0, -40.0, -45.0, -50.0]
Q_GRID = [0.50, 0.60, 0.70, 0.80]


def main() -> None:
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    V, RSI = px.values, rsi14(px).values
    H = sc.build_daily(px)["shares"].reindex(px.index).ffill().values
    DD = ((px / px.rolling(252, min_periods=60).max() - 1) * 100).values
    w = sc.build(px)

    # 문턱은 build 가 자르기 전 시리즈에서 낸다. 자른 뒤 다시 내면 창이 재시작한다.
    wide = sc.build_daily(px).attrs["wide"]
    wk = wide.resample("W-FRI").last()
    full = (wk.diff().sum(axis=1, min_count=1)
            / wk.shift(1).sum(axis=1, min_count=1) * 100)

    def thr(q):
        return full.expanding(sc.MIN_HIST).quantile(q).shift(1).reindex(w.index)

    big = {px.index.searchsorted(t, side="right")
           for t in w.index[(w["netpct"] <= BIG_SELL).fillna(False)]}
    big = {x for x in big if x < len(px)}

    def stat(mask):
        """라이브와 같은 replay() 로 재생한다. 연속 신호를 접지 않는다."""
        locs = {px.index.searchsorted(t, side="right") for t in w[mask].index}
        done, live = replay(V, RSI, {x for x in locs if x < len(px)}, H, DD, big)
        return np.array([d["ret"] for d in done] + [p["ret"] for p in live])

    # --- 자기 검사: 이 스크립트가 기존 규칙 A 를 재현하는가 ---
    repro = ((w["netpct"] >= thr(sc.QUANT)) & (w["dd"] <= sc.DD_FILTER)).fillna(False)
    n_repro, n_live = int(repro.sum()), int(w["sig"].sum())
    ok = n_repro == n_live
    print(f"[자기 검사] 규칙 A 재현 {n_repro}건 vs 라이브 {n_live}건  "
          f"{'통과' if ok else '실패 — 아래 숫자를 믿지 마라'}")
    if not ok:
        raise SystemExit("문턱 계산 위치가 파이프라인과 다르다. CLAUDE.md 23번.")

    base = stat(w["sig"])
    print(f"\n규칙 A 단독: {len(base)}건  평균 {base.mean()*100:+.1f}%  "
          f"중앙값 {np.median(base)*100:+.1f}%  승률 {(base>0).mean()*100:.0f}%  "
          f"최악 {base.min()*100:+.1f}%  평가 {sum(100*(1+x) for x in base):.0f}")

    print("\n" + "=" * 78)
    print("규칙 B 격자 — 셀 = 평균 / 중앙값 / 건수")
    print("평가액으로 고르면 안 된다. 건수가 늘면 평균이 떨어져도 커진다.")
    print("=" * 78)
    print(f"{'낙폭':<9s}" + "".join(f"{f'상위{(1-q)*100:.0f}%':>18s}" for q in Q_GRID))
    beat = 0
    for dd_b in DD_GRID:
        row = ""
        for q in Q_GRID:
            sig_b = ((w["netpct"] >= thr(q)) & (w["dd"] <= dd_b)).fillna(False)
            a = stat(w["sig"] | sig_b)
            win = a.mean() > base.mean()
            beat += win
            row += f"{a.mean()*100:>+7.1f}{'*' if win else ' '}/{np.median(a)*100:>+6.1f}/{len(a):>3d}"
        print(f"{f'<={dd_b:.0f}%':<9s}{row}")
    print(f"\n  * = A 단독 평균({base.mean()*100:+.1f}%) 초과 — "
          f"{len(DD_GRID)*len(Q_GRID)}칸 중 {beat}칸")
    print("  완화 폭이 클수록(왼쪽·위) 단조롭게 나빠진다 = 방향 자체가 틀렸다.")

    print("\n" + "=" * 78)
    print("채택 후보였던 값(-45%, 상위 40%) 상세")
    print("=" * 78)
    sig_b = ((w["netpct"] >= thr(0.60)) & (w["dd"] <= -45.0)).fillna(False)
    for label, mask in (("A 단독", w["sig"]), ("A | B", w["sig"] | sig_b),
                        ("  └ B 추가분", sig_b & ~w["sig"])):
        a = stat(mask)
        print(f"{label:<14s}{len(a):>4d}건  평균 {a.mean()*100:>+7.1f}%  "
              f"중앙값 {np.median(a)*100:>+7.1f}%  승률 {(a>0).mean()*100:>3.0f}%  "
              f"최악 {a.min()*100:>+7.1f}%  투입 {len(a)*100} -> "
              f"평가 {sum(100*(1+x) for x in a):.0f}")

    print(f"\n  B 문턱(상위 40%) 중앙값 {thr(0.60).median():.2f}%  "
          f"<- '아크가 조금이라도 사면' 이라는 뜻이다")
    print(f"  A 문턱(상위 10%) 중앙값 {thr(sc.QUANT).median():.2f}%")

    print("\n  B 가 만든 신호가 같은 국면에서 몇 주 연속으로 뜨는가:")
    only_b = w.index[(sig_b & ~w["sig"]).values]
    runs = []
    for t in only_b:
        if runs and (t - runs[-1][-1]).days <= 21:
            runs[-1].append(t)
        else:
            runs.append([t])
    for g in runs:
        print(f"    {g[0]:%Y-%m-%d} ~ {g[-1]:%Y-%m-%d}  {len(g)}주"
              + ("   <- 같은 국면을 여러 번 산다" if len(g) >= 3 else ""))

    print("\n" + "=" * 78)
    print("결론: 기각. 규칙 A 단독 유지. 상세는 docs/FINDINGS.md 13절.")
    print("=" * 78)


if __name__ == "__main__":
    main()
