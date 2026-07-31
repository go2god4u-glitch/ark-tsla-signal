"""진입 방식 비교 — 신호가 떴을 때 한 번에 넣는가, 나눠 넣는가.

이것은 자본 배분 조언이 아니다. 같은 신호에 '체결 방식'만 바꿨을 때
과거에 어떤 차이가 났는지를 계산한 것이다.

비교 방식 (전부 진입 1회차는 신호 다음 거래일 종가):
  일괄     100% 를 신호 직후 한 번에
  2분할    50% + 50% (2주 간격)
  3분할    1/3 씩 (2주 간격)
  4분할    25% 씩 (1주 간격)
  5분할    20% 씩 (2주 간격)
  10주분할  10% 씩 매주

측정:
  - 평균 매입단가(가중) 대비 '첫 진입일로부터 6개월 뒤' 종가 수익률
  - 즉 분할해도 매도 시점은 동일하다. 늦게 넣은 분은 보유기간이 짧다.
  - 최악 사례와 승률을 함께 본다. 분할의 값어치는 평균이 아니라 최악에서 나온다.

한계: 신호 사건이 8개뿐이다. 방식 6개를 비교하면 '가장 좋은 것'은 반드시 나온다.
      평균 차이보다 최악/분산의 방향을 보는 것이 낫다.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30
HOLD = 126        # 첫 진입일로부터 6개월 (백테스트에서 가장 좋았던 보유기간)
WEEK = 5          # 거래일


def load_prices() -> pd.Series:
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    return pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()


def episodes(locs):
    locs = sorted({int(i) for i in locs})
    out, prev = [], -10 ** 9
    for i in locs:
        if i - prev > GAP:
            out.append(i)
        prev = i
    return out


# (비중, 첫 진입일로부터 며칠 뒤) 목록
PLANS = {
    "일괄 (100%)":        [(1.0, 0)],
    "2분할 (2주 간격)":    [(0.5, 0), (0.5, 2 * WEEK)],
    "3분할 (2주 간격)":    [(1 / 3, 0), (1 / 3, 2 * WEEK), (1 / 3, 4 * WEEK)],
    "4분할 (1주 간격)":    [(0.25, 0), (0.25, WEEK), (0.25, 2 * WEEK), (0.25, 3 * WEEK)],
    "5분할 (2주 간격)":    [(0.2, k * 2 * WEEK) for k in range(5)],
    "10분할 (매주)":       [(0.1, k * WEEK) for k in range(10)],
}


def run(V: np.ndarray, entries, plan):
    """가중 평균 매입단가 대비, 첫 진입 + HOLD 시점 수익률."""
    out = []
    for i in entries:
        exit_i = i + HOLD
        if exit_i >= len(V):
            continue
        cost, filled = 0.0, 0.0
        for wgt, off in plan:
            j = i + off
            if j >= exit_i:          # 매도일 이후에 넣을 돈은 넣지 않는다
                continue
            cost += wgt * V[j]
            filled += wgt
        if filled == 0:
            continue
        avg = cost / filled          # 실제 체결된 비중 기준 평균단가
        out.append({"i": i, "ret": V[exit_i] / avg - 1, "filled": filled})
    return out


def main() -> None:
    from signal_check import build
    px = load_prices()
    V = px.values
    w = build(px)
    locs = [px.index.searchsorted(t, side="right") for t in w[w["sig"]].index]
    E = episodes([i for i in locs if i < len(px)])

    print(f"신호 사건 {len(E)}개 · 매도는 모두 '첫 진입 + 6개월' 동일\n")
    print(f"{'진입 방식':<20s}{'n':>4s}{'평균':>9s}{'중앙':>9s}{'승률':>7s}"
          f"{'최악':>9s}{'최고':>9s}{'표준편차':>10s}")
    print("-" * 78)
    rows = []
    for name, plan in PLANS.items():
        res = run(V, E, plan)
        if not res:
            continue
        r = np.array([x["ret"] for x in res])
        print(f"{name:<20s}{len(r):>4d}{r.mean()*100:>+8.1f}%{np.median(r)*100:>+8.1f}%"
              f"{(r>0).mean()*100:>6.0f}%{r.min()*100:>+8.1f}%{r.max()*100:>+8.1f}%"
              f"{r.std()*100:>9.1f}%")
        rows.append({"plan": name, "n": len(r), "mean": float(r.mean()),
                     "median": float(np.median(r)), "hit": float((r > 0).mean()),
                     "worst": float(r.min()), "best": float(r.max()),
                     "std": float(r.std())})

    print()
    print("사건별 비교 (일괄 vs 4분할):")
    lump = {x["i"]: x["ret"] for x in run(V, E, PLANS["일괄 (100%)"])}
    quad = {x["i"]: x["ret"] for x in run(V, E, PLANS["4분할 (1주 간격)"])}
    print(f"  {'진입일':<12s}{'진입가':>8s}{'일괄':>10s}{'4분할':>10s}{'차이':>9s}")
    for i in E:
        if i not in lump:
            continue
        d = quad[i] - lump[i]
        print(f"  {px.index[i]:%Y-%m-%d}  ${V[i]:>6.0f}{lump[i]*100:>+9.1f}%"
              f"{quad[i]*100:>+9.1f}%{d*100:>+8.1f}%p")

    print()
    print("참고: 신호 직후 주가가 더 빠진 사건 수 =",
          sum(1 for i in E if i + 4 * WEEK < len(V) and V[i + 4 * WEEK] < V[i]),
          f"/ {len([i for i in E if i + 4*WEEK < len(V)])}건 (진입 후 4주 시점 기준)")

    with open(f"{BASE}/data/entry_split.json", "w") as f:
        json.dump({"hold_days": HOLD, "episodes": len(E), "plans": rows},
                  f, ensure_ascii=True, indent=2, default=float)
    print("\n저장: data/entry_split.json")


if __name__ == "__main__":
    main()
