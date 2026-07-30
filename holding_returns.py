"""신호 발생 후 보유 기간별 수익률 — "몇 달 들고 있으면 몇 %인가"에 대한 답.

두 가지를 구분해서 낸다. 섞으면 숫자가 부풀거나 쪼그라든다.

  A. 사건 단위 기대치
     각 신호 사건의 첫날에 사서 N개월 뒤 파는 것을 독립적으로 계산한다.
     "신호 하나당 평균 몇 %인가"에 대한 답이다. 사건이 겹쳐도 각각 센다.

  B. 단일 포지션 복리
     현금 1단위로 실제 운용했을 때. 신호가 겹치면 두 배로 사는 것이 아니라
     보유를 연장한다. "실제로 계좌가 몇 배가 되는가"에 대한 답이다.
     A 보다 매매 수가 적고, 노출되지 않은 기간은 현금이다.

기준(무조건 보유)은 반드시 **신호가 존재하는 기간**에서만 계산한다.
테슬라는 2010년 이후 200배 올랐다. 15년 전체를 기준으로 삼으면 어떤 전략도
못 이기는 것처럼 보인다(score_model.py 에서 실제로 그 함정에 빠졌다).
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30
MONTHS = [(21, "1개월"), (42, "2개월"), (63, "3개월"), (126, "6개월"),
          (189, "9개월"), (252, "12개월"), (378, "18개월"), (504, "24개월")]
RNG = np.random.default_rng(20260730)
NB = 20000


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


def main() -> None:
    from signal_check import build
    px = load_prices()
    V = px.values
    w = build(px)

    # 진입 = 신호 주 다음 거래일 종가 (ARK 공시는 장 마감 후)
    locs = [px.index.searchsorted(t, side="right") for t in w[w["sig"]].index]
    E = episodes([i for i in locs if i < len(px)])

    lo = px.index.searchsorted(w.index[0])          # 신호가 존재할 수 있는 구간
    hi = len(V) - 1
    yrs = (px.index[hi] - px.index[lo]).days / 365.25

    print(f"신호 사건 {len(E)}개  |  검증 구간 {px.index[lo]:%Y-%m-%d} ~ {px.index[hi]:%Y-%m-%d} ({yrs:.1f}년)")
    print(f"무조건 보유: {V[hi]/V[lo]:.2f}배  CAGR {((V[hi]/V[lo])**(1/yrs)-1)*100:+.1f}%\n")

    print("=" * 100)
    print("A. 사건 단위 — 신호 하나당 기대치 (진입 = 신호 다음 거래일 종가)")
    print("=" * 100)
    print(f"{'보유':<8s}{'n':>4s}{'평균':>9s}{'중앙':>9s}{'승률':>7s}"
          f"{'최악':>9s}{'최고':>9s}{'무작위':>9s}{'초과':>9s}{'p':>8s}")
    rows = []
    for h, lab in MONTHS:
        done = [i for i in E if i + h < len(V)]
        if len(done) < 3:
            print(f"{lab:<8s}{len(done):>4d}   표본 부족 (3건 미만) — 아직 기간이 지나지 않았다")
            continue
        r = np.array([V[i + h] / V[i] - 1 for i in done])
        pool = np.arange(lo, len(V) - h)
        b = V[pool + h] / V[pool] - 1
        sims = np.array([b[RNG.integers(0, len(b), len(r))].mean() for _ in range(NB)])
        p = float((sims >= r.mean()).mean())
        print(f"{lab:<8s}{len(r):>4d}{r.mean()*100:>+8.1f}%{np.median(r)*100:>+8.1f}%"
              f"{(r>0).mean()*100:>6.0f}%{r.min()*100:>+8.1f}%{r.max()*100:>+8.1f}%"
              f"{b.mean()*100:>+8.1f}%{(r.mean()-b.mean())*100:>+8.1f}%{p:>8.3f}")
        rows.append({"h": h, "label": lab, "n": len(r), "mean": float(r.mean()),
                     "median": float(np.median(r)), "hit": float((r > 0).mean()),
                     "worst": float(r.min()), "best": float(r.max()),
                     "base": float(b.mean()), "edge": float(r.mean() - b.mean()), "p": p})

    print()
    print("=" * 100)
    print("B. 단일 포지션 복리 — 현금 1단위로 실제 운용했다면")
    print("=" * 100)
    print(f"{'보유':<8s}{'매매':>5s}{'누적':>8s}{'CAGR':>9s}{'MDD':>8s}{'노출':>7s}{'무조건보유 대비':>14s}")
    bh = V[hi] / V[lo]
    comp = []
    for h, lab in MONTHS:
        occupied = np.zeros(len(V), bool)
        trades = []
        for i in E:
            if occupied[i] or i + 1 >= len(V):
                continue
            j = min(i + h, hi)
            trades.append(V[j] / V[i] - 1)
            occupied[i:j + 1] = True
        if not trades:
            continue
        ret = pd.Series(V, index=px.index).pct_change().fillna(0)
        daily = ret.where(pd.Series(occupied, index=px.index).shift(1).fillna(False), 0.0)
        daily.iloc[:lo] = 0.0
        curve = (1 + daily).cumprod()
        eq = float(curve.iloc[hi] / curve.iloc[lo])
        mdd = float((curve.iloc[lo:] / curve.iloc[lo:].cummax() - 1).min())
        print(f"{lab:<8s}{len(trades):>5d}{eq:>7.2f}x{(eq**(1/yrs)-1)*100:>+8.1f}%"
              f"{mdd*100:>7.0f}%{occupied[lo:].mean()*100:>6.0f}%{eq/bh:>13.2f}배")
        comp.append({"label": lab, "trades": len(trades), "final": eq,
                     "cagr": eq ** (1 / yrs) - 1, "mdd": mdd,
                     "exposure": float(occupied[lo:].mean()), "vs_bh": eq / bh})

    print()
    print("신호별 개별 성과:")
    print(f"  {'진입일':<12s}{'진입가':>8s}" + "".join(f"{l:>9s}" for _, l in MONTHS[:6]))
    for i in E:
        cells = "".join(
            (f"{(V[i+h]/V[i]-1)*100:>+8.1f}%" if i + h < len(V) else f"{'진행중':>9s}")
            for h, _ in MONTHS[:6])
        print(f"  {px.index[i]:%Y-%m-%d}  ${V[i]:>6.0f}" + cells)

    with open(f"{BASE}/data/holding_returns.json", "w") as f:
        json.dump({"episodes": len(E), "years": yrs, "buy_hold": float(bh),
                   "by_period": rows, "compound": comp,
                   "signals": [{"d": px.index[i].strftime("%Y-%m-%d"), "px": float(V[i])}
                               for i in E]}, f, ensure_ascii=True, indent=2, default=float)
    print("\n저장: data/holding_returns.json")


if __name__ == "__main__":
    main()
