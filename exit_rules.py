"""매도 규칙 비교 — 언제 팔아야 하고, 실제로 몇 % 인가.

지금까지 검증한 것은 '신호 후 N개월 보유'뿐이었다. 매도 규칙 자체를 비교한 적이 없다.
이 파일은 같은 진입 신호에 매도 규칙만 바꿔 끼워 비교한다.

지키는 것:
  - 진입은 신호 다음 거래일 종가 (ARK 공시는 장 마감 후).
  - 신호/정제는 signal_check.py 와 같은 코드 경로. 즉 실전에서 재현 가능한 것만.
  - 복리 수익은 '겹치는 매매를 하나의 포지션으로' 처리한다. 신호가 연달아 뜨면
    두 배로 사는 것이 아니라 보유를 연장한다(현금 1단위로 운용 가능한 전략).
  - 비교 기준은 항상 무조건 보유(buy & hold). 테슬라가 오른 기간이므로
    절대 수익률만 보면 아무 규칙이나 좋아 보인다.

읽을 때 주의:
  매도 규칙을 8가지 시험했다. 그중 제일 좋은 것만 보면 과적합이다.
  전부 출력하고, 사건 수가 적다는 사실을 함께 본다.
"""

import json
import os

import numpy as np
import pandas as pd

from signal_check import build, build_daily, refresh_prices  # 동일 코드 경로

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30
RNG = np.random.default_rng(20260729)


def load_prices_cached() -> pd.Series:
    r = json.load(open(f"{BASE}/data/tsla_yahoo.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    return pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()


def entry_locs(px: pd.Series, w: pd.DataFrame):
    """신호 주 -> 다음 거래일 인덱스. 30거래일 이내로 붙은 것은 한 사건으로 묶는다."""
    locs = sorted({px.index.searchsorted(t, side="right") for t in w[w["sig"]].index})
    locs = [i for i in locs if i < len(px)]
    out, prev = [], -10**9
    for i in locs:
        if i - prev > GAP:
            out.append(i)
        prev = i
    return out


# ---- 매도 규칙: 진입 인덱스를 받아 매도 인덱스를 돌려준다 ----

def hold(n):
    def f(V, i, sellmask):
        return min(i + n, len(V) - 1)
    f.__name__ = f"{n}일 보유"
    return f


def trail(pct):
    def f(V, i, sellmask):
        peak = V[i]
        for j in range(i + 1, len(V)):
            peak = max(peak, V[j])
            if V[j] <= peak * (1 - pct):
                return j
        return len(V) - 1
    f.__name__ = f"고점대비 -{pct:.0%} 손절"
    return f


def target(up, cap):
    def f(V, i, sellmask):
        for j in range(i + 1, min(i + cap, len(V))):
            if V[j] >= V[i] * (1 + up):
                return j
        return min(i + cap, len(V) - 1)
    f.__name__ = f"+{up:.0%} 도달 시 (최대 {cap}일)"
    return f


def on_sell_signal(cap):
    def f(V, i, sellmask):
        for j in range(i + 1, min(i + cap, len(V))):
            if sellmask[j]:
                return j
        return min(i + cap, len(V) - 1)
    f.__name__ = f"아크 대량매도 신호 (최대 {cap}일)"
    return f


def simulate(px: pd.Series, entries, rule, sellmask):
    """겹치는 매매는 하나의 포지션으로 합친다(현금 1단위 운용)."""
    V = px.values
    trades, occupied = [], np.zeros(len(V), bool)
    for i in entries:
        if occupied[i]:
            continue
        j = rule(V, i, sellmask)
        trades.append((i, j, V[j] / V[i] - 1))
        occupied[i:j + 1] = True
    if not trades:
        return None
    r = np.array([t[2] for t in trades])
    eq = float(np.prod(1 + r))
    yrs = (px.index[-1] - px.index[0]).days / 365.25
    # 시장에 없던 기간은 현금(수익 0) 이므로 복리는 매매 수익만 곱한다
    daily = pd.Series(0.0, index=px.index)
    ret = px.pct_change().fillna(0)
    for i, j, _ in trades:
        daily.iloc[i + 1:j + 1] = ret.iloc[i + 1:j + 1]
    curve = (1 + daily).cumprod()
    mdd = float((curve / curve.cummax() - 1).min())
    return {"n": len(trades), "mean": float(r.mean()), "median": float(np.median(r)),
            "hit": float((r > 0).mean()), "best": float(r.max()), "worst": float(r.min()),
            "final": eq, "cagr": eq ** (1 / yrs) - 1, "mdd": mdd,
            "exposure": float(occupied.mean()),
            "avg_days": float(np.mean([j - i for i, j, _ in trades])),
            "curve": curve, "trades": trades}


def main() -> None:
    px = load_prices_cached()
    w = build(px)
    entries = entry_locs(px, w)

    # 매도 신호: 주간 순매수가 하위 10% (대량 매도) 인 주의 다음 거래일
    net = w["net"]
    sell_thr = net.expanding(52).quantile(0.10).shift(1)
    sell_weeks = w.index[(net <= sell_thr).fillna(False)]
    sellmask = np.zeros(len(px), bool)
    for t in sell_weeks:
        k = px.index.searchsorted(t, side="right")
        if k < len(px):
            sellmask[k] = True

    V = px.values
    yrs = (px.index[-1] - px.index[0]).days / 365.25
    bh = {"final": V[-1] / V[0], "cagr": (V[-1] / V[0]) ** (1 / yrs) - 1,
          "mdd": float((px / px.cummax() - 1).min())}

    rules = [hold(21), hold(63), hold(126), hold(252),
             trail(0.20), trail(0.30), target(0.30, 252), on_sell_signal(252)]

    print(f"진입 사건 {len(entries)}개 / 기간 {px.index[0]:%Y-%m} ~ {px.index[-1]:%Y-%m} ({yrs:.1f}년)")
    print(f"무조건 보유: {bh['final']:.2f}배  CAGR {bh['cagr']*100:+.1f}%  MDD {bh['mdd']*100:.0f}%\n")
    print(f"{'매도 규칙':<26s}{'매매':>4s}{'평균':>8s}{'중앙':>8s}{'승률':>6s}"
          f"{'최악':>8s}{'평균보유':>8s}{'누적':>7s}{'CAGR':>8s}{'MDD':>7s}{'노출':>6s}")
    print("-" * 96)
    rows = []
    for rule in rules:
        s = simulate(px, entries, rule, sellmask)
        if not s:
            continue
        print(f"{rule.__name__:<26s}{s['n']:>4d}{s['mean']*100:>+7.1f}%{s['median']*100:>+7.1f}%"
              f"{s['hit']*100:>5.0f}%{s['worst']*100:>+7.1f}%{s['avg_days']:>7.0f}일"
              f"{s['final']:>6.2f}x{s['cagr']*100:>+7.1f}%{s['mdd']*100:>6.0f}%{s['exposure']*100:>5.0f}%")
        rows.append({"rule": rule.__name__, **{k: v for k, v in s.items()
                                               if k not in ("curve", "trades")}})

    best = max(rows, key=lambda r: r["cagr"])
    print(f"\n무조건 보유 대비: {best['rule']} 가 CAGR {best['cagr']*100:+.1f}% vs "
          f"{bh['cagr']*100:+.1f}% ({best['final']:.2f}배 vs {bh['final']:.2f}배)")
    print(f"규칙 {len(rules)}개를 시험했다. 최고만 보면 과적합이다. "
          f"매매 표본은 규칙당 {rows[0]['n']}건뿐이다.")

    with open(f"{BASE}/data/exit_rules.json", "w") as f:
        json.dump({"buy_hold": bh, "rules": rows, "entries": len(entries),
                   "years": yrs}, f, ensure_ascii=True, indent=2, default=float)
    print("저장: data/exit_rules.json")


if __name__ == "__main__":
    main()
