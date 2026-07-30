"""매도 규칙 광범위 탐색 — 최고값이 아니라 '고원'을 찾는다.

앞선 분석에서 MACD 데드크로스가 4.28배로 1등이었지만 파라미터를 조금만 바꾸면
0.87배가 됐다. 그런 것은 규칙이 아니라 우연이다.

그래서 이번에는 규칙군마다 파라미터를 훑고, **최고값이 아니라 구간 전체의
중앙값과 최악값**으로 판단한다. 판단 기준:

  중앙값   파라미터를 잘못 골라도 이 정도는 나온다 -> 실제 기대치에 가깝다
  최솟값   가장 운 나쁜 선택. 이게 기준(6개월 고정)보다 크게 나쁘면 위험한 규칙이다
  최댓값   보고할 가치 없음. 어떤 규칙이든 파라미터를 늘리면 최댓값은 올라간다

기준선: 아크 매수 신호 진입 + 6개월 고정 보유 = 3.80배 (매매 5건)

한계는 그대로다. 매매가 4~8건이라 어떤 결론도 확정이 아니다.
이 탐색의 목적은 '무엇이 확실히 좋은가'가 아니라
'무엇이 파라미터를 바꿔도 무너지지 않는가'를 가리는 것이다.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30


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


def run_exit(V, buys, exit_fn, lo, yrs, cap=504):
    """exit_fn(i) -> 청산 인덱스. 겹치는 매매는 하나로 합친다."""
    trades, occ = [], np.zeros(len(V), bool)
    for i in buys:
        if occ[i]:
            continue
        j = min(exit_fn(i), i + cap, len(V) - 1)
        if j <= i:
            j = min(i + cap, len(V) - 1)
        trades.append((i, j, V[j] / V[i] - 1))
        occ[i:j + 1] = True
    if not trades:
        return None
    r = np.array([t[2] for t in trades])
    ret = pd.Series(V).pct_change().fillna(0).values
    daily = np.where(np.roll(occ, 1), ret, 0.0)
    daily[:lo] = 0.0
    curve = np.cumprod(1 + daily)
    eq = curve[-1] / curve[lo]
    mdd = float((curve[lo:] / np.maximum.accumulate(curve[lo:]) - 1).min())
    return {"n": len(r), "mean": float(r.mean()), "hit": float((r > 0).mean()),
            "final": float(eq), "cagr": float(eq ** (1 / yrs) - 1), "mdd": mdd,
            "days": float(np.mean([j - i for i, j, _ in trades]))}


def main() -> None:
    from signal_check import build
    px = load_prices()
    V = px.values
    w = build(px)
    lo = px.index.searchsorted(w.index[0])
    yrs = (px.index[-1] - px.index[lo]).days / 365.25
    buys = episodes([px.index.searchsorted(t, side="right") for t in w[w["sig"]].index
                     if px.index.searchsorted(t, side="right") < len(px)])

    # 지표 준비
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    RSI = (100 - 100 / (1 + up / dn)).values
    MA = {n: px.rolling(n).mean().values for n in (10, 20, 50, 100, 200)}
    tr = pd.concat([px.diff().abs()], axis=1).max(axis=1)
    ATR = tr.rolling(14).mean().values

    base = run_exit(V, buys, lambda i: i + 126, lo, yrs)
    print(f"기준선: 아크 매수 + 6개월 고정 = {base['final']:.2f}배 "
          f"(CAGR {base['cagr']*100:+.1f}%, MDD {base['mdd']*100:.0f}%, 매매 {base['n']}건)\n")

    families = {}

    # A. 고정 기간
    families["고정 기간 (1~12개월)"] = [
        (f"{m}개월", (lambda h: (lambda i: i + h))(m * 21)) for m in range(1, 13)]

    # B. 트레일링 스톱 (진입 후 최고가 대비)
    def make_trail(pct):
        def f(i):
            peak = V[i]
            for j in range(i + 1, len(V)):
                peak = max(peak, V[j])
                if V[j] <= peak * (1 - pct):
                    return j
            return len(V) - 1
        return f
    families["트레일링 스톱 (10~40%)"] = [
        (f"-{int(p*100)}%", make_trail(p)) for p in np.arange(0.10, 0.41, 0.05)]

    # C. 목표 수익 도달
    def make_target(up_):
        def f(i):
            for j in range(i + 1, len(V)):
                if V[j] >= V[i] * (1 + up_):
                    return j
            return len(V) - 1
        return f
    families["목표수익 도달 (+20~120%)"] = [
        (f"+{int(u*100)}%", make_target(u)) for u in np.arange(0.2, 1.21, 0.2)]

    # D. 이동평균 하향 이탈
    def make_ma(n):
        def f(i):
            for j in range(i + 1, len(V)):
                if not np.isnan(MA[n][j]) and V[j] < MA[n][j]:
                    return j
            return len(V) - 1
        return f
    families["이동평균 이탈 (10~200일)"] = [
        (f"{n}일선", make_ma(n)) for n in (10, 20, 50, 100, 200)]

    # E. RSI 과매수 후 하향 이탈
    def make_rsi(th):
        def f(i):
            armed = False
            for j in range(i + 1, len(V)):
                if RSI[j] >= th:
                    armed = True
                elif armed and RSI[j] < th:
                    return j
            return len(V) - 1
        return f
    families["RSI 과매수 이탈 (55~80)"] = [
        (f"RSI {t}", make_rsi(t)) for t in (55, 60, 65, 70, 75, 80)]

    # F. ATR 기반 샹들리에 스톱
    def make_atr(k):
        def f(i):
            peak = V[i]
            for j in range(i + 1, len(V)):
                peak = max(peak, V[j])
                if not np.isnan(ATR[j]) and V[j] <= peak - k * ATR[j]:
                    return j
            return len(V) - 1
        return f
    families["ATR 스톱 (2~7배)"] = [
        (f"{k}xATR", make_atr(k)) for k in (2, 3, 4, 5, 6, 7)]

    # G. 고정 기간 + 목표수익 (먼저 오는 쪽)
    def make_combo(h, u):
        t = make_target(u)
        return lambda i: min(i + h, t(i))
    families["6개월 상한 + 목표수익"] = [
        (f"6M/+{int(u*100)}%", make_combo(126, u)) for u in np.arange(0.2, 1.21, 0.2)]

    rows = []
    for fam, items in families.items():
        res = []
        for name, fn in items:
            r = run_exit(V, buys, fn, lo, yrs)
            if r:
                res.append((name, r))
        if not res:
            continue
        finals = np.array([r["final"] for _, r in res])
        best = max(res, key=lambda x: x[1]["final"])
        rows.append({"family": fam, "n_params": len(res),
                     "median": float(np.median(finals)),
                     "min": float(finals.min()), "max": float(finals.max()),
                     "best_param": best[0], "best": best[1]})

    rows.sort(key=lambda r: r["median"], reverse=True)
    print("=" * 92)
    print("규칙군별 — 파라미터 구간 전체의 성적 (중앙값 기준 정렬)")
    print("=" * 92)
    print(f"{'규칙군':<26s}{'파라미터':>6s}{'중앙값':>9s}{'최솟값':>9s}{'최댓값':>9s}"
          f"{'최고 파라미터':>16s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['family']:<26s}{r['n_params']:>6d}{r['median']:>8.2f}x"
              f"{r['min']:>8.2f}x{r['max']:>8.2f}x{r['best_param']:>16s}")
    print(f"\n기준선(6개월 고정) {base['final']:.2f}배와 비교하라. "
          f"중앙값이 기준선보다 높아야 '파라미터를 잘못 골라도 낫다'고 말할 수 있다.")

    print()
    print("=" * 92)
    print("상위 규칙군 상세 (파라미터별 전부)")
    print("=" * 92)
    for r in rows[:3]:
        print(f"\n[{r['family']}]")
        for name, fn in families[r["family"]]:
            x = run_exit(V, buys, fn, lo, yrs)
            if x:
                print(f"  {name:<12s} 매매 {x['n']:>2d}  평균 {x['mean']*100:>+7.1f}%  "
                      f"승률 {x['hit']*100:>3.0f}%  {x['final']:>5.2f}x  "
                      f"CAGR {x['cagr']*100:>+6.1f}%  MDD {x['mdd']*100:>4.0f}%  "
                      f"보유 {x['days']:>3.0f}일")

    with open(f"{BASE}/data/sell_search.json", "w") as f:
        json.dump({"baseline": base, "families": rows}, f,
                  ensure_ascii=True, indent=2, default=float)
    print("\n저장: data/sell_search.json")


if __name__ == "__main__":
    main()
