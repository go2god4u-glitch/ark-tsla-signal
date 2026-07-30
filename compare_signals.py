"""아크 매수 신호 vs 기술적 지표 신호 — 같은 조건에서 정면 비교.

공정 비교를 위해 맞춘 것:
  - 같은 기간 (아크 데이터가 존재하는 구간)
  - 같은 사건 단위 (30거래일 이내는 한 사건)
  - 같은 진입 규칙 (신호 다음 거래일 종가)
  - 같은 기준 (무작위 진입도 같은 기간에서만 추출)

아크 신호 정의는 signal_check.py 와 동일한 코드 경로를 쓴다.
아크가 테슬라를 담은 ETF 전부(ARKK/ARKQ/ARKW/ARKX/CTRU)를 합산하되,
펀드별로 차분한 뒤 더한다(신규 편입이 매수로 잡히지 않게).

지표 신호는 각각 '상위 10% 수준의 극단'으로 맞춰 아크와 빈도를 비슷하게 둔다.
빈도가 다르면 사건 수가 달라져 비교가 불공정해진다.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30
RNG = np.random.default_rng(20260730)
NB = 20000
HORIZONS = [(63, "3개월"), (126, "6개월"), (252, "12개월")]


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


def evaluate(V, starts, h, lo):
    done = [i for i in starts if i + h < len(V) and i >= lo]
    if len(done) < 3:
        return None
    r = np.array([V[i + h] / V[i] - 1 for i in done])
    pool = np.arange(lo, len(V) - h)
    b = V[pool + h] / V[pool] - 1
    sims = np.array([b[RNG.integers(0, len(b), len(r))].mean() for _ in range(NB)])
    return {"n": len(r), "mean": float(r.mean()), "edge": float(r.mean() - b.mean()),
            "hit": float((r > 0).mean()), "p": float((sims >= r.mean()).mean()),
            "worst": float(r.min())}


def tech_signals(px: pd.Series, weeks: pd.DatetimeIndex) -> dict:
    """주간 기준 지표 신호. 전부 인과적(과거 값만)이며 확장창 분위수로 극단을 정의한다."""
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
    macd, sigl = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    ma200 = px.rolling(200).mean()
    m20, s20 = px.rolling(20).mean(), px.rolling(20).std()
    dd = px / px.rolling(252, min_periods=60).max() - 1

    w = pd.DataFrame({"rsi": rsi, "hist": macd - sigl, "gap": px / ma200 - 1,
                      "bbz": (px - m20) / s20, "dd": dd,
                      "xup": ((macd > sigl) & (macd.shift(1) <= sigl.shift(1))).astype(float),
                      }).resample("W-FRI").last().reindex(weeks)

    def low10(s):    # 하위 10% (낮을수록 매수 우호적인 지표)
        return s <= s.expanding(52).quantile(0.10).shift(1)

    return {
        "RSI 하위10% (과매도)": low10(w["rsi"]),
        "낙폭 하위10% (깊은 낙폭)": low10(w["dd"]),
        "200일선 이격 하위10%": low10(w["gap"]),
        "볼린저 z 하위10%": low10(w["bbz"]),
        "MACD 골든크로스": w["xup"] > 0,
        "MACD 히스토그램 상위10%": w["hist"] >= w["hist"].expanding(52).quantile(0.90).shift(1),
    }


def main() -> None:
    from signal_check import build
    px = load_prices()
    V = px.values
    w = build(px)
    lo = px.index.searchsorted(w.index[0])

    def entries(weeks):
        locs = [px.index.searchsorted(t, side="right") for t in weeks]
        return episodes([i for i in locs if i < len(px)])

    cands = {"★ 아크 매수신호": entries(w[w["sig"]].index)}
    for name, mask in tech_signals(px, w.index).items():
        cands[name] = entries(w.index[mask.fillna(False)])

    print(f"검증 구간 {px.index[lo]:%Y-%m-%d} ~ {px.index[-1]:%Y-%m-%d}"
          f"  (무조건 보유 {V[-1]/V[lo]:.2f}배)\n")
    print(f"{'신호':<26s}{'사건':>5s}" + "".join(f"{l+' 초과':>13s}{'승률':>6s}{'p':>7s}"
                                                for _, l in HORIZONS))
    print("-" * 104)
    rows = []
    for name, E in cands.items():
        cells, rec = "", {"name": name, "eps": len(E)}
        for h, lab in HORIZONS:
            r = evaluate(V, E, h, lo)
            if r:
                cells += f"{r['edge']*100:>+12.1f}%{r['hit']*100:>5.0f}%{r['p']:>7.3f}"
                rec[lab] = r
            else:
                cells += f"{'표본부족':>13s}{'':>6s}{'':>7s}"
        print(f"{name:<26s}{len(E):>5d}" + cells)
        rows.append(rec)

    ark = rows[0]
    print()
    print("=" * 104)
    best = max((r for r in rows[1:] if "6개월" in r), key=lambda r: r["6개월"]["edge"], default=None)
    if best and "6개월" in ark:
        print(f"6개월 기준: 아크 {ark['6개월']['edge']*100:+.1f}%(승률 {ark['6개월']['hit']*100:.0f}%) "
              f"vs 지표 최고 {best['name']} {best['6개월']['edge']*100:+.1f}%"
              f"(승률 {best['6개월']['hit']*100:.0f}%)")
        print(f"  -> 차이 {(ark['6개월']['edge']-best['6개월']['edge'])*100:+.1f}%p")
    print(f"지표 신호 {len(rows)-1}개를 시험했다. 그중 최고만 아크와 비교하면 지표에 유리한 비교다.")

    with open(f"{BASE}/data/compare_signals.json", "w") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2, default=float)
    print("저장: data/compare_signals.json")


if __name__ == "__main__":
    main()
