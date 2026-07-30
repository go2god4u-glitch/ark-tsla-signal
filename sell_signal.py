"""매도 타이밍 분석 — 아크가 팔 때 같이 팔면 더 버는가.

두 질문을 나눈다. 섞으면 답이 안 나온다.

  A. 아크 매도 신호에 예측력이 있는가
     아크가 대량 매도한 뒤 주가가 실제로 빠졌는가를 사건 단위로 본다.
     매수 신호와 대칭인 질문이다. 여기서 초과수익이 '음수'여야 매도 신호로서
     의미가 있다(팔고 나서 떨어져야 잘 판 것이다).

  B. 매도 규칙으로서 6개월 보유를 이기는가
     진입은 아크 매수 신호로 고정하고 청산 규칙만 바꿔 비교한다.
     A 에서 예측력이 있더라도, 매수 신호와 청산 시점이 맞물리지 않으면
     전략 성과는 나빠질 수 있다. 별개의 질문이다.

  그리고 매수 때와 같은 질문을 매도에도 던진다:
     기술적 지표 기반 매도가 아크 매도보다 나은가.

규율은 매수 분석과 동일하다.
  - 신호/정제는 signal_check.py 와 같은 코드 경로 (실전 재현 가능한 것만)
  - 사건 단위 (30거래일 이내는 한 사건)
  - 진입·청산 모두 신호 다음 거래일 종가 (공시는 장 마감 후)
  - 기준은 같은 기간에서만 추출
  - 시험한 규칙을 전부 출력한다
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30
QUANT_SELL = 0.10          # 하위 10% = 대량 매도
RNG = np.random.default_rng(20260730)
NB = 20000
HOLD_CAP = 504             # 규칙이 안 걸리면 최대 2년에서 강제 청산


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
    # 매도 신호는 '이후가 나빠야' 좋은 것이므로 양쪽 꼬리를 다 본다
    return {"n": len(r), "mean": float(r.mean()), "edge": float(r.mean() - b.mean()),
            "hit": float((r > 0).mean()),
            "p_worse": float((sims <= r.mean()).mean())}


def tech_masks(px: pd.Series, weeks: pd.DatetimeIndex) -> dict:
    """매도 쪽 지표 조건. 전부 인과적."""
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
    macd, sigl = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    ma50 = px.rolling(50).mean()
    m20, s20 = px.rolling(20).mean(), px.rolling(20).std()
    dd = px / px.rolling(252, min_periods=60).max() - 1

    w = pd.DataFrame({
        "rsi": rsi,
        "dead": ((macd < sigl) & (macd.shift(1) >= sigl.shift(1))).astype(float),
        "bbz": (px - m20) / s20,
        "ma50_break": (px < ma50).astype(float),
        "dd": dd,
    }).resample("W-FRI").last().reindex(weeks)

    return {
        "RSI 상위10% (과매수)": w["rsi"] >= w["rsi"].expanding(52).quantile(0.90).shift(1),
        "MACD 데드크로스": w["dead"] > 0,
        "볼린저 z 상위10%": w["bbz"] >= w["bbz"].expanding(52).quantile(0.90).shift(1),
        "50일선 하향 이탈": w["ma50_break"] > 0,
        "전고점 회복 (낙폭 -10% 이내)": w["dd"] >= -0.10,
    }


def simulate(V, entries, exit_locs_sorted, cap=HOLD_CAP):
    """진입 후 첫 번째 청산 신호에서 판다. 없으면 cap 에서 강제 청산."""
    ex = np.array(exit_locs_sorted)
    trades, occupied = [], np.zeros(len(V), bool)
    for i in entries:
        if occupied[i]:
            continue
        later = ex[ex > i] if len(ex) else np.array([])
        j = int(later[0]) if len(later) else i + cap
        j = min(j, i + cap, len(V) - 1)
        trades.append((i, j, V[j] / V[i] - 1))
        occupied[i:j + 1] = True
    return trades


def summarize(V, trades, lo, yrs):
    if not trades:
        return None
    r = np.array([t[2] for t in trades])
    occ = np.zeros(len(V), bool)
    for i, j, _ in trades:
        occ[i:j + 1] = True
    ret = pd.Series(V).pct_change().fillna(0).values
    daily = np.where(np.roll(occ, 1), ret, 0.0)
    daily[:lo] = 0.0
    curve = np.cumprod(1 + daily)
    eq = curve[-1] / curve[lo]
    peak = np.maximum.accumulate(curve[lo:])
    mdd = float((curve[lo:] / peak - 1).min())
    return {"n": len(r), "mean": float(r.mean()), "median": float(np.median(r)),
            "hit": float((r > 0).mean()), "worst": float(r.min()),
            "days": float(np.mean([j - i for i, j, _ in trades])),
            "final": float(eq), "cagr": float(eq ** (1 / yrs) - 1), "mdd": mdd,
            "exposure": float(occ[lo:].mean())}


def main() -> None:
    from signal_check import build
    px = load_prices()
    V = px.values
    w = build(px)
    lo = px.index.searchsorted(w.index[0])
    yrs = (px.index[-1] - px.index[lo]).days / 365.25

    def to_locs(weeks):
        return [px.index.searchsorted(t, side="right") for t in weeks
                if px.index.searchsorted(t, side="right") < len(px)]

    buys = episodes(to_locs(w[w["sig"]].index))

    # 아크 매도 신호: 주간 순매수가 하위 10% (확장창, 1주 시프트 — 과거만 사용)
    sell_thr = w["netpct"].expanding(52).quantile(QUANT_SELL).shift(1)
    ark_sell_weeks = w.index[(w["netpct"] <= sell_thr).fillna(False)]
    ark_sell = episodes(to_locs(ark_sell_weeks))

    print("=" * 96)
    print("A. 아크가 팔면 주가가 빠지는가 (매도 신호의 예측력)")
    print("=" * 96)
    print("   매도 신호가 유효하려면 '초과'가 음수여야 한다 — 팔고 나서 떨어져야 잘 판 것이다.")
    print(f"\n{'신호':<28s}{'사건':>5s}" + "".join(f"{l:>11s}{'상승률':>7s}{'p(하락)':>9s}"
                                                for l in ("3개월 초과", "6개월 초과")))
    print("-" * 96)
    cands = {"★ 아크 대량매도": ark_sell}
    for name, mask in tech_masks(px, w.index).items():
        cands[name] = episodes(to_locs(w.index[mask.fillna(False)]))
    rows_a = []
    for name, E in cands.items():
        cells, rec = "", {"name": name, "eps": len(E)}
        for h, lab in [(63, "3개월"), (126, "6개월")]:
            r = evaluate(V, E, h, lo)
            if r:
                cells += f"{r['edge']*100:>+10.1f}%{r['hit']*100:>6.0f}%{r['p_worse']:>9.3f}"
                rec[lab] = r
            else:
                cells += f"{'표본부족':>11s}{'':>7s}{'':>9s}"
        print(f"{name:<28s}{len(E):>5d}" + cells)
        rows_a.append(rec)

    print()
    print("=" * 96)
    print("B. 매도 규칙으로서 6개월 보유를 이기는가 (진입은 아크 매수 신호로 고정)")
    print("=" * 96)
    print(f"진입 사건 {len(buys)}개 · 무조건 보유 {V[-1]/V[lo]:.2f}배 (CAGR {((V[-1]/V[lo])**(1/yrs)-1)*100:+.1f}%)")
    print(f"\n{'청산 규칙':<28s}{'매매':>5s}{'평균':>9s}{'중앙':>9s}{'승률':>6s}"
          f"{'최악':>9s}{'보유일':>7s}{'누적':>8s}{'CAGR':>8s}{'MDD':>7s}")
    print("-" * 96)
    rules = {"6개월 고정 (기준)": None, "★ 아크 대량매도": ark_sell}
    for name, mask in tech_masks(px, w.index).items():
        rules[name] = episodes(to_locs(w.index[mask.fillna(False)]))
    rows_b = []
    for name, ex in rules.items():
        if ex is None:
            trades = [(i, min(i + 126, len(V) - 1), V[min(i + 126, len(V) - 1)] / V[i] - 1)
                      for i in buys if i + 1 < len(V)]
            # 겹침 병합
            trades = simulate(V, buys, [i + 126 for i in buys])
        else:
            trades = simulate(V, buys, ex)
        s = summarize(V, trades, lo, yrs)
        if not s:
            continue
        print(f"{name:<28s}{s['n']:>5d}{s['mean']*100:>+8.1f}%{s['median']*100:>+8.1f}%"
              f"{s['hit']*100:>5.0f}%{s['worst']*100:>+8.1f}%{s['days']:>6.0f}일"
              f"{s['final']:>7.2f}x{s['cagr']*100:>+7.1f}%{s['mdd']*100:>6.0f}%")
        rows_b.append({"rule": name, **s})

    with open(f"{BASE}/data/sell_signal.json", "w") as f:
        json.dump({"predictive": rows_a, "strategy": rows_b,
                   "buys": len(buys), "ark_sell_events": len(ark_sell)},
                  f, ensure_ascii=True, indent=2, default=float)
    print(f"\n규칙 {len(rules)}개를 시험했다. 최고만 보면 과적합이다.")
    print("저장: data/sell_signal.json")


if __name__ == "__main__":
    main()
