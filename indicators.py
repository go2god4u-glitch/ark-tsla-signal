"""기술적 지표와 아크 신호의 합류(confluence) 검증.

설계 원칙 — 표본 크기가 이 분석의 전부다:

  아크 신호는 5년간 독립 사건 9개다. 여기에 조건을 하나 걸 때마다 표본이 반토막 난다.
  지표 3개를 추가하면 조합이 수십 개가 되고, 그중 '제일 좋은 것'은 반드시 나온다.
  그것은 발견이 아니라 자유도의 산물이다.

  그래서 두 층으로 나눈다:
    1층. 지표 단독 — 전체 가격 이력에서 검증한다. 사건이 20~60개라 통계가 성립한다.
    2층. 아크 ∩ 지표 — 사건이 2~5개다. 통계가 아니라 '사례'로만 읽는다.

  시험한 조합은 전부 출력한다. 골라내면 그 순간 거짓말이 된다.

모든 지표는 정의상 인과적이다(과거 값만 사용). 진입은 신호 다음 거래일 종가.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30          # 이만큼 떨어져야 별개 사건
RNG = np.random.default_rng(20260730)
NB = 20000


def load_prices() -> pd.Series:
    r = json.load(open(f"{BASE}/data/tsla_yahoo.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    return pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()


def indicators(px: pd.Series) -> pd.DataFrame:
    """표준 지표. 전부 과거 값만 쓰므로 룩어헤드가 없다."""
    d = pd.DataFrame({"close": px})

    # RSI(14) — Wilder 평활
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn)

    # MACD(12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # 이동평균
    for n in (20, 50, 200):
        d[f"ma{n}"] = px.rolling(n).mean()

    # 볼린저(20,2)
    m, s = px.rolling(20).mean(), px.rolling(20).std()
    d["bb_low"] = m - 2 * s

    # 전고점 대비 낙폭 (252일 고점)
    d["dd"] = (px / px.rolling(252, min_periods=60).max() - 1) * 100
    return d


def signals(d: pd.DataFrame) -> dict:
    """각 지표의 '매수 조건'. 교차형은 발생 시점만 True 가 된다."""
    x = d
    cross_up = (x["macd"] > x["macd_sig"]) & (x["macd"].shift(1) <= x["macd_sig"].shift(1))
    golden = (x["ma50"] > x["ma200"]) & (x["ma50"].shift(1) <= x["ma200"].shift(1))
    return {
        "rsi30":     x["rsi"] < 30,                                  # 과매도
        "rsi25":     x["rsi"] < 25,                                  # 강한 과매도
        "rsi_exit":  (x["rsi"] > 30) & (x["rsi"].shift(1) <= 30),    # 과매도 탈출
        "macd_x":    cross_up,                                       # MACD 골든크로스
        "macd_x_neg": cross_up & (x["macd"] < 0),                    # 0선 아래 교차
        "golden":    golden,                                         # 50/200 골든크로스
        "below200":  x["close"] < x["ma200"],                        # 200일선 아래
        "bb_low":    x["close"] < x["bb_low"],                       # 볼린저 하단 이탈
        "dd30":      x["dd"] <= -30,                                 # 낙폭 -30% 이하
    }


def episodes(locs, gap: int = GAP):
    """붙어 있는 진입 시점을 한 사건으로 묶고 첫날만 남긴다."""
    locs = sorted({int(i) for i in locs})
    if not locs:
        return []
    out, prev = [], -10 ** 9
    for i in locs:
        if i - prev > gap:
            out.append(i)
        prev = i
    return out


def evaluate(V: np.ndarray, starts, h: int):
    done = [i for i in starts if i + h < len(V)]
    if len(done) < 3:
        return None
    r = np.array([V[i + h] / V[i] - 1 for i in done])
    pool = np.arange(len(V) - h)
    b = V[pool + h] / V[pool] - 1
    sims = np.array([b[RNG.integers(0, len(b), len(r))].mean() for _ in range(NB)])
    return {"n": len(r), "mean": float(r.mean()), "edge": float(r.mean() - b.mean()),
            "hit": float((r > 0).mean()), "p": float((sims >= r.mean()).mean()),
            "worst": float(r.min())}


def ark_entries(px: pd.Series):
    """signal_check.py 와 동일한 코드 경로 — 실전에서 재현 가능한 신호만 쓴다."""
    from signal_check import build
    w = build(px)
    locs = [px.index.searchsorted(t, side="right") for t in w[w["sig"]].index]
    return episodes([i for i in locs if i < len(px)]), w


def main() -> None:
    px = load_prices()
    V = px.values
    d = indicators(px)
    sig = signals(d)
    ark, _ = ark_entries(px)
    ark_set = set(ark)

    print("=" * 92)
    print("1층. 지표 단독 — 전체 가격 이력에서 검증 (표본이 커서 통계가 성립한다)")
    print("=" * 92)
    print(f"{'지표':<16s}{'사건':>5s}{'6M초과':>9s}{'승률':>6s}{'p':>7s}"
          f"{'12M초과':>10s}{'승률':>6s}{'p':>7s}")
    rows = []
    for name, mask in sig.items():
        eps = episodes(np.where(mask.fillna(False).values)[0])
        a, b = evaluate(V, eps, 126), evaluate(V, eps, 252)
        if not a:
            continue
        print(f"{name:<16s}{len(eps):>5d}{a['edge']*100:>+8.1f}%{a['hit']*100:>5.0f}%{a['p']:>7.3f}"
              + (f"{b['edge']*100:>+9.1f}%{b['hit']*100:>5.0f}%{b['p']:>7.3f}" if b else " " * 22))
        rows.append({"id": name, "eps": len(eps), "h126": a, "h252": b})

    print()
    print("=" * 92)
    print("2층. 아크 ∩ 지표 — 사건 2~5개. 통계가 아니라 사례다")
    print("=" * 92)
    ark_eval = evaluate(V, ark, 126)
    print(f"{'아크 단독':<26s}{len(ark):>4d}건  6M 초과 {ark_eval['edge']*100:>+6.1f}% "
          f"승률 {ark_eval['hit']*100:>3.0f}%  p={ark_eval['p']:.3f}")
    print()

    # 아크 신호 시점에 지표가 함께 켜져 있었는가 (전후 5거래일 허용)
    conf = []
    for name, mask in sig.items():
        m = mask.fillna(False).values
        hit = [i for i in ark
               if m[max(0, i - 5):min(len(m), i + 1)].any()]
        e = evaluate(V, hit, 126)
        conf.append({"id": name, "n": len(hit),
                     "edge": e["edge"] if e else None, "hit": e["hit"] if e else None,
                     "p": e["p"] if e else None})
        tag = (f"6M 초과 {e['edge']*100:>+6.1f}% 승률 {e['hit']*100:>3.0f}% p={e['p']:.3f}"
               if e else "표본 부족(3건 미만) — 평가 불가")
        print(f"  아크 ∩ {name:<14s} {len(hit):>2d}/{len(ark)}건  {tag}")

    # 아크 신호별로 몇 개 지표가 동시에 켜졌는지 (합류 강도)
    print()
    print("=" * 92)
    print("3층. 합류 강도 — 아크 신호 시점에 켜진 지표 개수별")
    print("=" * 92)
    counts = []
    for i in ark:
        c = sum(1 for name, mask in sig.items()
                if mask.fillna(False).values[max(0, i - 5):i + 1].any())
        r6 = V[i + 126] / V[i] - 1 if i + 126 < len(V) else None
        counts.append({"date": px.index[i].strftime("%Y-%m-%d"), "px": float(V[i]),
                       "k": c, "r6": r6})
        print(f"  {px.index[i]:%Y-%m-%d}  ${V[i]:>6.0f}  지표 {c}/{len(sig)}개 켜짐  "
              + (f"6M {r6*100:+6.1f}%" if r6 is not None else "6M 진행중"))
    done = [c for c in counts if c["r6"] is not None]
    if done:
        ks = np.array([c["k"] for c in done]); rs = np.array([c["r6"] for c in done])
        print(f"\n  켜진 지표 개수 vs 6개월 수익률 상관: r = {np.corrcoef(ks, rs)[0,1]:+.3f} (n={len(done)})")
        print("  n 이 한 자리다. 이 상관계수로는 아무것도 결론지을 수 없다.")

    n_tests = len(sig) * 2 + len(conf)
    print(f"\n시험한 조합 {n_tests}개. Bonferroni 기준이면 p < {0.05/n_tests:.4f} 여야 한다.")

    with open(f"{BASE}/data/indicators.json", "w") as f:
        json.dump({"standalone": rows, "confluence": conf, "ark_events": counts,
                   "ark_alone": ark_eval, "n_tests": n_tests},
                  f, ensure_ascii=True, indent=2, default=float)
    print("저장: data/indicators.json")


if __name__ == "__main__":
    main()
