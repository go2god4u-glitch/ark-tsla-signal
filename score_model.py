"""종합 점수 모델 — 모든 데이터를 합쳐 매주 0~100점을 매기고, 점수로 신호를 준다.

왜 '사건 9개'가 아니라 '점수'인가:
  아크 신호를 켜짐/꺼짐으로 보면 5년간 사건이 9개다. 여기에 지표 조건을 걸면
  2~4개로 쪼그라들어 아무것도 검증할 수 없다.
  대신 매주 모든 데이터를 점수로 환산하면 관측이 290주가 된다. 점수 구간별로
  이후 수익률을 비교할 수 있고, 이건 표본이 있는 질문이다.

가중치를 '맞추지' 않는다:
  9개 사건에 맞춰 가중치를 최적화하면 그 순간 과적합이다.
  그래서 모든 구성요소를 동일 가중으로 둔다. 사전에 정한 값이고
  백테스트 성적을 보고 조정하지 않는다.

모든 구성요소는 인과적이다:
  각 지표를 '그 시점까지의 자기 이력' 안에서 백분위로 환산한다(확장창).
  미래 분포를 쓰지 않으므로 실전에서 같은 값이 재현된다.

구성요소 (전부 '높을수록 매수 우호적'):
  ark   아크 주간 순매수의 확장창 백분위
  rsi   RSI 의 역백분위 (낮을수록 높은 점수 = 과매도)
  dd    전고점 대비 낙폭의 역백분위 (깊을수록 높은 점수)
  macd  MACD 히스토그램의 백분위 (반등 전환)
  ma    200일선 대비 이격의 역백분위 (아래일수록 높은 점수)
  bb    볼린저 밴드 내 위치의 역백분위 (하단일수록 높은 점수)
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GAP = 30
MIN_HIST = 52          # 백분위 계산에 필요한 최소 주 수
RNG = np.random.default_rng(20260730)
NB = 20000


def load_prices() -> pd.Series:
    path = f"{BASE}/data/tsla_full.json"
    if not os.path.exists(path):
        path = f"{BASE}/data/tsla_yahoo.json"
    r = json.load(open(path))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    return pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()


def pctrank(s: pd.Series) -> pd.Series:
    """그 시점까지의 자기 이력 안에서의 백분위(0~1). 미래를 쓰지 않는다.

    NaN 주의: 아크 데이터는 2020-12 부터라 그 이전 주가 전부 NaN 이다.
    NaN 을 그대로 두고 expanding 을 돌리면 (a[-1] >= a) 비교가 False 가 되어
    분모만 커지고, 아크 점수가 영구히 눌린다(실측: 상위 20% 진입이 0주).
    유효값만 남겨 순위를 매긴 뒤 원래 인덱스로 되돌린다."""
    v = s.dropna()
    r = v.expanding(MIN_HIST).apply(lambda a: (a[-1] >= a).mean(), raw=True)
    return r.reindex(s.index)


def components(px: pd.Series, ark_w: pd.Series) -> pd.DataFrame:
    """주간 기준 구성요소. 전부 '높을수록 매수 우호적' 방향으로 정렬한다."""
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    ma200 = px.rolling(200).mean()
    m20, s20 = px.rolling(20).mean(), px.rolling(20).std()
    bb_pos = (px - (m20 - 2 * s20)) / (4 * s20)
    dd = px / px.rolling(252, min_periods=60).max() - 1

    wk = pd.DataFrame({
        "close": px, "rsi": rsi, "macd_hist": macd_hist,
        "ma_gap": px / ma200 - 1, "bb_pos": bb_pos, "dd": dd,
    }).resample("W-FRI").last()
    wk["ark_net"] = ark_w.reindex(wk.index)

    c = pd.DataFrame(index=wk.index)
    c["ark"] = pctrank(wk["ark_net"])              # 많이 살수록 높음
    c["rsi"] = 1 - pctrank(wk["rsi"])              # 낮을수록 높음
    c["dd"] = 1 - pctrank(wk["dd"])                # 깊을수록 높음
    c["macd"] = pctrank(wk["macd_hist"])           # 반등 전환일수록 높음
    c["ma"] = 1 - pctrank(wk["ma_gap"])            # 200일선 아래일수록 높음
    c["bb"] = 1 - pctrank(wk["bb_pos"])            # 하단일수록 높음
    c["close"] = wk["close"]
    return c


def episodes(locs, gap: int = GAP):
    locs = sorted({int(i) for i in locs})
    out, prev = [], -10 ** 9
    for i in locs:
        if i - prev > gap:
            out.append(i)
        prev = i
    return out


def evaluate(V: np.ndarray, starts, h: int, lo: int = None, hi: int = None):
    """무작위 진입 기준은 반드시 '신호와 같은 기간'에서 뽑아야 한다.

    테슬라는 2011년 이후 200배 올랐다. 15년 전체에서 기준을 뽑고 2021년 이후
    신호와 비교하면, 어떤 신호든 기준에 못 미쳐 전 구간이 음수로 나온다
    (실측: 모든 점수 구간 -13% ~ -39%). 국면이 다른 표본을 섞은 탓이지
    신호가 나쁜 것이 아니다."""
    done = [i for i in starts if i + h < len(V)]
    if len(done) < 3:
        return None
    r = np.array([V[i + h] / V[i] - 1 for i in done])
    a = 0 if lo is None else max(0, lo)
    b = len(V) - h if hi is None else min(len(V) - h, hi + 1)
    pool = np.arange(a, max(a + 1, b))
    b = V[pool + h] / V[pool] - 1
    sims = np.array([b[RNG.integers(0, len(b), len(r))].mean() for _ in range(NB)])
    return {"n": len(r), "mean": float(r.mean()), "edge": float(r.mean() - b.mean()),
            "hit": float((r > 0).mean()), "p": float((sims >= r.mean()).mean()),
            "worst": float(r.min())}


def span_of(px: pd.Series, idx) -> tuple:
    """비교 기준을 뽑을 구간 = 그 데이터가 존재하는 구간."""
    return (px.index.searchsorted(idx[0]), px.index.searchsorted(idx[-1]))


def entries_for(px: pd.Series, weeks) -> list:
    """신호 주 -> 다음 거래일. 공시가 장 마감 후이므로 당일 진입은 불가."""
    locs = [px.index.searchsorted(t, side="right") for t in weeks]
    return episodes([i for i in locs if i < len(px)])


def main() -> None:
    from signal_check import build_daily
    px = load_prices()
    V = px.values

    sh = build_daily(px)["shares"]
    ark_w = sh.resample("W-FRI").last().dropna().diff()

    c = components(px, ark_w)
    cols = ["ark", "rsi", "dd", "macd", "ma", "bb"]
    # 동일 가중. 백테스트를 보고 조정하지 않는다.
    # 아크가 있는 구간에서만 종합점수를 낸다. 아크 없는 주를 5개 요소로 채우면
    # 시대마다 점수의 의미가 달라져 구간 비교가 성립하지 않는다.
    c["score_noark"] = c[[x for x in cols if x != "ark"]].mean(axis=1) * 100
    c["score"] = c[cols].mean(axis=1) * 100
    full = c.copy()                       # 아크 제외 점수는 전체 이력에서 본다
    c = c.dropna(subset=cols)             # 종합점수는 아크 구간만

    SPAN = span_of(px, c.index)
    print(f"종합점수 구간: 주 {len(c)}개  ({c.index[0]:%Y-%m-%d} ~ {c.index[-1]:%Y-%m-%d})")
    print(f"아크 제외 점수: 주 {len(full.dropna(subset=['score_noark']))}개 "
          f"({full['score_noark'].dropna().index[0]:%Y-%m-%d} ~)\n")

    print("=" * 88)
    print("1. 점수 구간별 이후 수익률 (사건 단위, 진입 = 다음 거래일 종가)")
    print("=" * 88)
    print(f"{'점수 구간':<14s}{'주':>5s}{'사건':>5s}{'6M초과':>9s}{'승률':>6s}{'p':>7s}"
          f"{'12M초과':>10s}{'승률':>6s}{'p':>7s}")
    rows = []
    for lo, hi in [(0, 40), (40, 55), (55, 65), (65, 75), (75, 101)]:
        wk = c.index[(c["score"] >= lo) & (c["score"] < hi)]
        eps = entries_for(px, wk)
        a, b = evaluate(V, eps, 126, *SPAN), evaluate(V, eps, 252, *SPAN)
        lbl = f"{lo}~{hi if hi <= 100 else 100}점"
        print(f"{lbl:<14s}{len(wk):>5d}{len(eps):>5d}"
              + (f"{a['edge']*100:>+8.1f}%{a['hit']*100:>5.0f}%{a['p']:>7.3f}" if a else " " * 21)
              + (f"{b['edge']*100:>+9.1f}%{b['hit']*100:>5.0f}%{b['p']:>7.3f}" if b else ""))
        rows.append({"band": lbl, "weeks": len(wk), "eps": len(eps),
                     "h126": a, "h252": b})

    print()
    print("=" * 88)
    print("2. 아크를 넣은 점수 vs 뺀 점수 — 아크가 기여하는가")
    print("=" * 88)
    for name, src, col, cut in [("종합점수(아크 포함)", c, "score", 70),
                                ("같은 구간, 아크 제외", c, "score_noark", 70),
                                ("아크 제외·전체이력", full, "score_noark", 70)]:
        wk = src.index[src[col] >= cut]
        eps = entries_for(px, wk)
        a = evaluate(V, eps, 126, *span_of(px, src.index))
        print(f"  {name:<22s} {cut}점 이상  주 {len(wk):>3d} 사건 {len(eps):>2d}  "
              + (f"6M 초과 {a['edge']*100:>+6.1f}% 승률 {a['hit']*100:>3.0f}% p={a['p']:.3f}"
                 if a else "표본부족"))

    print()
    print("=" * 88)
    print("3. 구성요소별 단독 기여 (해당 요소만 상위 20%)")
    print("=" * 88)
    contrib = []
    for col in cols:
        wk = c.index[c[col] >= 0.80]
        eps = entries_for(px, wk)
        a = evaluate(V, eps, 126, *SPAN)
        print(f"  {col:<6s} 주 {len(wk):>3d} 사건 {len(eps):>2d}  "
              + (f"6M 초과 {a['edge']*100:>+6.1f}% 승률 {a['hit']*100:>3.0f}% p={a['p']:.3f}"
                 if a else "표본부족"))
        contrib.append({"c": col, "eps": len(eps),
                        "edge": a["edge"] if a else None, "p": a["p"] if a else None})

    cur = c.iloc[-1]
    print()
    print("=" * 88)
    print(f"현재 점수 {cur['score']:.0f}점  ({c.index[-1]:%Y-%m-%d} 주)")
    print("=" * 88)
    for col in cols:
        bar = "█" * int(cur[col] * 20)
        print(f"  {col:<6s} {cur[col]*100:>5.0f}점  {bar}")

    out = {"bands": rows, "contrib": contrib,
           "current": {"week": c.index[-1].strftime("%Y-%m-%d"),
                       "score": float(cur["score"]),
                       "components": {k: float(cur[k]) * 100 for k in cols}},
           "history": [{"d": d.strftime("%Y-%m-%d"), "s": round(float(r["score"]), 1)}
                       for d, r in c.tail(120).iterrows()]}
    with open(f"{BASE}/data/score_model.json", "w") as f:
        json.dump(out, f, ensure_ascii=True, indent=2, default=float)
    print("\n저장: data/score_model.json")


if __name__ == "__main__":
    main()
