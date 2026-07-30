"""조건부 기대수익률 — "지금 같은 상태에서 아크가 사면 얼마인가".

사장님 질문: 오르는 중에 사는 것 말고, 빠졌을 때 사고 싶다.
             지금 같은 기술적 상태에서 아크 매수가 이어지면 기대수익률은?

그래서 신호를 '진입 시점의 기술적 상태'로 나눠 이후 수익률을 계산한다.
신호는 사건 단위(8개)가 아니라 개별 신호(20개)를 쓴다. 표본이 필요하다.

청산 기준을 셋 다 계산해 비교한다:
  RSI 이탈  RSI(14) 70 돌파 후 이탈 (확정 규칙)
  고정      3개월 / 6개월
  아크 매도 아크 주간 순매수가 하위 10% 로 떨어지는 날

한계: 조건을 걸면 표본이 5~10건으로 준다. 방향만 읽어야 한다.
"""
import json
import numpy as np
import pandas as pd

RSI_LEVEL, MAX_HOLD = 70, 126

def rsi14(px):
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    return 100 - 100/(1 + up/dn)

def main():
    from signal_check import build
    r = json.load(open('data/tsla_full.json'))['chart']['result'][0]
    idx = pd.to_datetime(pd.Series(r['timestamp']), unit='s').dt.normalize()
    px = pd.Series(r['indicators']['quote'][0]['close'], index=idx).dropna().sort_index()
    V = px.values
    RSI = rsi14(px).values
    MA200 = px.rolling(200).mean().values
    DD = (px / px.rolling(252, min_periods=60).max() - 1).values * 100

    w = build(px)
    sig = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                  if px.index.searchsorted(t, side='right') < len(px)})
    # 아크 매도일 (하위 10%)
    st = w['netpct'].expanding(52).quantile(0.10).shift(1)
    sell_w = w.index[(w['netpct'] <= st).fillna(False)]
    sell_locs = sorted({px.index.searchsorted(t, side='right') for t in sell_w
                        if px.index.searchsorted(t, side='right') < len(px)})

    def exit_rsi(e):
        armed = False
        for j in range(e+1, len(V)):
            if RSI[j] >= RSI_LEVEL: armed = True
            if (armed and RSI[j] < RSI_LEVEL) or (j-e) >= MAX_HOLD: return j
        return None
    def exit_fixed(e, h):
        return e+h if e+h < len(V) else None
    def exit_arksell(e):
        nxt = [s for s in sell_locs if s > e]
        if not nxt: return None
        return nxt[0] if nxt[0]-e < 504 else None

    rows = []
    for e in sig:
        rows.append({
            'i': e, 'date': px.index[e], 'px': V[e],
            'rsi': RSI[e], 'dd': DD[e],
            'ma200': (V[e]/MA200[e]-1)*100 if not np.isnan(MA200[e]) else np.nan,
            'rsi_exit': exit_rsi(e), 'm3': exit_fixed(e, 63),
            'm6': exit_fixed(e, 126), 'ark': exit_arksell(e)})
    def ret(e, x):
        return None if x is None else V[x]/V[e]-1

    print(f"신호 {len(rows)}건 · 현재 상태: RSI {RSI[-1]:.1f} / 낙폭 {DD[-1]:.0f}% / "
          f"200일선 대비 {(V[-1]/MA200[-1]-1)*100:+.0f}%\n")

    print("=" * 100)
    print("진입 시점 RSI 구간별 (낮을수록 '빠졌을 때 산 것')")
    print("=" * 100)
    bins = [(0,35,'RSI<35 (과매도)'),(35,45,'RSI 35~45'),(45,55,'RSI 45~55'),(55,100,'RSI 55↑ (오름세)')]
    print(f"{'구간':<20s}{'n':>4s}" + "".join(f"{c:>18s}" for c in
          ('RSI이탈 청산','3개월 보유','6개월 보유','아크매도 청산')))
    for lo,hi,lab in bins:
        sub=[r_ for r_ in rows if lo<=r_['rsi']<hi]
        if not sub: continue
        cells=''
        for key in ('rsi_exit','m3','m6','ark'):
            rr=[ret(r_['i'],r_[key]) for r_ in sub]
            rr=[x for x in rr if x is not None]
            cells += (f"{np.mean(rr)*100:>+11.1f}% ({len(rr)})" if rr else f"{'표본없음':>18s}")
        print(f"{lab:<20s}{len(sub):>4d}"+cells)

    print()
    print("=" * 100)
    print("진입 시점 낙폭 구간별")
    print("=" * 100)
    dbins=[(-100,-35,'낙폭 -35%↓'),(-35,-20,'낙폭 -35~-20%'),(-20,-10,'낙폭 -20~-10%'),(-10,1,'낙폭 -10%↑')]
    print(f"{'구간':<20s}{'n':>4s}" + "".join(f"{c:>18s}" for c in
          ('RSI이탈 청산','3개월 보유','6개월 보유','아크매도 청산')))
    for lo,hi,lab in dbins:
        sub=[r_ for r_ in rows if lo<=r_['dd']<hi]
        if not sub: continue
        cells=''
        for key in ('rsi_exit','m3','m6','ark'):
            rr=[ret(r_['i'],r_[key]) for r_ in sub]
            rr=[x for x in rr if x is not None]
            cells += (f"{np.mean(rr)*100:>+11.1f}% ({len(rr)})" if rr else f"{'표본없음':>18s}")
        print(f"{lab:<20s}{len(sub):>4d}"+cells)

    print()
    print("=" * 100)
    print("지금과 가장 비슷했던 신호 (RSI<40 이면서 200일선 아래)")
    print("=" * 100)
    sim=[r_ for r_ in rows if r_['rsi']<40 and r_['ma200']<0]
    print(f"  {'진입일':<12s}{'진입가':>9s}{'RSI':>6s}{'낙폭':>7s}{'200일선':>8s}"
          f"{'RSI청산':>10s}{'3개월':>9s}{'6개월':>9s}")
    for r_ in sim:
        f=lambda k: (f"{ret(r_['i'],r_[k])*100:>+8.1f}%" if r_[k] is not None else f"{'보유중':>9s}")
        print(f"  {r_['date']:%Y-%m-%d}${r_['px']:>8.2f}{r_['rsi']:>6.1f}{r_['dd']:>6.0f}%"
              f"{r_['ma200']:>7.0f}%{f('rsi_exit'):>10s}{f('m3'):>9s}{f('m6'):>9s}")
    for key,lab in (('rsi_exit','RSI 이탈'),('m3','3개월'),('m6','6개월'),('ark','아크매도')):
        rr=[ret(r_['i'],r_[key]) for r_ in sim]
        rr=[x for x in rr if x is not None]
        if rr: print(f"    {lab:<10s} n={len(rr)} 평균 {np.mean(rr)*100:+.1f}% "
                     f"승률 {np.mean([x>0 for x in rr])*100:.0f}% 최악 {min(rr)*100:+.1f}%")

if __name__ == '__main__':
    main()
