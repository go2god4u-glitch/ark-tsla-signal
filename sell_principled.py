"""원칙 있는 매도 규칙 탐색 — 아크 보유량 감소를 축으로.

'6개월 고정' 은 성적은 괜찮지만 원칙이 없다. 진입 규칙과 대칭인 청산 규칙을 찾는다.

  진입: 아크가 많이 샀고(순매수율 상위 10%) 낙폭이 깊을 때(-30% 이하)
  청산: 아크가 그 포지션을 줄였거나(보유량 감소) 낙폭이 회복됐을 때

핵심 지표 두 가지 — 둘 다 진입 조건의 거울상이다:

  아크 보유량 감소  진입 시점 합산 보유 대비 X% 줄면 청산.
                    '이번 주에 팔았나'(잡음이 많다)가 아니라
                    '진입 이후 누적으로 빠져나갔나'를 본다.
  낙폭 회복        진입 조건이 '낙폭 -30% 이하'였으니
                    '낙폭이 -Y% 까지 회복하면' 이 자연스러운 대칭이다.

상한은 안전장치로만 둔다(무한 보유 방지). 규칙이 걸리지 않는 경우를 위한 것이다.
"""
import json
import numpy as np
import pandas as pd

BASELINE = 2290      # 126일 고정 보유 (비교 기준)

def main():
    from signal_check import build, build_daily
    r = json.load(open('data/tsla_full.json'))['chart']['result'][0]
    idx = pd.to_datetime(pd.Series(r['timestamp']), unit='s').dt.normalize()
    px = pd.Series(r['indicators']['quote'][0]['close'], index=idx).dropna().sort_index()
    V = px.values
    DD = (px / px.rolling(252, min_periods=60).max() - 1).values * 100
    MA200 = px.rolling(200).mean().values

    # 아크 일별 합산 보유 -> 가격 인덱스에 맞춰 채운다
    daily = build_daily(px)
    H = daily['shares'].reindex(px.index).ffill().values

    w = build(px)
    buys = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                   if px.index.searchsorted(t, side='right') < len(px)})

    def evaluate(f, cap=504):
        rs, days, live, why = [], [], 0, []
        for e in buys:
            j = None
            for k in range(e+1, min(e+cap+1, len(V))):
                w_ = f(e, k)
                if w_: j, wh = k, w_; break
            if j is None:
                j = min(e+cap, len(V)-1)
                wh = '상한' if e+cap < len(V) else '보유중'
                if e+cap >= len(V): live += 1
            rs.append(V[j]/V[e]-1); days.append(j-e); why.append(wh)
        a = np.array(rs)
        return {'mean': a.mean(), 'hit': (a>0).mean(), 'worst': a.min(),
                'value': sum(100*(1+x) for x in a), 'days': np.mean(days),
                'live': live, 'why': why}

    def show(name, s, mark=''):
        d = s['value'] - BASELINE
        print(f"  {name:<34s} 평균 {s['mean']*100:>+6.1f}% 승률 {s['hit']*100:>3.0f}% "
              f"최악 {s['worst']*100:>+6.1f}% 보유 {s['days']:>4.0f}일 "
              f"평가 {s['value']:>6.0f} ({d:>+5.0f}){mark}")

    print(f"기준: 126일 고정 보유 = {BASELINE}\n")

    print("=" * 104)
    print("A. 아크 보유량 감소 — 진입 시점 대비 X% 줄면 청산")
    print("=" * 104)
    for drop in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        show(f"아크 보유 -{int(drop*100)}%",
             evaluate(lambda e,k,d=drop: ('아크감소' if H[k] <= H[e]*(1-d) else None)))

    print()
    print("=" * 104)
    print("B. 낙폭 회복 — 낙폭이 -Y% 까지 올라오면 청산 (진입 조건의 거울상)")
    print("=" * 104)
    for y in (-25, -20, -15, -10, -5, 0):
        show(f"낙폭 {y}% 회복",
             evaluate(lambda e,k,y=y: ('낙폭회복' if DD[k] >= y else None)))

    print()
    print("=" * 104)
    print("C. 아크 감소 OR 낙폭 회복 (먼저 오는 쪽)")
    print("=" * 104)
    best=None
    for drop in (0.10, 0.15, 0.20):
        for y in (-20, -15, -10):
            s = evaluate(lambda e,k,d=drop,y=y:
                         ('아크감소' if H[k] <= H[e]*(1-d) else
                          '낙폭회복' if DD[k] >= y else None))
            m = ' ★' if s['value'] > BASELINE else ''
            show(f"아크 -{int(drop*100)}% OR 낙폭 {y}%", s, m)
            if best is None or s['value'] > best[1]['value']: best=(f"아크 -{int(drop*100)}% OR 낙폭 {y}%", s)

    print()
    print("=" * 104)
    print("D. 아크 감소 AND 낙폭 회복 (둘 다 만족)")
    print("=" * 104)
    for drop in (0.10, 0.15, 0.20):
        for y in (-25, -20, -15):
            s = evaluate(lambda e,k,d=drop,y=y:
                         ('둘다' if (H[k] <= H[e]*(1-d) and DD[k] >= y) else None))
            show(f"아크 -{int(drop*100)}% AND 낙폭 {y}%", s, ' ★' if s['value']>BASELINE else '')

    print()
    print("=" * 104)
    print("E. 아크 감소 OR 200일선 회복 (4번 조합)")
    print("=" * 104)
    for drop in (0.10, 0.15, 0.20):
        s = evaluate(lambda e,k,d=drop:
                     ('아크감소' if H[k] <= H[e]*(1-d) else
                      '200일선' if (not np.isnan(MA200[k]) and V[k] > MA200[k]) else None))
        show(f"아크 -{int(drop*100)}% OR 200일선 상회", s, ' ★' if s['value']>BASELINE else '')

    print()
    print("=" * 104)
    print(f"최고 조합 상세: {best[0]}")
    print("=" * 104)
    from collections import Counter
    print(f"  청산 사유 분포: {dict(Counter(best[1]['why']))}")

if __name__ == '__main__':
    main()
