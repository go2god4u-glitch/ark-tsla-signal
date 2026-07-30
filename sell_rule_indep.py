"""매도 규칙 재선정 — 신호마다 100 넣는 '독립 포지션' 틀에서.

왜 다시 고르나:
  매도 규칙(RSI 70 이탈)은 '단일 포지션 복리' 틀에서 골랐다. 그 틀은 겹치는
  신호를 하나로 병합하므로 매매가 5건뿐이었다.
  그 뒤 자본 모델을 '신호마다 독립 100' 으로 바꿨는데 매도 규칙은 다시 고르지 않았다.
  틀이 바뀌면 순위가 바뀐다 — 실제로 뒤집혔다.

판단 기준은 같다: 최고값이 아니라 파라미터 구간 전체, 그리고 최악 사례.
"""
import json
import numpy as np
import pandas as pd

MAXCAP = 504

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
    V, RSI = px.values, rsi14(px).values
    w = build(px)
    buys = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                   if px.index.searchsorted(t, side='right') < len(px)})

    def stats(f):
        rs, live = [], 0
        for e in buys:
            j = f(e)
            if j is None or j >= len(V):
                rs.append(V[-1]/V[e]-1); live += 1
            else:
                rs.append(V[j]/V[e]-1)
        a = np.array(rs)
        days = [min(f(e) or len(V)-1, len(V)-1)-e for e in buys]
        return {'mean': a.mean(), 'hit': (a>0).mean(), 'worst': a.min(),
                'value': sum(100*(1+x) for x in a), 'days': np.mean(days), 'live': live}

    def rsi_cross(lvl, cap):
        def f(e):
            armed = False
            for j in range(e+1, len(V)):
                if RSI[j] >= lvl: armed = True
                if armed and RSI[j] < lvl: return j
                if j-e >= cap: return j
            return None
        return f

    fams = {}
    fams['고정 보유'] = [(f"{m}개월", (lambda h: (lambda e: e+h if e+h < len(V) else None))(m*21))
                       for m in (2,3,4,5,6,7,8,9,10,12)]
    fams['RSI 이탈 + 6개월 상한'] = [(f"RSI {l}", rsi_cross(l, 126)) for l in (55,60,65,70,75,80)]
    fams['RSI 이탈 + 9개월 상한'] = [(f"RSI {l}", rsi_cross(l, 189)) for l in (55,60,65,70,75,80)]
    def trail(p, cap):
        def f(e):
            peak = V[e]
            for j in range(e+1, len(V)):
                peak = max(peak, V[j])
                if V[j] <= peak*(1-p) or j-e >= cap: return j
            return None
        return f
    fams['트레일링 스톱 + 6개월'] = [(f"-{int(p*100)}%", trail(p,126)) for p in (.15,.20,.25,.30,.35)]

    print("규칙군별 — 파라미터 구간 전체 (신호마다 100, 총 16건)")
    print(f"{'규칙군':<24s}{'중앙값':>8s}{'최솟값':>8s}{'최댓값':>8s}   최고 파라미터")
    print("-"*76)
    rows=[]
    for fam, items in fams.items():
        vs = {n: stats(f) for n, f in items}
        vals = np.array([v['value'] for v in vs.values()])
        best = max(vs, key=lambda k: vs[k]['value'])
        print(f"{fam:<24s}{np.median(vals):>8.0f}{vals.min():>8.0f}{vals.max():>8.0f}   "
              f"{best} ({vs[best]['value']:.0f})")
        rows.append((fam, vs))
    print()
    for fam, vs in rows:
        print(f"[{fam}]")
        for n, v in vs.items():
            print(f"  {n:<10s} 평균 {v['mean']*100:>+6.1f}% 승률 {v['hit']*100:>3.0f}% "
                  f"최악 {v['worst']*100:>+6.1f}% 평균보유 {v['days']:>4.0f}일 평가 {v['value']:>6.0f}"
                  + (f"  (보유중 {v['live']})" if v['live'] else ""))
        print()

if __name__ == '__main__':
    main()
