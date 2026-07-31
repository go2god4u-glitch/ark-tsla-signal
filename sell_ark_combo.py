"""아크 매도를 매도 규칙에 넣으면 나아지는가 — 조합을 전부 훑는다.

앞서 '아크 매도 단독 매도' 은 0.91배로 최악이었다. 하지만 조합은 시험하지 않았다.
아크 매도가 단독으로는 무의미해도 다른 조건과 겹칠 때 값어치가 있을 수 있다.

판단 기준은 앞서와 같다: 최고값이 아니라 **파라미터 구간 전체**.
기준선은 확정 규칙(RSI 70 이탈 / 126거래일 상한)이다.

아크 매도 신호 정의: 주간 순매수율이 그 시점까지의 하위 q 분위 이하 (확장창, 1주 시프트).
q 를 바꿔가며 훑는다.
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
    V, RSI = px.values, rsi14(px).values
    w = build(px)
    buys = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                   if px.index.searchsorted(t, side='right') < len(px)})

    def ark_sell_locs(q):
        thr = w['netpct'].expanding(52).quantile(q).shift(1)
        wk = w.index[(w['netpct'] <= thr).fillna(False)]
        return sorted({px.index.searchsorted(t, side='right') for t in wk
                       if px.index.searchsorted(t, side='right') < len(px)})

    def run(exit_fn):
        """신호마다 독립 포지션. 총자본 100 씩."""
        rets, opens = [], []
        for e in buys:
            j = exit_fn(e)
            if j is None:
                opens.append(V[-1]/V[e]-1)
            else:
                rets.append(V[j]/V[e]-1)
        allr = rets + opens
        if not allr: return None
        return {'done': len(rets), 'open': len(opens),
                'mean_done': np.mean(rets) if rets else None,
                'hit_done': np.mean([x>0 for x in rets]) if rets else None,
                'mean_all': np.mean(allr), 'hit_all': np.mean([x>0 for x in allr]),
                'worst': min(allr),
                'value': sum(100*(1+x) for x in allr), 'n': len(allr)}

    def show(name, s):
        if not s: print(f"  {name:<34s} 표본부족"); return
        print(f"  {name:<34s} 완료 {s['done']:>2d}/{s['n']:<2d} "
              f"완료평균 {(s['mean_done'] or 0)*100:>+6.1f}% "
              f"전체평균 {s['mean_all']*100:>+6.1f}% 승률 {s['hit_all']*100:>3.0f}% "
              f"최악 {s['worst']*100:>+6.1f}% 평가 {s['value']:.0f}")

    def rsi_exit(e):
        armed = False
        for j in range(e+1, len(V)):
            if RSI[j] >= RSI_LEVEL: armed = True
            if armed and RSI[j] < RSI_LEVEL: return j
            if j-e >= MAX_HOLD: return j
        return None

    print("=" * 104)
    print("기준선")
    print("=" * 104)
    show("RSI 70 이탈 + 126일 상한 (확정)", run(rsi_exit))

    print()
    print("=" * 104)
    print("A. 아크 매도 단독 (분위수를 훑는다)")
    print("=" * 104)
    for q in (0.05, 0.10, 0.15, 0.20, 0.30):
        S = ark_sell_locs(q)
        def f(e, S=S):
            nx = [s for s in S if s > e]
            j = nx[0] if nx else None
            return min(j, e+MAX_HOLD) if j is not None else (e+MAX_HOLD if e+MAX_HOLD < len(V) else None)
        show(f"아크 매도 하위 {int(q*100)}% + 상한", run(f))

    print()
    print("=" * 104)
    print("B. 아크 매도 OR RSI 이탈 (먼저 오는 쪽)")
    print("=" * 104)
    for q in (0.05, 0.10, 0.15, 0.20, 0.30):
        S = ark_sell_locs(q)
        def f(e, S=S):
            a = rsi_exit(e)
            nx = [s for s in S if s > e]
            b = nx[0] if nx else None
            c = [x for x in (a, b) if x is not None]
            return min(c) if c else None
        show(f"아크 하위 {int(q*100)}% OR RSI", run(f))

    print()
    print("=" * 104)
    print("C. 아크 매도 AND RSI 과열 (둘 다 만족해야 매도)")
    print("=" * 104)
    for q in (0.10, 0.20, 0.30):
        for lvl in (55, 60, 65):
            S = set(ark_sell_locs(q))
            def f(e, S=S, lvl=lvl):
                for j in range(e+1, len(V)):
                    if j in S and RSI[j] >= lvl: return j
                    if j-e >= MAX_HOLD: return j
                return None
            show(f"아크 하위 {int(q*100)}% AND RSI≥{lvl}", run(f))

    print()
    print("=" * 104)
    print("D. 아크 매도가 오면 매도하되, 손실 중이면 무시 (이익 실현만)")
    print("=" * 104)
    for q in (0.10, 0.20, 0.30):
        S = set(ark_sell_locs(q))
        def f(e, S=S):
            for j in range(e+1, len(V)):
                if j in S and V[j] > V[e]: return j
                if j-e >= MAX_HOLD: return j
            return None
        show(f"아크 하위 {int(q*100)}% (이익 중일 때만)", run(f))

    print()
    print("=" * 104)
    print("E. RSI 이탈 우선, 아크 매도는 보조 (RSI 미도달 시에만 아크로 매도)")
    print("=" * 104)
    for q in (0.10, 0.20, 0.30):
        S = sorted(ark_sell_locs(q))
        def f(e, S=S):
            armed = False
            for j in range(e+1, len(V)):
                if RSI[j] >= RSI_LEVEL: armed = True
                if armed and RSI[j] < RSI_LEVEL: return j
                if not armed and j in S and V[j] > V[e]: return j
                if j-e >= MAX_HOLD: return j
            return None
        show(f"RSI 우선 / 미도달 시 아크 하위 {int(q*100)}%", run(f))

if __name__ == '__main__':
    main()
