"""아크 대량매도가 하락 신호인가 — 절대 규모 기준으로 검증.

앞서 시험한 것은 '하위 10% 분위수' 였다. 그것은 상대 기준이라
평범한 매도도 신호로 잡는다. 여기서는 **절대적 규모**로 본다.

매도 신호가 유효하려면 이후 수익률이 **음수**여야 한다(팔고 나서 떨어져야 잘 판 것).
"""
import json
import numpy as np
import pandas as pd

GAP = 30
RNG = np.random.default_rng(20260731)
NB = 20000

def episodes(locs, gap=GAP):
    locs = sorted({int(i) for i in locs})
    out, prev = [], -10**9
    for i in locs:
        if i - prev > gap:
            out.append(i)
        prev = i
    return out

def main():
    from signal_check import build
    r = json.load(open('data/tsla_full.json'))['chart']['result'][0]
    idx = pd.to_datetime(pd.Series(r['timestamp']), unit='s').dt.normalize()
    px = pd.Series(r['indicators']['quote'][0]['close'], index=idx).dropna().sort_index()
    V = px.values
    w = build(px)
    lo = px.index.searchsorted(w.index[0])

    def ev(weeks, h):
        locs = episodes([px.index.searchsorted(t, side='right') for t in weeks
                         if px.index.searchsorted(t, side='right') < len(px)])
        done = [i for i in locs if i + h < len(V) and i >= lo]
        if len(done) < 3:
            return None
        rr = np.array([V[i+h]/V[i]-1 for i in done])
        pool = np.arange(lo, len(V)-h)
        b = V[pool+h]/V[pool]-1
        sims = np.array([b[RNG.integers(0,len(b),len(rr))].mean() for _ in range(NB)])
        return {'n': len(rr), 'eps': len(locs), 'mean': rr.mean(),
                'edge': rr.mean()-b.mean(), 'down': (rr<0).mean(),
                'p': float((sims <= rr.mean()).mean())}

    print("=" * 96)
    print("순매수율 임계별 — '이만큼 팔면 이후 떨어지는가'")
    print("=" * 96)
    print(f"{'임계':<14s}{'사건':>5s}" + "".join(f"{l:>12s}{'하락률':>8s}{'p':>8s}"
                                              for l in ('3개월 초과','6개월 초과')))
    print("-" * 96)
    rows = []
    for cut in (-3, -5, -7, -10, -12, -15):
        weeks = w.index[w['netpct'] <= cut]
        cells, rec = '', {'cut': cut}
        for h, lab in [(63,'3개월'), (126,'6개월')]:
            s = ev(weeks, h)
            if s:
                cells += f"{s['edge']*100:>+11.1f}%{s['down']*100:>7.0f}%{s['p']:>8.3f}"
                rec[lab] = s
            else:
                cells += f"{'표본부족':>12s}{'':>8s}{'':>8s}"
        e = rec.get('3개월') or rec.get('6개월')
        print(f"{f'순매수율 ≤{cut}%':<14s}{e['eps'] if e else 0:>5d}" + cells)
        rows.append(rec)

    print()
    print("=" * 96)
    print("절대 주식수 기준 (규모가 큰 매도만)")
    print("=" * 96)
    print(f"{'임계':<16s}{'사건':>5s}" + "".join(f"{l:>12s}{'하락률':>8s}{'p':>8s}"
                                              for l in ('3개월 초과','6개월 초과')))
    print("-" * 96)
    for cut in (-200_000, -300_000, -400_000, -500_000):
        weeks = w.index[w['net'] <= cut]
        cells = ''
        e0 = None
        for h in (63, 126):
            s = ev(weeks, h)
            if s:
                cells += f"{s['edge']*100:>+11.1f}%{s['down']*100:>7.0f}%{s['p']:>8.3f}"
                e0 = e0 or s
            else:
                cells += f"{'표본부족':>12s}{'':>8s}{'':>8s}"
        print(f"{f'{cut:,}주 이하':<16s}{e0['eps'] if e0 else 0:>5d}" + cells)

    print()
    print("=" * 96)
    print("참고: 매수 신호(비교용)")
    print("=" * 96)
    s = ev(w[w['sig']].index, 126)
    print(f"  매수 신호        사건 {s['eps']}  6개월 초과 {s['edge']*100:+.1f}%  "
          f"하락률 {s['down']*100:.0f}%")

if __name__ == '__main__':
    main()
