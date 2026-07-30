"""매도 규칙 병렬화(A+B) 이후 전면 재백테스트.

규칙이 바뀌면 그 위에서 고른 모든 결론을 다시 검증해야 한다(CLAUDE.md 규칙 4).
매수 필터·진입 방식·자본 배분·조건부 수익률을 새 청산 규칙 위에서 다시 돌린다.
"""
import json
import numpy as np
import pandas as pd

ARK_DROP, DD_RECOVER, MAX_HOLD, BIG = 0.20, -20.0, 504, -7.0


def load():
    from signal_check import build, build_daily
    r = json.load(open('data/tsla_full.json'))['chart']['result'][0]
    idx = pd.to_datetime(pd.Series(r['timestamp']), unit='s').dt.normalize()
    px = pd.Series(r['indicators']['quote'][0]['close'], index=idx).dropna().sort_index()
    daily = build_daily(px)
    w = build(px)
    return px, w, daily


def main():
    px, w, daily = load()
    V = px.values
    DD = (px / px.rolling(252, min_periods=60).max() - 1).values * 100
    H = daily['shares'].reindex(px.index).ffill().values
    RSIu = px.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    RSId = (-px.diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    RSI = (100 - 100/(1 + RSIu/RSId)).values
    MA200 = px.rolling(200).mean().values

    bigs = sorted({px.index.searchsorted(t, side='right')
                   for t in w.index[(w['netpct'] <= BIG).fillna(False)]
                   if px.index.searchsorted(t, side='right') < len(px)})

    def exit_of(e):
        exA = None
        for j in range(e + 1, min(e + MAX_HOLD + 1, len(V))):
            if H[j] <= H[e] * (1 - ARK_DROP) and DD[j] >= DD_RECOVER:
                exA = j
                break
        nx = [b for b in bigs if b > e and b <= e + MAX_HOLD]
        exB = nx[0] if nx else None
        c = [(x, t) for x, t in ((exA, 'A'), (exB, 'B')) if x is not None]
        if c:
            return min(c)
        return (min(e + MAX_HOLD, len(V) - 1), '상한') if e + MAX_HOLD < len(V) else (None, None)

    def stat(locs):
        rs, ws = [], []
        for e in locs:
            j, t = exit_of(e)
            if j is None:
                rs.append(V[-1]/V[e]-1); ws.append('보유중')
            else:
                rs.append(V[j]/V[e]-1); ws.append(t)
        if not rs:
            return None
        a = np.array(rs)
        return {'n': len(a), 'mean': a.mean(), 'hit': (a > 0).mean(),
                'worst': a.min(), 'value': sum(100*(1+x) for x in a)}

    sig_locs = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                       if px.index.searchsorted(t, side='right') < len(px)})
    base = stat(sig_locs)
    print(f"현재 규칙: 신호 {base['n']}건 · 평균 {base['mean']*100:+.1f}% · "
          f"승률 {base['hit']*100:.0f}% · 최악 {base['worst']*100:+.1f}% · 평가 {base['value']:.0f}\n")

    # --- 1. 매수 낙폭 필터를 다시 고른다 ---
    print("=" * 78)
    print("1. 매수 낙폭 필터 재검증 (청산 규칙이 바뀌었으므로)")
    print("=" * 78)
    netpct, thr = w['netpct'], w['thr_pct']
    ddw = pd.Series(DD, index=px.index).reindex(w.index, method='ffill')
    print(f"  {'문턱':<10s}{'신호':>5s}{'평균':>9s}{'승률':>6s}{'최악':>9s}{'평가':>8s}")
    for cut in (0, -10, -20, -25, -30, -35, -40):
        weeks = w.index[((netpct >= thr) & (ddw <= cut)).fillna(False)]
        locs = sorted({px.index.searchsorted(t, side='right') for t in weeks
                       if px.index.searchsorted(t, side='right') < len(px)})
        s = stat(locs)
        if not s:
            continue
        lab = '없음' if cut == 0 else f'≤{cut}%'
        mark = ' <- 채택' if cut == -30 else ''
        print(f"  {lab:<10s}{s['n']:>5d}{s['mean']*100:>+8.1f}%{s['hit']*100:>5.0f}%"
              f"{s['worst']*100:>+8.1f}%{s['value']:>8.0f}{mark}")

    # --- 2. 진입 시점 상태별 ---
    print()
    print("=" * 78)
    print("2. 진입 시점 상태별 수익률")
    print("=" * 78)
    for name, f in [("RSI < 35", lambda e: RSI[e] < 35),
                    ("RSI 35~50", lambda e: 35 <= RSI[e] < 50),
                    ("RSI ≥ 50", lambda e: RSI[e] >= 50),
                    ("낙폭 ≤ -45%", lambda e: DD[e] <= -45),
                    ("낙폭 -45~-35%", lambda e: -45 < DD[e] <= -35),
                    ("낙폭 > -35%", lambda e: DD[e] > -35),
                    ("200일선 아래", lambda e: V[e] < MA200[e])]:
        s = stat([e for e in sig_locs if f(e)])
        if s:
            print(f"  {name:<16s} n={s['n']:>2d}  평균 {s['mean']*100:>+7.1f}%  "
                  f"승률 {s['hit']*100:>3.0f}%  최악 {s['worst']*100:>+6.1f}%")

    # --- 3. 국면 내 순서별 ---
    print()
    print("=" * 78)
    print("3. 국면 내 순서별 (첫 신호가 더 나은가)")
    print("=" * 78)
    runs, cur = [], [sig_locs[0]]
    for i in sig_locs[1:]:
        if i - cur[-1] <= 30:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    for pos in (1, 2, 3):
        s = stat([r[pos-1] for r in runs if len(r) >= pos])
        if s:
            print(f"  {pos}번째 신호      n={s['n']:>2d}  평균 {s['mean']*100:>+7.1f}%  "
                  f"승률 {s['hit']*100:>3.0f}%")

    # --- 4. 자본 배분 ---
    print()
    print("=" * 78)
    print("4. 자본 배분 (총자본 100 기준)")
    print("=" * 78)
    for name, alloc in [("일괄 100%", lambda k, c: c if k == 0 else 0.0),
                        ("균등 2회", lambda k, c: 50.0 if k < 2 else 0.0),
                        ("균등 3회", lambda k, c: 100/3 if k < 3 else 0.0),
                        ("잔액 1/2", lambda k, c: c * 0.5)]:
        done, cash, sh, buys_, first, holding = [], 100.0, 0.0, [], None, False
        for i in range(len(V)):
            if not holding:
                if i in set(sig_locs):
                    holding, cash, sh, buys_, first = True, 100.0, 0.0, [], i
                    amt = min(alloc(0, cash), cash); cash -= amt; sh += amt/V[i]; buys_.append(i)
                continue
            if i in set(sig_locs):
                amt = min(alloc(len(buys_), cash), cash)
                if amt > 0:
                    cash -= amt; sh += amt/V[i]; buys_.append(i)
            j, t = exit_of(first)
            if j is not None and i >= j:
                done.append((cash + sh*V[j])/100 - 1); holding = False
        if done:
            a = np.array(done)
            print(f"  {name:<12s} {len(a)}사이클  평균 {a.mean()*100:>+7.1f}%  "
                  f"누적 {np.prod(1+a):>5.2f}x")


if __name__ == '__main__':
    main()
