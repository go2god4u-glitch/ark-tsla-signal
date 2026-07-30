"""대량매도를 여러 각도로 끝까지 검증한다.

앞서 '순매수율 ≤ -7%' 하나로 답을 냈다. 그것으로 끝내면 규칙 15 위반이다.
안 해본 각도를 전부 훑는다.
"""
import json
import numpy as np
import pandas as pd

GAP = 30
RNG = np.random.default_rng(20260731)
NB = 12000


def episodes(locs, gap=GAP):
    locs = sorted({int(i) for i in locs})
    out, prev = [], -10 ** 9
    for i in locs:
        if i - prev > gap:
            out.append(i)
        prev = i
    return out


def main():
    from signal_check import build, build_daily
    r = json.load(open('data/tsla_full.json'))['chart']['result'][0]
    idx = pd.to_datetime(pd.Series(r['timestamp']), unit='s').dt.normalize()
    px = pd.Series(r['indicators']['quote'][0]['close'], index=idx).dropna().sort_index()
    V = px.values
    w = build(px)
    lo = px.index.searchsorted(w.index[0])
    DD = (px / px.rolling(252, min_periods=60).max() - 1).values * 100
    daily = build_daily(px)
    wide = daily.attrs["wide"]

    def ev(weeks, h, label=''):
        locs = episodes([px.index.searchsorted(t, side='right') for t in weeks
                         if px.index.searchsorted(t, side='right') < len(px)])
        done = [i for i in locs if i + h < len(V) and i >= lo]
        if len(done) < 3:
            return None
        rr = np.array([V[i + h] / V[i] - 1 for i in done])
        pool = np.arange(lo, len(V) - h)
        b = V[pool + h] / V[pool] - 1
        sims = np.array([b[RNG.integers(0, len(b), len(rr))].mean() for _ in range(NB)])
        return {'n': len(rr), 'eps': len(locs), 'mean': float(rr.mean()),
                'edge': float(rr.mean() - b.mean()), 'down': float((rr < 0).mean()),
                'p': float((sims <= rr.mean()).mean()), 'worst': float(rr.max())}

    def row(name, s, w_=30):
        if not s:
            print(f"  {name:<{w_}s} 표본부족")
            return
        print(f"  {name:<{w_}s} n={s['n']:>2d} 초과 {s['edge']*100:>+7.1f}% "
              f"하락률 {s['down']*100:>3.0f}% p={s['p']:.3f}")

    print("=" * 92)
    print("A. 기간(horizon)별 — 하락이 언제 나타나고 언제 사라지나")
    print("=" * 92)
    weeks7 = w.index[w['netpct'] <= -7]
    for h, lab in [(5, '1주'), (21, '1개월'), (42, '2개월'), (63, '3개월'),
                   (126, '6개월'), (189, '9개월'), (252, '12개월')]:
        row(lab, ev(weeks7, h), 10)

    print()
    print("=" * 92)
    print("B. 매도 시점의 상태별 — 어떤 매도가 더 나쁜가")
    print("=" * 92)
    net = w['netpct']
    ddw = pd.Series(DD, index=px.index).reindex(w.index, method='ffill')
    for name, cond in [
        ("낙폭 얕을 때(-20% 위) 매도", (net <= -7) & (ddw > -20)),
        ("낙폭 깊을 때(-20% 이하) 매도", (net <= -7) & (ddw <= -20)),
        ("직전 4주 상승 중 매도", (net <= -7) & (w['shares'].pct_change(4) > 0)),
        ("직전 4주 하락 중 매도", (net <= -7) & (w['shares'].pct_change(4) <= 0)),
    ]:
        row(name, ev(w.index[cond.fillna(False)], 63))

    print()
    print("=" * 92)
    print("C. 몇 개 펀드가 동시에 팔았나 — 전 펀드 매도가 더 강한 신호인가")
    print("=" * 92)
    wk = wide.resample('W-FRI').last()
    fw = wk.pct_change() * 100
    core = [c for c in ('ARKK', 'ARKQ', 'ARKW') if c in fw.columns]
    nsell = (fw[core] <= -3).sum(axis=1).reindex(w.index)
    for k in (1, 2, 3):
        row(f"{k}개 이상 펀드가 -3% 이상 매도", ev(w.index[(nsell >= k).fillna(False)], 63))

    print()
    print("=" * 92)
    print("D. 연속 매도 — 2주 이상 이어지면 더 나쁜가")
    print("=" * 92)
    consec = (net <= -3) & (net.shift(1) <= -3)
    row("2주 연속 -3% 이상 매도", ev(w.index[consec.fillna(False)], 63))
    row("단발 -7% 매도(연속 아님)",
        ev(w.index[((net <= -7) & ~consec).fillna(False)], 63))

    print()
    print("=" * 92)
    print("E. 매수 신호와 겹칠 때 — 무엇이 이기나")
    print("=" * 92)
    sig_after = []
    for t in w.index[(net <= -7).fillna(False)]:
        nxt = w.index[(w.index > t) & (w.index <= t + pd.Timedelta(days=90)) & w['sig']]
        if len(nxt):
            sig_after.append((t, nxt[0]))
    print(f"  대량매도 후 90일 내 매수 신호 발생: {len(sig_after)}건")
    for t, n in sig_after:
        k = px.index.searchsorted(n, side='right')
        if k + 126 < len(V):
            print(f"    매도 {t:%Y-%m-%d} -> 매수신호 {n:%Y-%m-%d} "
                  f"-> 6개월 {(V[k+126]/V[k]-1)*100:+.1f}%")

    print()
    print("=" * 92)
    print("F. 되돌림 — 대량매도 후 아크가 다시 사면?")
    print("=" * 92)
    rebuy = []
    for t in w.index[(net <= -7).fillna(False)]:
        fut = w[(w.index > t) & (w.index <= t + pd.Timedelta(days=60))]
        if len(fut) and fut['netpct'].max() >= 3:
            rebuy.append(fut.index[fut['netpct'].argmax()])
    row("매도 후 60일 내 재매수(+3%) 시점", ev(pd.DatetimeIndex(rebuy), 126))

if __name__ == '__main__':
    main()
