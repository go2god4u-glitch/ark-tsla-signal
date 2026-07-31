"""신호마다 독립 포지션 — 매 신호에 100 을 넣고, 각자의 매도 시계를 돈다.

자본 배분 문제를 없앤 방식이다. 신호가 뜰 때마다 새로 100 을 투입하고,
그 포지션은 자기 진입일 기준으로 매도 조건을 본다. 이미 보유 중이어도 상관없다.

  진입  아크 주간 신호 -> 다음 거래일 종가, 100 투입
  매도  그 포지션 진입 이후 RSI(14) 가 70 을 넘었다가 다시 70 아래로
        (진입일로부터 126거래일 안에 안 걸리면 그때 정리)

'70 을 넘은 적이 있는가'(armed)는 포지션마다 따로 센다. 같은 날 여러 포지션이
동시에 매도될 수 있다.

주의: 이 방식은 동시에 여러 포지션을 들 수 있으므로 필요 자본이 가변이다.
      수익률은 '포지션 하나당' 이지 '계좌 전체' 가 아니다.
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
    sig = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                  if px.index.searchsorted(t, side='right') < len(px)})

    rows = []
    for e in sig:
        armed, exit_i, why = False, None, None
        for j in range(e+1, len(V)):
            if RSI[j] >= RSI_LEVEL: armed = True
            if armed and RSI[j] < RSI_LEVEL: exit_i, why = j, 'RSI 이탈'; break
            if j - e >= MAX_HOLD: exit_i, why = j, '6개월 상한'; break
        rows.append({'entry': e, 'exit': exit_i, 'why': why,
                     'ret': (V[exit_i]/V[e]-1) if exit_i else (V[-1]/V[e]-1),
                     'open': exit_i is None})

    print(f"신호 {len(rows)}건 — 각각 100 투입, 각자의 매도 시계\n")
    print(f"  {'진입':<12s}{'진입가':>9s}{'매도':<14s}{'매도가':>9s}{'보유':>6s}{'수익':>9s}  사유")
    for r_ in rows:
        e = r_['entry']
        if r_['open']:
            print(f"  {px.index[e]:%Y-%m-%d}${V[e]:>8.2f}  {'보유중':<12s}{V[-1]:>9.2f}"
                  f"{len(V)-1-e:>5d}일{r_['ret']*100:>+8.1f}%  (평가)")
        else:
            x = r_['exit']
            print(f"  {px.index[e]:%Y-%m-%d}${V[e]:>8.2f}  {px.index[x]:%Y-%m-%d}  "
                  f"${V[x]:>8.2f}{x-e:>5d}일{r_['ret']*100:>+8.1f}%  {r_['why']}")

    done = [r_ for r_ in rows if not r_['open']]
    op = [r_ for r_ in rows if r_['open']]
    dr = np.array([r_['ret'] for r_ in done])
    print(f"\n=== 완료 {len(done)}건 ===")
    print(f"  평균 {dr.mean()*100:+.1f}% · 중앙 {np.median(dr)*100:+.1f}% · "
          f"승률 {(dr>0).mean()*100:.0f}% · 최악 {dr.min()*100:+.1f}% · 최고 {dr.max()*100:+.1f}%")
    print(f"  투입 {len(done)*100} → 회수 {sum(100*(1+x) for x in dr):.0f}  "
          f"(총 {(sum(1+x for x in dr)/len(dr)-1)*100:+.1f}%)")
    if op:
        orr = np.array([r_['ret'] for r_ in op])
        print(f"\n=== 보유 중 {len(op)}건 ===")
        print(f"  평균 평가 {orr.mean()*100:+.1f}% · 투입 {len(op)*100} → 평가 {sum(100*(1+x) for x in orr):.0f}")
        allr = np.concatenate([dr, orr])
        print(f"\n=== 전체 {len(allr)}건 (평가 포함) ===")
        print(f"  평균 {allr.mean()*100:+.1f}% · 승률 {(allr>0).mean()*100:.0f}%")
        print(f"  총 투입 {len(allr)*100} → 총 평가 {sum(100*(1+x) for x in allr):.0f}  "
              f"({(sum(1+x for x in allr)/len(allr)-1)*100:+.1f}%)")

    json.dump([{'entry': px.index[r_['entry']].strftime('%Y-%m-%d'),
                'entry_price': round(float(V[r_['entry']]),2),
                'exit': None if r_['open'] else px.index[r_['exit']].strftime('%Y-%m-%d'),
                'exit_price': None if r_['open'] else round(float(V[r_['exit']]),2),
                'days': int((len(V)-1 if r_['open'] else r_['exit']) - r_['entry']),
                'ret': round(float(r_['ret']),4), 'open': r_['open'], 'why': r_['why']}
               for r_ in rows], open('data/independent_positions.json','w'),
              ensure_ascii=True, indent=2)
    print("\n저장: data/independent_positions.json")

if __name__ == '__main__':
    main()
