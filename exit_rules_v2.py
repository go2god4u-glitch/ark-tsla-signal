"""매도 규칙 2종 병렬 — 성격이 다른 신호를 하나로 합치지 않는다.

지금까지 모든 매도 신호를 하나의 규칙에 욱여넣으려다 실패했다.
아크 대량매도를 매도 조건에 AND 로 넣으면 개선이 없었는데,
그 이유는 **두 신호의 성격이 다르기 때문**이다.

  규칙 A (정상 매도)  아크 누적 -20% + 낙폭 -20% 회복
                      -> "목표 달성. 이익 실현"
  규칙 B (긴급 탈출)  아크 단일 주 대량매도
                      -> "전제가 깨졌다. 손실이든 이익이든 나온다"

A 는 이익 상태에서만 걸린다(낙폭 회복이 조건). B 는 상태와 무관하다.
둘을 AND 로 묶으면 B 가 죽고, OR 로 묶으면 A 의 이익 실현이 B 에 잘린다.
**병렬로 두고 먼저 걸리는 쪽으로 매도**하는 것이 맞다.

깊은 분석에서 확인한 B 의 성질:
  - 12개월 -52.1% (p=0.004). 시간이 갈수록 강해진다 = 장기 하락 예고
  - 1개월은 무효(p=0.31), 2개월부터 유효. 즉시 반응이 아니다
  - 3개 펀드 동시 매도 -28.7%(p=0.016) vs 1개만 -0.7%(p=0.48)
  - 단발 -7% 가 2주 연속보다 강하다
"""
import json
import numpy as np
import pandas as pd

ARK_DROP, DD_RECOVER, MAX_HOLD = 0.20, -20.0, 504


def main():
    from signal_check import build, build_daily
    r = json.load(open('data/tsla_full.json'))['chart']['result'][0]
    idx = pd.to_datetime(pd.Series(r['timestamp']), unit='s').dt.normalize()
    px = pd.Series(r['indicators']['quote'][0]['close'], index=idx).dropna().sort_index()
    V = px.values
    DD = (px / px.rolling(252, min_periods=60).max() - 1).values * 100
    daily = build_daily(px)
    H = daily['shares'].reindex(px.index).ffill().values
    wide = daily.attrs['wide']
    w = build(px)
    buys = sorted({px.index.searchsorted(t, side='right') for t in w[w['sig']].index
                   if px.index.searchsorted(t, side='right') < len(px)})

    # 규칙 B 후보들 — 각각 '그 신호가 뜬 다음 거래일' 을 매도일로
    wk = wide.resample('W-FRI').last()
    fw = wk.pct_change() * 100
    core = [c for c in ('ARKK', 'ARKQ', 'ARKW') if c in fw.columns]
    nsell = (fw[core] <= -3).sum(axis=1).reindex(w.index)

    def locs_of(weeks):
        return sorted({px.index.searchsorted(t, side='right') for t in weeks
                       if px.index.searchsorted(t, side='right') < len(px)})

    B_CANDS = {
        "없음 (규칙 A 단독)": [],
        "단일주 ≤-5%": locs_of(w.index[(w['netpct'] <= -5).fillna(False)]),
        "단일주 ≤-7%": locs_of(w.index[(w['netpct'] <= -7).fillna(False)]),
        "단일주 ≤-10%": locs_of(w.index[(w['netpct'] <= -10).fillna(False)]),
        "3개 펀드 동시 -3%": locs_of(w.index[(nsell >= 3).fillna(False)]),
        "2개 펀드 동시 -3%": locs_of(w.index[(nsell >= 2).fillna(False)]),
        "≤-7% 또는 3펀드동시": locs_of(w.index[((w['netpct'] <= -7) | (nsell >= 3)).fillna(False)]),
    }

    def run(bl):
        bs = np.array(sorted(bl)) if len(bl) else np.array([], int)
        rets, why, days = [], [], []
        for e in buys:
            exA = None
            for j in range(e + 1, min(e + MAX_HOLD + 1, len(V))):
                if H[j] <= H[e] * (1 - ARK_DROP) and DD[j] >= DD_RECOVER:
                    exA = j
                    break
            later = bs[bs > e] if len(bs) else np.array([])
            exB = int(later[0]) if len(later) else None
            cands = [(x, t) for x, t in ((exA, 'A'), (exB, 'B')) if x is not None]
            if cands:
                j, t = min(cands)
            else:
                j, t = min(e + MAX_HOLD, len(V) - 1), '상한'
            rets.append(V[j] / V[e] - 1)
            why.append(t)
            days.append(j - e)
        a = np.array(rets)
        return {'mean': a.mean(), 'hit': (a > 0).mean(), 'worst': a.min(),
                'value': sum(100 * (1 + x) for x in a), 'days': np.mean(days),
                'byB': why.count('B'), 'byA': why.count('A')}

    base = run([])
    print(f"기준(규칙 A 단독): 평가 {base['value']:.0f} · 평균 {base['mean']*100:+.1f}% · "
          f"승률 {base['hit']*100:.0f}% · 최악 {base['worst']*100:+.1f}%\n")
    print(f"{'규칙 B 후보':<22s}{'평가':>7s}{'변화':>7s}{'평균':>9s}{'승률':>6s}"
          f"{'최악':>9s}{'보유':>7s}{'B매도':>6s}")
    print("-" * 78)
    for name, bl in B_CANDS.items():
        s = run(bl)
        d = s['value'] - base['value']
        mark = ' ★' if d > 0 else ''
        print(f"{name:<22s}{s['value']:>7.0f}{d:>+7.0f}{s['mean']*100:>+8.1f}%"
              f"{s['hit']*100:>5.0f}%{s['worst']*100:>+8.1f}%{s['days']:>6.0f}일"
              f"{s['byB']:>6d}{mark}")


if __name__ == '__main__':
    main()
