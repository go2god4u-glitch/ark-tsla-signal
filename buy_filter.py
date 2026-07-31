"""매수 신호에 기술적 필터를 걸까 — 파라미터 고원으로 판단한다.

앞선 결과 두 가지가 충돌한다.
  (1) 지표 단독은 매수 신호로 무효였다 (RSI -2.1%, MACD +6.6% 등, 전부 p>0.19)
  (2) 그런데 아크 신호를 진입 시점 RSI/낙폭으로 나누면 차이가 컸다
      (RSI<35 진입 +40.2% vs RSI 55↑ 진입 +18.6%)

(2)는 '지표가 신호를 만든다'가 아니라 '아크 신호를 걸러낸다'는 뜻이다.
둘은 다른 주장이므로 따로 검증해야 한다.

여기서는 아크 신호에 필터를 걸었을 때를 본다. 판단 기준은 매도 규칙 때와 같다:
  최고값이 아니라 **파라미터 구간 전체**. 한 점에서만 좋으면 우연이다.
  그리고 필터는 신호 수를 줄이므로, 줄어든 만큼 값어치가 있는지 봐야 한다.

매도는 확정 규칙(RSI 70 이탈 / 126거래일 상한)으로 고정한다.
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

    def outcome(e):
        armed = False
        for j in range(e+1, len(V)):
            if RSI[j] >= RSI_LEVEL: armed = True
            if (armed and RSI[j] < RSI_LEVEL) or (j-e) >= MAX_HOLD:
                return V[j]/V[e]-1, False
        return V[-1]/V[e]-1, True

    res = {e: outcome(e) for e in sig}

    def stats(sel):
        if not sel: return None
        done = [res[e][0] for e in sel if not res[e][1]]
        live = [res[e][0] for e in sel if res[e][1]]
        allr = done + live
        return {'n': len(sel), 'done': len(done),
                'mean_done': np.mean(done) if done else None,
                'hit_done': np.mean([x > 0 for x in done]) if done else None,
                'mean_all': np.mean(allr), 'hit_all': np.mean([x > 0 for x in allr]),
                'worst': min(allr), 'live_mean': np.mean(live) if live else None}

    base = stats(sig)
    print(f"필터 없음 (아크 단독): 신호 {base['n']}건 · 완료 {base['done']}건 "
          f"평균 {base['mean_done']*100:+.1f}% · 전체 평균 {base['mean_all']*100:+.1f}% "
          f"승률 {base['hit_all']*100:.0f}% · 최악 {base['worst']*100:+.1f}%\n")

    print("=" * 96)
    print("A. RSI 상한 필터 — 'RSI 가 X 미만일 때만 산다'")
    print("=" * 96)
    print(f"{'필터':<16s}{'신호':>5s}{'완료':>5s}{'완료평균':>10s}{'전체평균':>10s}"
          f"{'전체승률':>9s}{'최악':>9s}{'놓친신호':>9s}")
    rows_a = []
    for th in (30, 35, 40, 45, 50, 55, 60, 100):
        sel = [e for e in sig if RSI[e] < th]
        s = stats(sel)
        if not s: continue
        lab = f"RSI < {th}" if th < 100 else "제한 없음"
        print(f"{lab:<16s}{s['n']:>5d}{s['done']:>5d}"
              + (f"{s['mean_done']*100:>+9.1f}%" if s['mean_done'] is not None else f"{'-':>10s}")
              + f"{s['mean_all']*100:>+9.1f}%{s['hit_all']*100:>8.0f}%{s['worst']*100:>+8.1f}%"
              + f"{base['n']-s['n']:>8d}건")
        rows_a.append((th, s))

    print()
    print("=" * 96)
    print("B. 낙폭 필터 — '전고점 대비 X% 이하일 때만 산다'")
    print("=" * 96)
    print(f"{'필터':<16s}{'신호':>5s}{'완료':>5s}{'완료평균':>10s}{'전체평균':>10s}"
          f"{'전체승률':>9s}{'최악':>9s}{'놓친신호':>9s}")
    for th in (-45, -40, -35, -30, -25, -20, 0):
        sel = [e for e in sig if DD[e] <= th]
        s = stats(sel)
        if not s: continue
        lab = f"낙폭 ≤ {th}%" if th < 0 else "제한 없음"
        print(f"{lab:<16s}{s['n']:>5d}{s['done']:>5d}"
              + (f"{s['mean_done']*100:>+9.1f}%" if s['mean_done'] is not None else f"{'-':>10s}")
              + f"{s['mean_all']*100:>+9.1f}%{s['hit_all']*100:>8.0f}%{s['worst']*100:>+8.1f}%"
              + f"{base['n']-s['n']:>8d}건")

    print()
    print("=" * 96)
    print("C. 200일선 아래에서만 산다")
    print("=" * 96)
    for lab, cond in (("200일선 아래", lambda e: V[e] < MA200[e]),
                      ("200일선 위", lambda e: V[e] >= MA200[e])):
        sel = [e for e in sig if not np.isnan(MA200[e]) and cond(e)]
        s = stats(sel)
        if s:
            print(f"  {lab:<14s} 신호 {s['n']:>2d}건 완료 {s['done']:>2d}건 "
                  + (f"완료평균 {s['mean_done']*100:>+6.1f}% " if s['mean_done'] is not None else "")
                  + f"전체평균 {s['mean_all']*100:>+6.1f}% 승률 {s['hit_all']*100:>3.0f}% "
                  f"최악 {s['worst']*100:>+6.1f}%")

    print()
    print("=" * 96)
    print("판단")
    print("=" * 96)
    print("  필터를 조일수록 완료 평균은 오르지만 신호 수가 줄고 최악이 나빠지는지 본다.")
    print("  '전체 평균'은 아직 열려 있는 손실 포지션까지 포함한 값이다. 이쪽을 봐야 한다.")

if __name__ == '__main__':
    main()
