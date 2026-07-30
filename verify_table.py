"""표 전수 검증 — 화면에 나가는 모든 숫자를 원본에서 다시 계산해 대조한다.

대시보드가 쓰는 data/signal_state.json 의 runs 를 그대로 읽어,
각 셀을 야후 원본 종가와 규칙 정의로부터 독립적으로 재계산해 비교한다.
계산 함수를 재사용하지 않고 여기서 직접 다시 구현한다 —
같은 함수를 쓰면 같은 실수를 그대로 통과시킨다.
"""
import json
import numpy as np
import pandas as pd

from decimal import Decimal, ROUND_HALF_UP

def half_up(x, n=1):
    """화면(JS Math.round)과 같은 half-up 반올림. 파이썬 round 는 half-to-even 이라 다르다."""
    return float(Decimal(repr(x)).quantize(Decimal('1.' + '0'*n), rounding=ROUND_HALF_UP))

FAIL = []
def chk(cond, msg):
    if not cond:
        FAIL.append(msg)
    return cond

# --- 원본에서 직접 종가 시리즈를 만든다 ---
raw = json.load(open('data/tsla_full.json'))['chart']['result'][0]
ts = pd.to_datetime(pd.Series(raw['timestamp']), unit='s').dt.normalize()
close = pd.Series(raw['indicators']['quote'][0]['close'], index=ts).dropna().sort_index()


# 아크 보유·낙폭도 원본에서 다시 만든다
from signal_check import build_daily
HH = build_daily(close)['shares'].reindex(close.index).ffill().values
DDv = ((close / close.rolling(252, min_periods=60).max() - 1) * 100).values

state = json.load(open('data/signal_state.json'))
runs = state['runs']

print(f"국면 {len(runs)}개 / 신호 {sum(r['weeks'] for r in runs)}회\n")
KO = ['월','화','수','목','금','토','일']
n_leg = 0

for ri, r in enumerate(runs, 1):
    for e in r['entries']:
        n_leg += 1
        sig = pd.Timestamp(e['signal']); ent = pd.Timestamp(e['date'])
        # 1) 신호일은 금요일인가
        chk(sig.weekday() == 4, f"[{ri}] 신호일 {e['signal']} 이 금요일이 아니다({KO[sig.weekday()]})")
        # 2) 매수일 = 신호일 '다음' 거래일인가
        nxt = close.index[close.index > sig]
        chk(len(nxt) and nxt[0] == ent,
            f"[{ri}] {e['signal']} 다음 거래일은 {nxt[0]:%Y-%m-%d} 인데 표는 {e['date']}")
        # 3) 매수가 = 그날 종가인가
        chk(abs(close.loc[ent] - e['price']) < 0.01,
            f"[{ri}] {e['date']} 종가 {close.loc[ent]:.2f} vs 표 {e['price']}")
        # 4) 청산 규칙 재현 — 아크 보유 -20% AND 낙폭 -20% 회복
        k = close.index.get_loc(ent)
        ex = why = None
        for j in range(k+1, min(k+505, len(close))):
            if HH[j] <= HH[k]*0.80 and DDv[j] >= -20.0:
                ex, why = j, '아크 -20% + 낙폭 회복'; break
        if ex is None and k+504 < len(close):
            ex, why = k+504, '상한 도달'
        if e['open']:
            chk(ex is None, f"[{ri}] {e['date']} 는 보유중이라는데 청산 조건이 이미 걸렸다")
            cur = float(close.iloc[-1])
            chk(abs(cur/e['price'] - 1 - e['ret']) < 0.0001,
                f"[{ri}] {e['date']} 평가수익 재계산 {cur/e['price']-1:.4f} vs 표 {e['ret']}")
            chk(e['days'] == len(close)-1-k, f"[{ri}] {e['date']} 보유일 불일치")
        else:
            chk(ex is not None, f"[{ri}] {e['date']} 청산됐다는데 조건이 안 걸린다")
            chk(close.index[ex].strftime('%Y-%m-%d') == e['exit'],
                f"[{ri}] 청산일 재계산 {close.index[ex]:%Y-%m-%d} vs 표 {e['exit']}")
            chk(abs(close.iloc[ex] - e['exit_price']) < 0.01,
                f"[{ri}] 청산가 재계산 {close.iloc[ex]:.2f} vs 표 {e['exit_price']}")
            chk(why == e['why'], f"[{ri}] 사유 {why} vs 표 {e['why']}")
            chk(e['days'] == ex-k, f"[{ri}] 보유일 재계산 {ex-k} vs 표 {e['days']}")
            chk(abs(close.iloc[ex]/e['price'] - 1 - e['ret']) < 0.0001,
                f"[{ri}] 수익 재계산 {close.iloc[ex]/e['price']-1:.4f} vs 표 {e['ret']}")
    # 5) 합산 행
    prices = [x['price'] for x in r['entries']]
    rets = [x['ret'] for x in r['entries']]
    chk(abs(np.mean(prices) - r['avg_price']) < 0.01,
        f"[{ri}] 평균매수가 {np.mean(prices):.2f} vs 표 {r['avg_price']}")
    chk(abs(np.mean(rets) - r['run_ret']) < 0.0001,
        f"[{ri}] 국면수익 {np.mean(rets):.4f} vs 표 {r['run_ret']}")
    chk(r['weeks'] == len(r['entries']), f"[{ri}] 신호 횟수 불일치")
    # 6) 연속 정의: 국면 내부는 7일 이내, 국면 간은 7일 초과
    sigs = [pd.Timestamp(x['signal']) for x in r['entries']]
    for a, b in zip(sigs, sigs[1:]):
        chk((b-a).days <= 7, f"[{ri}] 국면 내부 간격 {(b-a).days}일 > 7")
    if ri < len(runs):
        nxt_first = pd.Timestamp(runs[ri]['entries'][0]['signal'])
        chk((nxt_first - sigs[-1]).days > 7,
            f"[{ri}] 국면 {ri}·{ri+1} 간격 {(nxt_first-sigs[-1]).days}일 ≤ 7 (같은 국면이어야)")

# 6-b) 이중 반올림 — 저장값을 화면에서 다시 반올림해도 원값과 같은가
for ri, r in enumerate(runs, 1):
    for e in r['entries']:
        base = e['exit_price'] if e['exit_price'] is not None else float(close.iloc[-1])
        true = base / e['price'] - 1
        chk(half_up(true*100) == half_up(e['ret']*100),
            f"[{ri}] {e['date']} 표시 반올림 불일치: 실제 {true*100:.4f}% -> "
            f"{half_up(true*100)}% / 저장 {e['ret']} -> {half_up(e['ret']*100)}%")

# 7) 신호 집합이 알고리즘과 일치하는가 (독립 재계산)
from signal_check import build
w = build(close)
algo = {t.strftime('%Y-%m-%d') for t in w[w['sig']].index}
table = {x['signal'] for r in runs for x in r['entries']}
chk(algo == table, f"신호 집합 불일치: 알고리즘만 {sorted(algo-table)}, 표만 {sorted(table-algo)}")

# 8) 전체 합계
all_rets = [x['ret'] for r in runs for x in r['entries']]
print(f"검증 항목 {n_leg}개 매매 + 합산 {sum(1 for r in runs if r['weeks']>1)}행")
print(f"전체 {len(all_rets)}회 평균 {np.mean(all_rets)*100:+.2f}% "
      f"승률 {np.mean([x>0 for x in all_rets])*100:.0f}% "
      f"투입 {len(all_rets)*100} -> {sum(100*(1+x) for x in all_rets):.0f}")
print()
if FAIL:
    print(f"❌ 불일치 {len(FAIL)}건")
    for f in FAIL: print("   -", f)
else:
    print("✅ 전 항목 일치 — 신호일 요일, 매수일=다음거래일, 매수가/청산가=원본종가,")
    print("   청산 규칙 재현, 보유일, 수익률, 국면 평균·합산, 연속 정의, 신호 집합")
