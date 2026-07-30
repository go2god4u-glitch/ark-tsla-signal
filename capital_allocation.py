"""자본 배분 방식 비교 — 신호마다 얼마씩 넣을 것인가.

앞선 '추가 매수' 백테스트에는 실행 불가능한 가정이 있었다.
평단을 매수 시점들의 **단순 평균**으로 계산했는데, 그것은 신호가 몇 번 올지
미리 알고 그 수만큼 균등 분할했다는 뜻이다. 실전에서는 알 수 없다.
게다가 현금으로 남은 부분의 수익률(0%)을 반영하지 않았다.

여기서는 **총자본 100 기준**으로 계산한다.
  - 매 사이클 시작 시 현금 100
  - 신호가 뜰 때마다 정해진 배분 규칙에 따라 매수
  - 남은 현금은 수익률 0
  - 청산 시 전량 매도, 수익률은 100 대비로 계산

이렇게 해야 "100 넣고 1% 번 것"과 "10 넣고 10% 번 것"이 같은 자로 비교된다.

배분 규칙(전부 미래를 모르는 상태에서 실행 가능):
  일괄100    첫 신호에 100% — 추가 매수 불가
  균등N      신호마다 100/N%. N번 채우면 더 사지 않는다
  잔액절반   신호마다 남은 현금의 50%. 소진되지 않지만 뒤로 갈수록 작아진다
  잔액1/3    신호마다 남은 현금의 1/3
  선착순감소  50/30/20% 순서로

청산 규칙은 확정된 것을 쓴다: RSI(14) 70 돌파 후 이탈, 없으면 126거래일 상한.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RSI_LEVEL = 70
MAX_HOLD = 126


def rsi14(px: pd.Series) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# 배분 규칙: (이미 매수한 횟수, 남은 현금) -> 이번에 넣을 금액
SCHEMES = {
    "일괄 100%": lambda k, cash: cash if k == 0 else 0.0,
    "균등 2회 (50%씩)": lambda k, cash: 50.0 if k < 2 else 0.0,
    "균등 3회 (33%씩)": lambda k, cash: 100 / 3 if k < 3 else 0.0,
    "균등 4회 (25%씩)": lambda k, cash: 25.0 if k < 4 else 0.0,
    "잔액의 1/2": lambda k, cash: cash * 0.5,
    "잔액의 1/3": lambda k, cash: cash / 3,
    "50/30/20": lambda k, cash: [50.0, 30.0, 20.0][k] if k < 3 else 0.0,
}


def run(V, RSI, sigset, alloc):
    """사이클마다 자본 100 으로 시작. 총자본 기준 수익률을 돌려준다."""
    done, open_pos = [], None
    cash, shares, buys, first, armed = 100.0, 0.0, [], None, False
    holding = False
    for i in range(len(V)):
        if not holding:
            if i in sigset:
                holding, cash, shares, buys, first, armed = True, 100.0, 0.0, [], i, False
                amt = min(alloc(0, cash), cash)
                cash -= amt
                shares += amt / V[i]
                buys.append((i, amt))
            continue
        if i in sigset:
            amt = min(alloc(len(buys), cash), cash)
            if amt > 0:
                cash -= amt
                shares += amt / V[i]
                buys.append((i, amt))
        if RSI[i] >= RSI_LEVEL:
            armed = True
        by_rsi = armed and RSI[i] < RSI_LEVEL
        by_cap = (i - first) >= MAX_HOLD
        if by_rsi or by_cap:
            total = cash + shares * V[i]
            done.append({"first": first, "exit": i, "buys": len(buys),
                         "invested": 100 - cash, "ret": total / 100 - 1,
                         "why": "RSI 이탈" if by_rsi else "6개월 상한"})
            holding = False
    if holding:
        total = cash + shares * V[-1]
        open_pos = {"first": first, "buys": len(buys), "invested": 100 - cash,
                    "cash": cash, "ret": total / 100 - 1,
                    "lots": [(i, round(a, 1)) for i, a in buys]}
    return done, open_pos


def main() -> None:
    from signal_check import build
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    V, RSI = px.values, rsi14(px).values
    w = build(px)
    sigset = {px.index.searchsorted(t, side="right") for t in w[w["sig"]].index}
    sigset = {i for i in sigset if i < len(px)}

    print("총자본 100 기준. 현금으로 남은 부분은 수익률 0% 로 반영한다.\n")
    print(f"{'배분 방식':<20s}{'완료':>5s}{'평균':>9s}{'승률':>6s}{'최악':>9s}"
          f"{'누적':>8s}{'평균투입':>9s}{'현재포지션':>12s}{'포함누적':>10s}")
    print("-" * 92)
    rows = []
    for name, alloc in SCHEMES.items():
        done, op = run(V, RSI, sigset, alloc)
        rets = np.array([d["ret"] for d in done])
        cum = float(np.prod(1 + rets))
        inv = float(np.mean([d["invested"] for d in done]))
        with_open = cum * (1 + op["ret"]) if op else cum
        print(f"{name:<20s}{len(done):>5d}{rets.mean()*100:>+8.1f}%"
              f"{(rets>0).mean()*100:>5.0f}%{rets.min()*100:>+8.1f}%"
              f"{cum:>7.2f}x{inv:>8.0f}%"
              + (f"{op['ret']*100:>+11.1f}%" if op else f"{'-':>12s}")
              + f"{with_open:>9.2f}x")
        rows.append({"scheme": name, "n": len(done), "mean": float(rets.mean()),
                     "hit": float((rets > 0).mean()), "worst": float(rets.min()),
                     "cum": cum, "avg_invested": inv,
                     "open_ret": op["ret"] if op else None,
                     "cum_with_open": float(with_open)})

    print("\n" + "=" * 92)
    print("현재 열린 포지션 상세 (배분 방식별)")
    print("=" * 92)
    for name in ("일괄 100%", "균등 3회 (33%씩)", "잔액의 1/2"):
        done, op = run(V, RSI, sigset, SCHEMES[name])
        if not op:
            continue
        print(f"\n[{name}]  투입 {op['invested']:.0f}% / 현금 {op['cash']:.0f}% "
              f"/ 평가 {op['ret']*100:+.1f}%")
        for i, a in op["lots"]:
            print(f"    {px.index[i]:%Y-%m-%d}  ${V[i]:>7.2f}  {a:>5.1f}% 투입")

    with open(f"{BASE}/data/capital_allocation.json", "w") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2, default=float)
    print("\n저장: data/capital_allocation.json")


if __name__ == "__main__":
    main()
