"""보유 상태 추적 — 매수/매도 신호를 실제 행동 단위로 관리한다.

백테스트에서 확정한 규칙을 그대로 실행 상태로 옮긴다.

  진입  아크 주간 순매수가 문턱(과거 기준 상위 10%)을 넘은 주 -> 다음 거래일 종가
  청산  RSI(14) 가 70 을 넘었다가 다시 70 아래로 내려오면
        (6개월 안에 안 걸리면 6개월째에 정리)

왜 상태 파일이 필요한가:
  청산 조건이 '70 을 넘은 적이 있는가'에 의존한다(armed).
  매일 새로 계산하면 그 이력을 알 수 없으므로 파일에 남긴다.
  워크플로가 매일 data/ 를 커밋하므로 상태가 이어진다.

진입 시점 처리:
  신호는 금요일에 확정되고 진입은 다음 거래일 종가다.
  판정 시점에는 그 종가를 아직 모르므로 pending 으로 두고,
  다음 실행에서 실제 종가가 들어오면 그때 진입가를 확정한다.
  백테스트와 같은 시점에 같은 가격으로 들어가기 위한 장치다.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "data", "position.json")

RSI_LEVEL = 70          # 백테스트 고원 66~74 의 중앙
MAX_HOLD_DAYS = 126     # 6개월 상한. 없으면 무한 보유가 된다


def rsi14(px: pd.Series) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"open": False, "pending": False, "armed": False,
            "entry_date": None, "entry_price": None, "history": []}


def main() -> None:
    from signal_check import build, load_raw, refresh_prices  # noqa: F401
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    RSI = rsi14(px)

    w = build(px)
    sig_on = bool(w["sig"].iloc[-1])
    today = px.index[-1]
    price = float(px.iloc[-1])
    rsi_now = float(RSI.iloc[-1])

    st = load_state()
    action, reason = "NONE", ""

    # 1) 대기 중이던 진입을 확정한다 (신호 다음 거래일 종가)
    if st.get("pending") and not st["open"]:
        if st.get("pending_after") and today.strftime("%Y-%m-%d") > st["pending_after"]:
            st.update({"open": True, "pending": False, "armed": False,
                       "entry_date": today.strftime("%Y-%m-%d"), "entry_price": price})
            action, reason = "BUY", "신호 다음 거래일 종가로 진입 확정"

    # 2) 보유 중이면 청산 조건을 본다
    if st["open"]:
        entry = pd.Timestamp(st["entry_date"])
        held = int(px.index.searchsorted(today) - px.index.searchsorted(entry))
        if rsi_now >= RSI_LEVEL:
            st["armed"] = True
        if st["armed"] and rsi_now < RSI_LEVEL:
            action, reason = "SELL", f"RSI {rsi_now:.1f} 로 {RSI_LEVEL} 아래 복귀"
        elif held >= MAX_HOLD_DAYS:
            action, reason = "SELL", f"6개월 상한 도달 ({held}거래일)"
        if action == "SELL":
            ret = price / st["entry_price"] - 1
            st["history"].append({
                "entry": st["entry_date"], "entry_price": st["entry_price"],
                "exit": today.strftime("%Y-%m-%d"), "exit_price": price,
                "days": held, "ret": round(ret, 4), "reason": reason})
            st.update({"open": False, "pending": False, "armed": False,
                       "entry_date": None, "entry_price": None})

    # 3) 보유가 없고 신호가 켜졌으면 다음 거래일 진입을 예약한다
    if not st["open"] and not st["pending"] and sig_on:
        st.update({"pending": True, "pending_after": today.strftime("%Y-%m-%d")})
        action = action if action == "SELL" else "SIGNAL"
        reason = reason or "이번 주 매수 신호 — 다음 거래일 종가 진입 예정"

    st["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    st["rsi"] = round(rsi_now, 1)
    st["price"] = price
    st["price_date"] = today.strftime("%Y-%m-%d")
    if st["open"]:
        st["unrealized"] = round(price / st["entry_price"] - 1, 4)
        st["held_days"] = int(px.index.searchsorted(today)
                              - px.index.searchsorted(pd.Timestamp(st["entry_date"])))
        st["days_left"] = MAX_HOLD_DAYS - st["held_days"]
    else:
        st.pop("unrealized", None)
        st.pop("held_days", None)
        st.pop("days_left", None)

    with open(STATE, "w") as f:
        json.dump(st, f, ensure_ascii=True, indent=2)

    print(f"보유: {'있음' if st['open'] else '없음'} | RSI {rsi_now:.1f} | "
          f"armed={st['armed']} | 동작: {action or 'NONE'} {reason}")
    if st["open"]:
        print(f"  진입 {st['entry_date']} ${st['entry_price']:.2f} -> 현재 ${price:.2f} "
              f"({st['unrealized']*100:+.1f}%), {st['held_days']}일 보유, "
              f"상한까지 {st['days_left']}일")
    if st["history"]:
        h = st["history"][-1]
        print(f"  최근 청산: {h['entry']} -> {h['exit']} {h['ret']*100:+.1f}% ({h['reason']})")

    if out := os.environ.get("GITHUB_OUTPUT"):
        with open(out, "a") as f:
            f.write(f"action={action}\n")
            f.write(f"reason={reason}\n")
            f.write(f"pos_open={'true' if st['open'] else 'false'}\n")
            f.write(f"rsi={rsi_now:.1f}\n")
            f.write(f"entry_date={st.get('entry_date') or '-'}\n")
            f.write(f"entry_price={st.get('entry_price') or 0}\n")
            f.write(f"unrealized={round(st.get('unrealized', 0) * 100, 1)}\n")
            f.write(f"held_days={st.get('held_days', 0)}\n")
            f.write(f"days_left={st.get('days_left', 0)}\n")


if __name__ == "__main__":
    main()
