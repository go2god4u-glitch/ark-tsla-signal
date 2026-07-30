"""보유 상태 — 데이터에서 매번 재구성한다. 상태 파일에 의존하지 않는다.

규칙:
  진입  아크 주간 순매수가 문턱(과거 기준 상위 10%)을 넘은 주 -> 다음 거래일 종가
  청산  RSI(14) 가 70 을 넘었다가(armed) 다시 70 아래로 내려오면
        (6개월 = 126거래일 안에 안 걸리면 그때 정리)
  보유 중에 뜬 매수 신호는 무시한다(이미 들어가 있으므로).

왜 누적 상태 파일을 쓰지 않는가:
  처음에는 매 실행마다 상태를 파일에 이어 쓰는 방식으로 만들었다. 그런데
  파일이 없거나 초기화되면 과거 진입을 통째로 잊는다. 실제로 그 버그가 났다 —
  2026-06-01 에 진입해 보유 중인데도 상태 파일은 '보유 없음' 이라고 했다.

  보유 여부는 과거 데이터로 완전히 결정되는 값이다. 저장할 것이 아니라
  계산할 것이다. 매 실행마다 전 구간을 재생해 현재 위치를 확정한다.
  파일은 결과를 화면에 넘기기 위한 출력일 뿐, 판단의 입력이 아니다.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "position.json")

RSI_LEVEL = 70          # 백테스트 고원 66~74 의 중앙
MAX_HOLD = 126          # 6개월 상한. 없으면 무한 보유가 된다


def rsi14(px: pd.Series) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def replay(V, RSI, sig_locs):
    """전 구간을 재생해 완료 매매와 현재 보유를 돌려준다."""
    done, pos, armed = [], None, False
    for i in range(len(V)):
        if pos is None:
            if i in sig_locs:
                pos, armed = i, False
            continue
        if RSI[i] >= RSI_LEVEL:
            armed = True
        by_rsi = armed and RSI[i] < RSI_LEVEL
        by_cap = (i - pos) >= MAX_HOLD
        if by_rsi or by_cap:
            done.append({"entry_i": pos, "exit_i": i,
                         "ret": float(V[i] / V[pos] - 1),
                         "reason": "RSI 이탈" if by_rsi else "6개월 상한"})
            pos, armed = None, False
    return done, pos, armed


def main() -> None:
    from signal_check import build
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    V, RSI = px.values, rsi14(px).values

    w = build(px)
    sig_locs = {px.index.searchsorted(t, side="right") for t in w[w["sig"]].index}
    sig_locs = {i for i in sig_locs if i < len(px)}

    done, pos, armed = replay(V, RSI, sig_locs)
    last_i = len(V) - 1

    # 오늘 새로 발생한 것인지 판정 (알림은 '오늘 바뀐 것'에만 보낸다)
    action, reason = "NONE", ""
    if done and done[-1]["exit_i"] == last_i:
        action, reason = "SELL", done[-1]["reason"]
    elif pos == last_i:
        action, reason = "BUY", "매수 신호 다음 거래일 종가로 진입"

    st = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "price_date": px.index[last_i].strftime("%Y-%m-%d"),
        "price": float(V[last_i]),
        "rsi": round(float(RSI[last_i]), 1),
        "rsi_level": RSI_LEVEL,
        "open": pos is not None,
        "armed": bool(armed),
        "action": action, "reason": reason,
        "closed_trades": [
            {"entry": px.index[d["entry_i"]].strftime("%Y-%m-%d"),
             "entry_price": round(float(V[d["entry_i"]]), 2),
             "exit": px.index[d["exit_i"]].strftime("%Y-%m-%d"),
             "exit_price": round(float(V[d["exit_i"]]), 2),
             "days": int(d["exit_i"] - d["entry_i"]),
             "ret": round(d["ret"], 4), "reason": d["reason"]} for d in done],
        "cumulative": round(float(np.prod([1 + d["ret"] for d in done])), 3) if done else 1.0,
    }
    if pos is not None:
        st.update({
            "entry_date": px.index[pos].strftime("%Y-%m-%d"),
            "entry_price": round(float(V[pos]), 2),
            "held_days": int(last_i - pos),
            "days_left": int(MAX_HOLD - (last_i - pos)),
            "unrealized": round(float(V[last_i] / V[pos] - 1), 4),
            # 상한 도달일은 아직 오지 않은 날일 수 있다. 인덱스를 벗어나면
            # 남은 거래일을 달력일로 환산해 추정한다(주 5거래일 가정).
            "deadline": (px.index[pos + MAX_HOLD].strftime("%Y-%m-%d")
                         if pos + MAX_HOLD < len(px)
                         else (px.index[last_i] + pd.Timedelta(
                             days=int((MAX_HOLD - (last_i - pos)) * 7 / 5))
                         ).strftime("%Y-%m-%d")),
        })

    with open(OUT, "w") as f:
        json.dump(st, f, ensure_ascii=True, indent=2)

    print(f"완료 매매 {len(done)}건, 누적 {st['cumulative']:.2f}배")
    for t in st["closed_trades"]:
        print(f"  {t['entry']} ${t['entry_price']:>7.2f} -> {t['exit']} ${t['exit_price']:>7.2f}"
              f"  {t['days']:>3d}일 {t['ret']*100:>+7.1f}% ({t['reason']})")
    if st["open"]:
        print(f"\n★ 보유 중: {st['entry_date']} ${st['entry_price']} 진입 -> "
              f"${st['price']} ({st['unrealized']*100:+.1f}%)")
        print(f"  {st['held_days']}거래일 보유, 상한까지 {st['days_left']}일 (~{st['deadline']})")
        print(f"  RSI {st['rsi']} / 70 돌파 이력 {'있음' if armed else '없음'}")
        print(f"  -> 기다리는 것: 매도 신호")
    else:
        print(f"\n보유 없음 -> 기다리는 것: 매수 신호 (RSI {st['rsi']})")
    print(f"오늘 동작: {action} {reason}")

    if o := os.environ.get("GITHUB_OUTPUT"):
        with open(o, "a") as f:
            f.write(f"action={action}\nreason={reason}\n")
            f.write(f"pos_open={'true' if st['open'] else 'false'}\n")
            f.write(f"rsi={st['rsi']}\n")
            f.write(f"entry_date={st.get('entry_date', '-')}\n")
            f.write(f"entry_price={st.get('entry_price', 0)}\n")
            f.write(f"unrealized={round(st.get('unrealized', 0) * 100, 1)}\n")
            f.write(f"held_days={st.get('held_days', 0)}\n")
            f.write(f"days_left={st.get('days_left', 0)}\n")
            f.write(f"deadline={st.get('deadline', '-')}\n")


if __name__ == "__main__":
    main()
