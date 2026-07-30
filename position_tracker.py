"""보유 상태 — 데이터에서 매번 재구성한다. 상태 파일에 의존하지 않는다.

규칙:
  진입  아크 주간 순매수가 문턱(과거 기준 상위 10%)을 넘고
        낙폭이 -30% 이하인 주 -> 다음 거래일 종가
  청산  진입 후 126거래일(약 6개월) 경과

청산 규칙을 RSI 70 이탈에서 6개월 고정으로 바꾼 이유:
  RSI 규칙은 '단일 포지션 복리' 틀에서 골랐다. 그 틀은 겹치는 신호를 하나로
  병합해 매매가 5건뿐이었다. 이후 자본 모델을 '신호마다 독립 100' 으로 바꿨는데
  매도 규칙은 다시 고르지 않았다. 틀이 바뀌면 순위가 바뀐다 — 실제로 뒤집혔다.

  독립 포지션 틀(총 16건) 기준 평가액:
    6개월 고정        2290   (승률 88%, 최악 -3.5%, 평균보유 118일)
    RSI 70 이탈       2093   (승률 94%, 최악 -3.5%, 평균보유  49일)
    9개월 고정        2464   (승률 88%, 최악 -37.0%)
    10개월 고정       2514   (승률 88%, 최악 -61.2%)

  더 오래 들면 평가액은 오르지만 최악 사례가 -3.5% -> -61.2% 로 급격히 나빠진다.
  5~7개월 구간이 최악 -3.5% 를 유지하면서 평가 2226~2290 으로 안정적이다.

  아크 매도를 넣어도 개선되지 않는다: 아크 하위10% AND RSI≥70 이 2328 인데
  아크 없는 126일 고정이 2290 이다(차이 1.7%). 게다가 그 규칙의 청산 15건 중
  10건이 상한 도달이라, 실제로는 아크가 아니라 '오래 들고 있던' 효과였다.
  신호마다 독립 포지션이다. 이미 보유 중이어도 새 신호가 뜨면 새로 100 을 넣고,
  그 포지션은 자기 진입일 기준으로 청산 조건을 본다.
  'RSI 70 을 넘은 적이 있는가'(armed)도 포지션마다 따로 센다.

  처음에는 '보유 중 신호는 무시' 로 만들었는데 근거 없는 임의 규칙이었다.
  아크가 물타기 하는 구간(2026-07-13, 07-27)을 통째로 흘려보내고 있었다.

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

MAX_HOLD = 126          # 6개월 고정 보유 (독립 포지션 틀에서 재선정)
RSI_LEVEL = 70          # 참고 표시용. 청산 판정에는 쓰지 않는다


def rsi14(px: pd.Series) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def replay(V, RSI, sig_locs):
    """신호마다 독립 포지션을 돌린다. (완료 목록, 보유 중 목록)."""
    done, live = [], []
    for e in sorted(sig_locs):
        j = e + MAX_HOLD
        if j < len(V):
            done.append({"entry_i": e, "exit_i": j,
                         "ret": float(V[j] / V[e] - 1),
                         "reason": "6개월 경과"})
        else:
            live.append({"entry_i": e,
                         "armed": bool((RSI[e + 1:] >= RSI_LEVEL).any()),
                         "ret": float(V[-1] / V[e] - 1)})
    return done, live


def main() -> None:
    from signal_check import build
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    V, RSI = px.values, rsi14(px).values

    w = build(px)
    sig_locs = {px.index.searchsorted(t, side="right") for t in w[w["sig"]].index}
    sig_locs = {i for i in sig_locs if i < len(px)}

    done, live = replay(V, RSI, sig_locs)
    last_i = len(V) - 1

    # 오늘 새로 생긴 것만 알림 대상이다
    sold = [d for d in done if d["exit_i"] == last_i]
    bought = [p for p in live if p["entry_i"] == last_i]
    action = "SELL" if sold else ("BUY" if bought else "NONE")
    reason = (sold[0]["reason"] if sold else
              ("매수 신호 다음 거래일 종가로 진입" if bought else ""))

    dr = np.array([d["ret"] for d in done]) if done else np.array([])
    lr = np.array([p["ret"] for p in live]) if live else np.array([])
    allr = np.concatenate([dr, lr]) if len(dr) or len(lr) else np.array([])

    st = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "price_date": px.index[last_i].strftime("%Y-%m-%d"),
        "price": round(float(V[last_i]), 2),
        "rsi": round(float(RSI[last_i]), 1),
        "rsi_level": RSI_LEVEL,
        "max_hold": MAX_HOLD,
        "action": action, "reason": reason,
        "open_count": len(live),
        "open": len(live) > 0,
        "closed": [
            {"entry": px.index[d["entry_i"]].strftime("%Y-%m-%d"),
             "entry_price": round(float(V[d["entry_i"]]), 2),
             "exit": px.index[d["exit_i"]].strftime("%Y-%m-%d"),
             "exit_price": round(float(V[d["exit_i"]]), 2),
             "days": int(d["exit_i"] - d["entry_i"]),
             "ret": round(d["ret"], 6), "reason": d["reason"]} for d in done],
        "live": [
            {"entry": px.index[p["entry_i"]].strftime("%Y-%m-%d"),
             "entry_price": round(float(V[p["entry_i"]]), 2),
             "days": int(last_i - p["entry_i"]),
             "days_left": int(MAX_HOLD - (last_i - p["entry_i"])),
             "ret": round(p["ret"], 6), "armed": bool(p["armed"]),
             "deadline": (px.index[p["entry_i"] + MAX_HOLD].strftime("%Y-%m-%d")
                          if p["entry_i"] + MAX_HOLD < len(px)
                          else (px.index[last_i] + pd.Timedelta(
                              days=int((MAX_HOLD - (last_i - p["entry_i"])) * 7 / 5))
                          ).strftime("%Y-%m-%d"))} for p in live],
        "stats": {
            "closed_n": len(done),
            "closed_mean": round(float(dr.mean()), 6) if len(dr) else None,
            "closed_hit": round(float((dr > 0).mean()), 3) if len(dr) else None,
            "closed_worst": round(float(dr.min()), 6) if len(dr) else None,
            "live_mean": round(float(lr.mean()), 6) if len(lr) else None,
            "all_n": len(allr),
            "all_mean": round(float(allr.mean()), 6) if len(allr) else None,
            "all_hit": round(float((allr > 0).mean()), 3) if len(allr) else None,
            "invested": len(allr) * 100,
            "value": round(float(sum(100 * (1 + x) for x in allr)), 0) if len(allr) else 0,
        },
    }
    with open(OUT, "w") as f:
        json.dump(st, f, ensure_ascii=True, indent=2)

    s_ = st["stats"]
    print(f"완료 {s_['closed_n']}건 평균 {s_['closed_mean']*100:+.1f}% 승률 {s_['closed_hit']*100:.0f}%")
    print(f"보유 {len(live)}건 평균 {s_['live_mean']*100:+.1f}%" if live else "보유 없음")
    print(f"전체 {s_['all_n']}건: 투입 {s_['invested']} -> 평가 {s_['value']:.0f} "
          f"({s_['all_mean']*100:+.1f}%, 승률 {s_['all_hit']*100:.0f}%)")
    for p in st["live"]:
        print(f"  보유: {p['entry']} ${p['entry_price']} {p['ret']*100:+.1f}% "
              f"({p['days']}일, 상한 {p['deadline']}, RSI70돌파 {'O' if p['armed'] else 'X'})")
    print(f"오늘 동작: {action} {reason}")

    if o := os.environ.get("GITHUB_OUTPUT"):
        with open(o, "a") as f:
            f.write(f"action={action}\nreason={reason}\n")
            f.write(f"open_count={len(live)}\nrsi={st['rsi']}\n")
            f.write(f"all_mean={round((s_['all_mean'] or 0)*100,1)}\n")
            f.write(f"live_mean={round((s_['live_mean'] or 0)*100,1)}\n")
            if sold:
                d = st["closed"][-1]
                f.write(f"sold_entry={d['entry']}\nsold_ret={round(d['ret']*100,1)}\n"
                        f"sold_days={d['days']}\n")


if __name__ == "__main__":
    main()
