"""보유 상태 — 데이터에서 매번 재구성한다. 상태 파일에 의존하지 않는다.

규칙 (진입과 매도가 서로의 거울상이다):

  진입  아크 주간 순매수율이 문턱(과거 기준 상위 10%)을 넘고
        전고점 대비 낙폭이 -30% 이하인 주  ->  다음 거래일 종가
  매도  두 규칙을 **병렬로** 두고 먼저 걸리는 쪽으로 매도한다.

    규칙 A (정상 매도)  아크 누적 -20% AND 낙폭 -20% 회복
                        "목표 달성. 이익 실현"
    규칙 B (긴급 탈출)  아크가 단일 주에 -7% 이상 매도
                        "전제가 깨졌다. 손실이든 이익이든 나온다"

    (둘 다 안 걸리면 504거래일 상한)

  왜 병렬인가:
    A 는 이익 상태에서만 걸린다(낙폭 회복이 조건). B 는 상태와 무관하다.
    AND 로 묶으면 B 가 죽고, 단일 규칙에 욱여넣으면 성격이 섞여 둘 다 약해진다.
    실제로 대량매도를 A 의 조건에 AND 로 넣었을 때는 개선이 없었다.

  규칙 B 의 근거:
    주간 순매수율 ≤ -7% 인 5건은 3개월 뒤 전부 하락(초과 -27.3%, p=0.011).
    12개월로 가면 -52.1%(p=0.004) 로 더 강해진다 — 장기 하락 예고다.
    -6% ~ -12% 전 구간이 A 단독(2689)을 넘는 고원이다(최고 2819).
    적용 효과: 평가 2689 -> 2819, 최악 -9.2% -> -3.5%.

  신호마다 독립 포지션이다. 이미 보유 중이어도 새 신호가 뜨면 새로 진입하고,
  각 포지션은 자기 진입일 기준으로 매도 조건을 본다.

왜 이 규칙인가:
  '6개월 고정' 은 성적은 났지만 원칙이 없다. 진입이 '아크가 사고 + 낙폭 깊을 때'
  이므로 매도는 '아크가 줄이고 + 낙폭 회복' 이 자연스러운 대칭이다.

  아크 감소는 '이번 주에 팔았나'(잡음이 많다)가 아니라
  '진입 이후 누적으로 얼마나 빠져나갔나'를 본다.

  검증: 아크 -10~-25% × 낙폭 -30~-10% 격자 35칸이 **전부** 126일 고정(2290)을
  넘었다(중앙값 2610, 최솟값 2364). 한 점이 아니라 구간 전체다.
  채택값(-20%/-20%)은 격자 중앙이며 최고값(-25%/-10%, 3066)이 아니다.

  기간 분할도 통과: 전반 8건 +82.6%(승률 100%), 후반 8건 +56.6%(88%).

왜 누적 상태 파일을 쓰지 않는가:
  파일이 없거나 초기화되면 과거 진입을 통째로 잊는다. 실제로 그 버그가 났다.
  보유 여부는 과거 데이터로 완전히 결정되는 값이므로 매 실행마다 재생한다.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "position.json")

ARK_DROP = 0.20         # [규칙 A] 아크 보유가 진입 대비 이만큼 줄면
DD_RECOVER = -20.0      # [규칙 A] AND 낙폭이 이 수준까지 회복하면
BIG_SELL = -7.0         # [규칙 B] 아크가 단일 주에 이만큼 팔면 (긴급 탈출)
MAX_HOLD = 504          # 안전장치. 규칙이 안 걸릴 때만 쓰인다
RSI_LEVEL = 70          # 참고 표시용. 매도 판정에는 쓰지 않는다


def rsi14(px: pd.Series) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def replay(V, RSI, sig_locs, H, DD, big_locs):
    """신호마다 독립 포지션. (완료 목록, 보유 중 목록).

    H  = 아크 합산 보유 주식 수 (가격 인덱스에 맞춰 채운 것)
    DD = 전고점 대비 낙폭 (%)
    """
    done, live = [], []
    big = np.array(sorted(big_locs)) if len(big_locs) else np.array([], dtype=int)
    for e in sorted(sig_locs):
        # 규칙 A: 아크 누적 감소 + 낙폭 회복
        exA = None
        for j in range(e + 1, min(e + MAX_HOLD + 1, len(V))):
            if H[j] <= H[e] * (1 - ARK_DROP) and DD[j] >= DD_RECOVER:
                exA = j
                break
        # 규칙 B: 아크 단일 주 대량매도 (긴급 탈출)
        later = big[big > e] if len(big) else np.array([], dtype=int)
        exB = int(later[0]) if len(later) and later[0] <= e + MAX_HOLD else None
        cand = [(x, t) for x, t in ((exA, "A: 아크 -20% + 낙폭 회복"),
                                    (exB, "B: 아크 대량매도")) if x is not None]
        hit = min(cand) if cand else None
        if hit is None and e + MAX_HOLD < len(V):
            hit = (e + MAX_HOLD, "상한 도달")
        if hit is None:
            live.append({"entry_i": e,
                         "ark_drop": float(H[-1] / H[e] - 1),
                         "dd": float(DD[-1]),
                         "ret": float(V[-1] / V[e] - 1)})
        else:
            j, why = hit
            done.append({"entry_i": e, "exit_i": j,
                         "ret": float(V[j] / V[e] - 1), "reason": why})
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

    from signal_check import build_daily
    H = build_daily(px)["shares"].reindex(px.index).ffill().values
    DD = ((px / px.rolling(252, min_periods=60).max() - 1) * 100).values
    # 규칙 B 발동 시점 (주간 순매수율 <= BIG_SELL)
    big_locs = {px.index.searchsorted(t, side="right")
                for t in w.index[(w["netpct"] <= BIG_SELL).fillna(False)]}
    big_locs = {i for i in big_locs if i < len(px)}
    done, live = replay(V, RSI, sig_locs, H, DD, big_locs)
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

    cur_netpct = float(w["netpct"].iloc[-1]) if len(w) else 0.0
    st = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # 규칙 B 진행 상황. 화면·알림이 '지금 얼마나 팔았나' 를 보여주려면 필요하다.
        "cur_netpct": round(cur_netpct, 2),
        "big_sell": BIG_SELL,
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
             "ret": round(p["ret"], 6),
             "ark_drop": round(p["ark_drop"], 4), "dd": round(p["dd"], 1),
             "need_ark": ARK_DROP * 100, "need_dd": DD_RECOVER,
             "big_sell": BIG_SELL,
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
        print(f"  보유: {p['entry']} ${p['entry_price']} {p['ret']*100:+.1f}% ({p['days']}일)")
        print(f"        매도 조건  아크 {p['ark_drop']*100:+.1f}% (필요 -{p['need_ark']:.0f}%) "
              f"AND 낙폭 {p['dd']:.1f}% (필요 {p['need_dd']:.0f}% 이상)")
    print(f"오늘 동작: {action} {reason}")

    if o := os.environ.get("GITHUB_OUTPUT"):
        with open(o, "a") as f:
            f.write(f"action={action}\nreason={reason}\n")
            f.write(f"open_count={len(live)}\nrsi={st['rsi']}\n")
            f.write(f"all_mean={round((s_['all_mean'] or 0)*100,1)}\n")
            f.write(f"live_mean={round((s_['live_mean'] or 0)*100,1)}\n")
            # 같은 날 여러 포지션이 동시 매도될 수 있다(실제로 2023-06-20 에 6건).
            # 한 건만 보내면 나머지를 놓친다.
            if sold:
                ds = [d for d in st["closed"] if d["exit"] == st["price_date"]]
                f.write(f"sold_n={len(ds)}\n")
                f.write(f"sold_ret={round(float(np.mean([d['ret'] for d in ds]))*100,1)}\n")
                f.write("sold_list=" + " / ".join(
                    f"{d['entry']} ${d['entry_price']:.0f} {d['ret']*100:+.1f}%" for d in ds) + "\n")
            # 규칙 B 진행 상황.
            # 예전에는 "이번 주 2.74% / 발동 -7.0% 이하" 로만 보냈는데,
            # 2.74% 는 위쪽 매수 순매수율과 같은 숫자다. 같은 값을 두 방향으로
            # 재는 것이라(위: +문턱을 넘나 / 아래: -7% 아래로 가나) 라벨만 보고는
            # 어느 쪽으로 가야 발동인지 알 수 없었다. 방향과 남은 거리를 밝힌다.
            if cur_netpct <= BIG_SELL:
                b_line = f"발동 — 이번 주 {cur_netpct:.2f}% 매도"
                b_sub = ("보유 중이면 이번 주가 매도 시점이다"
                         if live else "보유 포지션이 없어 해당 없음")
            elif cur_netpct < 0:
                b_line = f"아크가 파는 중 ({cur_netpct:.2f}%)"
                b_sub = (f"발동은 {BIG_SELL}% 이하. "
                         f"{abs(cur_netpct - BIG_SELL):.2f}%p 남음")
            else:
                b_line = f"아크가 사는 중 (+{cur_netpct:.2f}%)"
                b_sub = (f"발동은 {BIG_SELL}% 이하로 팔 때. "
                         f"{abs(cur_netpct - BIG_SELL):.2f}%p 남음")
            f.write(f"bsell_line={b_line}\n")
            f.write(f"bsell_sub={b_sub}\n")
            f.write(f"cur_netpct={st.get('cur_netpct', 0)}\n")
            f.write(f"big_sell={BIG_SELL}\n")


if __name__ == "__main__":
    main()
