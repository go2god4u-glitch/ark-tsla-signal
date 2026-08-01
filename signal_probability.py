"""신호 발생 확률 — 주중에 미리 알기.

문제 정의:
  판정은 금요일에 확정된다. 그런데 화요일에 이미 순매수가 문턱의 80% 라면,
  금요일에 신호가 뜰 확률이 꽤 높을 것이다. 그 확률을 과거 데이터로 추정한다.

왜 이건 표본이 되는가:
  신호 사건은 8개뿐이지만, '주 × 요일' 관측은 1,000개가 넘는다.
  각 주의 각 시점에서 (그때까지의 진행률 -> 그 주가 신호로 끝났는가) 를 세면
  경험적 확률을 만들 수 있다. 수익률 예측이 아니라 '문턱 도달' 예측이므로
  주가 예측보다 훨씬 쉬운 문제다.

지키는 것:
  - 워크포워드. t 주의 확률은 t 주 이전 데이터로만 만든 표에서 읽는다.
    전체 표본으로 만든 표를 과거에 적용하면 미래를 쓰는 것이다.
  - 문턱도 그 시점의 확장창 분위수를 쓴다(signal_check 와 동일).
  - 보정(calibration) 을 확인한다. '90% 라고 말한 구간이 실제로 90% 였는가'.
    이게 맞지 않으면 확률이 아니라 그냥 숫자다.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MIN_TRAIN = 60          # 확률표를 만들기 위한 최소 과거 주 수
BINS = [-np.inf, 0, 0.25, 0.5, 0.75, 1.0, np.inf]
BIN_LABELS = ["음수", "0~25%", "25~50%", "50~75%", "75~100%", "100%↑"]


def load_prices() -> pd.Series:
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    return pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()


def build_panel(px: pd.Series) -> pd.DataFrame:
    """주 × 요일별 '그때까지의 진행률'과 '그 주의 최종 신호 여부'."""
    from signal_check import build_daily, MIN_HIST, QUANT, DD_FILTER
    daily = build_daily(px)
    wide = daily.attrs["wide"]

    # 주간 확정치 (signal_check.build 와 동일한 정의)
    wk = wide.resample("W-FRI").last()
    net_w = wk.diff().sum(axis=1, min_count=1)
    base_w = wk.shift(1).sum(axis=1, min_count=1)
    netpct_w = (net_w / base_w * 100).dropna()
    thr = netpct_w.expanding(MIN_HIST).quantile(QUANT).shift(1)
    # 낙폭 필터도 신호 조건이다. 낙폭은 주가만으로 정해지므로 주중에도 알 수 있고,
    # 필터에 걸린 주는 아무리 많이 사도 신호가 아니다 -> 확률 0.
    dd_w = ((px / px.rolling(252, min_periods=60).max() - 1) * 100
            ).reindex(netpct_w.index, method="ffill")
    sig = (netpct_w >= thr) & (dd_w <= DD_FILTER)

    # 일별 누적: 그 주 시작(직전 주 금요일) 대비
    lvl = wide.copy()
    week_id = lvl.index.to_period("W-FRI")
    prev_close = wk.sum(axis=1, min_count=1).shift(1)

    rows = []
    for wid, g in lvl.groupby(week_id):
        wend = wid.end_time.normalize()
        if wend not in netpct_w.index or pd.isna(thr.get(wend, np.nan)):
            continue
        base = prev_close.get(wend, np.nan)
        if pd.isna(base) or base <= 0:
            continue
        prev_row = wk.shift(1).loc[wend]
        for k, (dt, row) in enumerate(g.iterrows(), start=1):
            # 그날까지의 누적 순매수 = (오늘 각 펀드 - 직전 주 금요일 각 펀드) 합
            cum = (row - prev_row).sum(min_count=1)
            if pd.isna(cum):
                continue
            rows.append({"week": wend, "day": k, "date": dt,
                         "cum_pct": cum / base * 100,
                         "thr": thr[wend], "final": netpct_w[wend],
                         "dd": float(dd_w.get(wend, np.nan)),
                         "sig": bool(sig[wend])})
    return pd.DataFrame(rows)


def progress(p: pd.DataFrame) -> pd.Series:
    """문턱 대비 진행률. 문턱이 0 이하인 주는 비율이 무의미하므로 제외한다."""
    return np.where(p["thr"] > 0, p["cum_pct"] / p["thr"], np.nan)


def main() -> None:
    px = load_prices()
    p = build_panel(px)
    p["prog"] = progress(p)
    p = p.dropna(subset=["prog"])
    p["bin"] = pd.cut(p["prog"], BINS, labels=BIN_LABELS)
    weeks = sorted(p["week"].unique())

    print(f"관측 {len(p)}개 (주 {len(weeks)}개 × 요일)  "
          f"{p['date'].min():%Y-%m-%d} ~ {p['date'].max():%Y-%m-%d}")
    print(f"신호 주 {p.groupby('week')['sig'].first().sum()}개 "
          f"(기저 확률 {p.groupby('week')['sig'].first().mean()*100:.1f}%)\n")

    # ---- 워크포워드 예측: 각 주의 확률은 그 이전 주들로만 만든 표에서 읽는다 ----
    preds = []
    for i, wk in enumerate(weeks):
        if i < MIN_TRAIN:
            continue
        past = p[p["week"] < wk]
        tbl = past.groupby(["day", "bin"], observed=True)["sig"].agg(["mean", "size"])
        cur = p[p["week"] == wk]
        for _, r in cur.iterrows():
            key = (r["day"], r["bin"])
            if key in tbl.index and tbl.loc[key, "size"] >= 8:
                preds.append({"week": wk, "day": int(r["day"]), "bin": str(r["bin"]),
                              "prob": float(tbl.loc[key, "mean"]),
                              "n_train": int(tbl.loc[key, "size"]),
                              "actual": bool(r["sig"])})
    pr = pd.DataFrame(preds)

    print("=" * 84)
    print("1. 요일 × 진행률별 신호 확률 (전체 표본 기준 표)")
    print("=" * 84)
    tbl = p.pivot_table(index="bin", columns="day", values="sig",
                        aggfunc=["mean", "size"], observed=True)
    m = (tbl["mean"] * 100).round(0)
    n = tbl["size"]
    print(f"{'진행률':<10s}" + "".join(f"{f'{d}일차':>14s}" for d in m.columns))
    for b in BIN_LABELS:
        if b not in m.index:
            continue
        cells = "".join(
            (f"{m.loc[b, d]:>9.0f}% (n={int(n.loc[b, d])})" if not pd.isna(m.loc[b, d])
             else f"{'-':>14s}") for d in m.columns)
        print(f"{b:<10s}" + cells)

    print()
    print("=" * 84)
    print("2. 보정 확인 — 워크포워드 예측이 실제와 맞는가")
    print("=" * 84)
    if len(pr):
        pr["pbin"] = pd.cut(pr["prob"], [-.01, .1, .3, .5, .7, .9, 1.01],
                            labels=["0~10%", "10~30%", "30~50%", "50~70%", "70~90%", "90%↑"])
        cal = pr.groupby("pbin", observed=True).agg(
            예측평균=("prob", "mean"), 실제비율=("actual", "mean"), 관측수=("actual", "size"))
        cal["예측평균"] = (cal["예측평균"] * 100).round(0)
        cal["실제비율"] = (cal["실제비율"] * 100).round(0)
        print(cal.to_string())
        brier = float(((pr["prob"] - pr["actual"].astype(float)) ** 2).mean())
        base = p.groupby("week")["sig"].first().mean()
        brier0 = float(((base - pr["actual"].astype(float)) ** 2).mean())
        print(f"\n  Brier 점수 {brier:.4f} (낮을수록 좋음) / 기저확률만 쓰면 {brier0:.4f}")
        print(f"  개선율 {(1-brier/brier0)*100:+.1f}%  — 양수면 예측이 기저보다 낫다")
    else:
        print("  워크포워드 표본 부족")

    # ---- 현재 주 상태 ----
    last_week = weeks[-1]
    cur = p[p["week"] == last_week].sort_values("day")
    past = p[p["week"] < last_week]
    tbl2 = past.groupby(["day", "bin"], observed=True)["sig"].agg(["mean", "size"])
    print()
    print("=" * 84)
    print(f"3. 이번 주 ({pd.Timestamp(last_week):%Y-%m-%d} 마감 예정) 진행 상황")
    print("=" * 84)
    out_now = []
    for _, r in cur.iterrows():
        key = (r["day"], r["bin"])
        prob = (float(tbl2.loc[key, "mean"]) if key in tbl2.index
                and tbl2.loc[key, "size"] >= 8 else None)
        out_now.append({"date": r["date"].strftime("%Y-%m-%d"), "day": int(r["day"]),
                        "cum_pct": round(float(r["cum_pct"]), 2),
                        "thr": round(float(r["thr"]), 2),
                        "prog": round(float(r["prog"]) * 100, 0),
                        "prob": None if prob is None else round(prob * 100, 0)})
        print(f"  {r['date']:%Y-%m-%d} ({int(r['day'])}일차)  누적 {r['cum_pct']:+.2f}% "
              f"/ 문턱 {r['thr']:.2f}%  진행률 {r['prog']*100:>3.0f}%  "
              + (f"신호 확률 {prob*100:.0f}%" if prob is not None else "표본부족"))

    table_out = []
    for b in BIN_LABELS:
        if b not in m.index:
            continue
        for d in m.columns:
            if not pd.isna(m.loc[b, d]):
                table_out.append({"bin": b, "day": int(d),
                                  "prob": float(m.loc[b, d]),
                                  "n": int(n.loc[b, d])})
    with open(f"{BASE}/data/signal_probability.json", "w") as f:
        json.dump({"table": table_out, "current": out_now,
                   "base_rate": float(p.groupby("week")["sig"].first().mean()),
                   "brier": brier if len(pr) else None,
                   "brier_base": brier0 if len(pr) else None}, f,
                  ensure_ascii=True, indent=2, default=float)
    print("\n저장: data/signal_probability.json")

    # 확률은 이 스텝이 직접 내보낸다.
    # signal_check 가 이 파일을 읽게 하면 실행 순서상 하루 전 값을 쓰게 된다
    # (signal_check 가 먼저 돌고 이 파일은 그 뒤에 갱신되므로).
    #
    # 알림 문구를 여기서 만든다. 예전에는 확률만 넘기고 진행률은 signal_check 것을
    # 썼는데, 둘이 서로 다른 날 값이라 "진행률 82% 인데 확률 0%" 처럼 나갔다.
    # 최신 날짜가 표본 부족이면 확률을 이전 날로 물리게 되므로, 어느 날 값인지
    # 두 줄 모두에 밝힌다.
    if o := os.environ.get("GITHUB_OUTPUT"):
        cur_day = out_now[-1] if out_now else None
        fb = next((c for c in reversed(out_now) if c.get("prob") is not None), None)
        if cur_day is None:
            line, sub = "계산 불가", "이번 주 관측 없음"
        elif cur_day.get("prob") is not None:
            line = f"{cur_day['prob']:.0f}% ({cur_day['day']}일차, 진행률 {cur_day['prog']:.0f}%)"
            sub = (f"누적 {cur_day['cum_pct']:+.2f}% / 문턱 {cur_day['thr']:.2f}%")
        else:
            line = f"표본부족 ({cur_day['day']}일차 진행률 {cur_day['prog']:.0f}%)"
            sub = (f"{fb['day']}일차 기준으로는 {fb['prob']:.0f}% 였음 "
                   f"(진행률 {fb['prog']:.0f}%)" if fb else
                   "이번 주는 아직 비교할 과거 표본이 없다")
        with open(o, "a") as f:
            f.write(f"prob_line={line}\n")
            f.write(f"prob_sub={sub}\n")
            # 화면·기존 참조용으로 원시값도 남긴다
            f.write(f"prob={fb['prob'] if fb else '-'}\n")
            f.write(f"prog={cur_day['prog'] if cur_day else '-'}\n")
            f.write(f"cum_pct={cur_day['cum_pct'] if cur_day else '-'}\n")
            f.write(f"day={cur_day['day'] if cur_day else '-'}\n")


if __name__ == "__main__":
    main()
