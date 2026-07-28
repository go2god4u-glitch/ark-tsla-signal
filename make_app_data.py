"""대시보드용 데이터 생성 — 사건(episode) 단위 분석.

앞선 버전들의 결함과 수정 이력:

1) 룩어헤드 기준선 (수정됨)
   '상위 10%' 문턱을 5년 전체로 계산했다. 실전에서는 미래를 모른다.
   -> 확장창(최소 52주) 분위수를 한 주 시프트해 과거만 쓴다.

2) 겹치는 관측을 독립으로 셈 (이번에 수정)
   일/주 단위로 세면 같은 국면의 날들이 전부 독립 관측으로 잡힌다.
   낙폭 -50% '92일'은 실제로는 서로 붙어 있는 3개 국면이고,
   -60% '28일'은 국면 1개다. 이걸 n=92, n=28 로 취급하면
   p 값이 0.000 까지 내려가지만 그건 통계가 아니라 중복이다.
   -> 30거래일 이상 떨어진 것만 별개 사건으로 묶고, 각 사건의
      '첫날'에 한 번만 진입한 것으로 계산한다.

   이 수정으로 결론이 뒤집혔다. 주 단위에서는 낙폭 단독이 아크보다
   좋아 보였지만(아크를 볼 필요 없다), 사건 단위로는 아크가 낫다.
   아크 신호는 9개 국면에 흩어져 뜨는 반면 낙폭 -50% 는 3개뿐이다.
"""
import json, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260728)
HOR = [63, 126, 252]
GAP = 30          # 이만큼 떨어지면 별개 사건
NB = 20000

J = json.load(open(f"{BASE}/data/tsla_yahoo.json"))["chart"]["result"][0]
px = pd.Series(J["indicators"]["quote"][0]["close"],
      index=pd.to_datetime(pd.Series(J["timestamp"]), unit="s").dt.normalize()).dropna().sort_index()
V = px.values

REFS = {"h126": ("126일 고점", px.rolling(126, min_periods=60).max()),
        "h252": ("252일 고점", px.rolling(252, min_periods=60).max()),
        "h504": ("504일 고점", px.rolling(504, min_periods=60).max()),
        "ath":  ("사상 최고가", px.cummax())}
dd252 = (px / REFS["h252"][1] - 1) * 100

d = pd.read_csv(f"{BASE}/data/holdings_clean.csv", parse_dates=["date"]).set_index("date")
d["close"] = d["TSLA_close"].ffill()
d["dd"] = dd252.reindex(d.index).ffill()
w = d.resample("W-FRI").last().dropna(subset=["combined_shares_adj"])
w["net"] = w["combined_shares_adj"].diff()
w["dd"] = dd252.reindex(w.index).ffill()
w["thr"] = w["net"].expanding(52).quantile(0.90).shift(1)
w = w.dropna(subset=["thr", "net"])
w["sig"] = w["net"] >= w["thr"]
ark = w[w["sig"]]


def episodes(locs):
    """연속/근접한 진입 시점을 하나의 사건으로 묶고 각 사건의 첫날만 남긴다."""
    locs = sorted(set(int(i) for i in locs if 0 <= i < len(V)))
    if not locs:
        return []
    out, start, prev = [], locs[0], locs[0]
    for i in locs[1:]:
        if i - prev > GAP:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def stats(starts, h):
    done = [i for i in starts if i + h < len(V)]
    if not done:
        return None
    r = np.array([V[i + h] / V[i] - 1 for i in done])
    pool = np.arange(len(V) - h)
    b = V[pool + h] / V[pool] - 1
    sims = np.array([b[RNG.integers(0, len(b), len(r))].mean() for _ in range(NB)])
    return {"n": len(r), "mean": float(r.mean()), "base_mean": float(b.mean()),
            "edge": float(r.mean() - b.mean()), "hit": float((r > 0).mean()),
            "p": float((sims >= r.mean()).mean())}


ark_eps = episodes([px.index.searchsorted(t, side="right") for t in ark.index])
ark_starts = [a for a, _ in ark_eps]

CANDS = [("ark", "아크 매수신호", ark_starts)]
for th in (-30, -40, -50):
    st = [a for a, _ in episodes(np.where((dd252 <= th).values)[0])]
    CANDS.append((f"dd{abs(th)}", f"낙폭 {th}% 이하 (아크 무시)", st))

compare = []
for cid, label, starts in CANDS:
    row = {"id": cid, "label": label, "eps": len(starts), "h": {}}
    for h in HOR:
        s = stats(starts, h)
        if s:
            row["h"][str(h)] = s
    row["outcomes"] = [None if a + 126 >= len(V) else round(float(V[a + 126] / V[a] - 1), 4)
                       for a in starts]
    row["dates"] = [px.index[a].strftime("%Y-%m-%d") for a in starts]
    compare.append(row)

# 기준 고점 민감도: 방향이 기준 선택에 좌우되지 않는지. 사건 수를 함께 싣는다.
grid = []
for rid, (rlabel, ref) in REFS.items():
    dd = (px / ref - 1) * 100
    for th in (-30, -40, -50, -60):
        st = [a for a, _ in episodes(np.where((dd <= th).values)[0])]
        s = stats(st, 126)
        grid.append({"ref": rlabel, "th": th, "eps": len(st),
                     "edge": s["edge"] if s else None, "hit": s["hit"] if s else None,
                     "n": s["n"] if s else 0})

# 아크 사건별 상세 + 2023-01 제외 민감도
hist = []
for a, b in ark_eps:
    hist.append({"date": px.index[a].strftime("%Y-%m-%d"), "px": float(V[a]),
                 "dd": float(dd252.iloc[a]), "weeks": int(round((b - a) / 5)) + 1,
                 "r3": (float(V[a + 63] / V[a] - 1) if a + 63 < len(V) else None),
                 "r6": (float(V[a + 126] / V[a] - 1) if a + 126 < len(V) else None),
                 "r12": (float(V[a + 252] / V[a] - 1) if a + 252 < len(V) else None)})
ex = [h for h in hist if h["r6"] is not None and not h["date"].startswith("2023-01")]
ark_ex = {"n": len(ex), "mean": float(np.mean([h["r6"] for h in ex])),
          "hit": float(np.mean([h["r6"] > 0 for h in ex]))}

lag = []
for L, lab in [(0, "신호 다음날"), (5, "+1주"), (10, "+2주"), (21, "+1달"), (63, "+3달")]:
    s = stats([a + L for a in ark_starts], 126)
    if s:
        lag.append({"label": lab, **s})

pre = []
for k, lab in [(5, "직전 1주"), (21, "직전 1달"), (63, "직전 3달")]:
    s = [V[i] / V[i - k] - 1 for i in ark_starts if i - k >= 0]
    a = [V[i] / V[i - k] - 1 for i in range(k, len(V))]
    pre.append({"label": lab, "sig": float(np.mean(s)), "base": float(np.mean(a))})

base = d.index.min()
out = {"base": base.strftime("%Y-%m-%d"),
  "t": [int((x - base).days) for x in d.index],
  "sh": [round(v / 1000, 1) for v in d["combined_shares_adj"]],
  "p": [None if pd.isna(v) else round(v, 2) for v in d["TSLA_close"]],
  "dd": [None if pd.isna(v) else round(v, 1) for v in d["dd"]],
  "wt": [int((x - base).days) for x in w.index],
  "wnet": [None if pd.isna(v) else round(v / 1000, 1) for v in w["net"]],
  "wthr": [round(v / 1000, 1) for v in w["thr"]],
  "compare": compare, "grid": grid, "hist": hist, "arkEx": ark_ex,
  "lag": lag, "pre": pre, "gap": GAP,
  "now": {"thr": float(w["thr"].iloc[-1] / 1000), "net": float(w["net"].iloc[-1] / 1000),
          "on": bool(w["sig"].iloc[-1]), "px": float(V[-1]),
          "date": w.index[-1].strftime("%Y-%m-%d"),
          "ath": float(px.max()), "athDate": px.idxmax().strftime("%Y-%m-%d"),
          "ddNow": float(V[-1] / px.max() - 1) * 100},
  "sigDates": [px.index[a].strftime("%Y-%m-%d") for a in ark_starts]}
open(f"{BASE}/data/app_data.json", "w").write(json.dumps(out, separators=(",", ":")))

print(f"아크 신호 {len(ark)}주 -> 독립 사건 {len(ark_starts)}개")
for c in compare:
    s = c["h"].get("126")
    print(f"  {c['label']:<24s} 사건 {c['eps']:>2d}개  "
          + (f"6개월 n={s['n']} 평균 {s['mean']*100:+6.1f}% 초과 {s['edge']*100:+6.1f}% "
             f"승률 {s['hit']*100:3.0f}% p={s['p']:.3f}" if s else "표본부족"))
print(f"  2023-01 제외 시 아크: n={ark_ex['n']} 평균 {ark_ex['mean']*100:+.1f}% 승률 {ark_ex['hit']*100:.0f}%")
print(f"현재 {'ON' if out['now']['on'] else 'OFF'} / 사상최고 대비 {out['now']['ddNow']:.1f}%")
print("저장:", os.path.getsize(f"{BASE}/data/app_data.json"), "bytes")
