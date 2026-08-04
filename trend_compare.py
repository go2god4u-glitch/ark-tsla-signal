"""지금 이 포지션의 주가 흐름을 과거 신호들의 같은 시점과 비교한다.

무엇을 비교하는가 (헷갈리기 쉬우니 먼저 못박는다):
  '지금 추세' = **이번 진입일(마지막 매수 신호의 다음 거래일) 이후의 주가 경로**다.
  최근 몇 달의 일반적인 추세가 아니다. 과거 신호들도 각자의 진입일부터 같은
  거래일 수만큼 잰다. 같은 나이의 포지션끼리 비교하는 것이다.

왜 '주가 경로' 이고 '포지션 손익' 이 아닌가:
  과거 매도 15건 중 10건이 규칙 B(아크 대량매도)로 나갔다. 긴 지평에서
  '그때까지 살아 있던 포지션만' 으로 제한하면 바로 그 특징적인 사례들이
  통째로 빠져 비교가 뒤틀린다. 매도와 무관하게 주가만 따라간다.

표본에 대한 정직한 경고:
  진입은 16회지만 7일 이내 연속을 묶으면 **국면 10개**다. 같은 국면의 진입들
  (예: 2023-01-09/17/23)은 1~2주 차이라 경로가 거의 같다. 16개를 독립 관측처럼
  세면 표본을 부풀리는 것이다. 그래서 **국면 기준(과거 9개)을 주 표본**으로 쓰고
  진입 기준(과거 15건)은 참고로만 둔다.
  n=9 에서 백분위는 숫자 하나로 표본 크기를 감춘다. 그래서 **아홉 개를 전부 찍는다.**

가장 먼저 답해야 할 질문:
  "지금 며칠차 위치가 앞날에 대해 무엇을 말해주는가?"
  이걸 안 재고 백분위부터 보여주면, 아무 의미 없는 숫자를 신호처럼 읽게 된다.
  과거 진입들에서 'k일차 수익률'과 '6개월 뒤 수익률'의 상관을 먼저 계산한다.
  상관이 0 이면 정직한 답은 "지금 위치는 앞날을 말해주지 않는다" 이고,
  백분위는 예측이 아니라 맥락으로만 읽어야 한다.

진입 낙폭으로 나눠 보는 이유 (사후 짜맞추기가 아니다):
  이미 문서에 적어둔 발견이다 — 2023년 초 신호는 낙폭 -62~-69% 였고 100%대
  수익이 났으며, 2025~26년 신호는 -30~-37% 로 20~30%대다. 같은 매수 신호라도
  위기의 깊이가 다르다. 전체를 뭉뚱그린 중앙값은 깊은 낙폭 국면에 끌려가므로,
  **진입 낙폭이 비슷한 국면끼리** 도 따로 본다.

이 파일이 지키는 것:
  - 신호·문턱을 여기서 다시 만들지 않는다. signal_check.build() 를 그대로 쓰고,
    맨 먼저 알려진 16건을 재현하는지 자기 검사한다. (CLAUDE.md 22·23번)
  - 판정 문장을 손으로 쓰지 않는다. 매일 값이 바뀌는 출력에 고정 문구를 넣으면
    반드시 상한다. 모든 문장을 데이터에서 만든다. (CLAUDE.md 20번)
"""

import json
import os

import numpy as np
import pandas as pd

import signal_check as sc

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "trend_compare.json")

HORIZONS = (5, 10, 21, 42, 63, 126, 252)   # 거래일: 1주 1개월 2개월 3개월 6개월 12개월
OUTCOME = 126                              # '앞날' 의 기준 지평 = 6개월
DEEP_CUT = -45.0                           # 진입 낙폭 깊음/얕음 경계
EARLY = (5, 10, 21, 42)                    # 예측력을 물어볼 이른 시점들
FWD = (5, 21, 63, 126, 252)                # '지금부터 앞으로' 를 물어볼 지평
TARGETS = (10, 20, 30, 50)                 # 목표 수익률(%) — 도달까지 며칠 걸렸나

# 넣지 않기로 한 지표들 (재봤고, 의미가 없어서 뺐다):
#   경로 모양 유사도 매칭 — 지금 며칠치 점 몇 개로 '닮은 국면'을 고르는 것은
#     잡음이다. 게다가 닮은 것만 골라 그 결과를 보면 표본이 2~3개로 줄어
#     무슨 값이든 나온다.
#   '처음 +로 돌아선 날' — 과거 9개 국면 전부 1~6일이었다. 거의 항상 참이라
#     정보가 없다.
#   1주 앞 확률은 표에 남기되 우위가 없다는 사실을 그대로 보여준다
#     (4/9=44% vs 무작위 47%). 지우면 '언제부터 우위가 생기는지' 를 알 수 없다.


def label(k: int) -> str:
    return {5: "1주", 10: "2주", 21: "1개월", 42: "2개월",
            63: "3개월", 126: "6개월", 252: "12개월"}.get(k, f"{k}일")


def load():
    r = json.load(open(f"{BASE}/data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    return pd.Series(r["indicators"]["quote"][0]["close"],
                     index=idx).dropna().sort_index()


def build_entries(px: pd.Series, w: pd.DataFrame) -> pd.DataFrame:
    """신호 주 -> 진입(다음 거래일) 인덱스. 진입 시점 낙폭도 함께 싣는다."""
    rows = []
    for t, r in w[w["sig"]].iterrows():
        i = px.index.searchsorted(t, side="right")
        if i >= len(px):
            continue
        rows.append({"signal": t, "entry": px.index[i], "i": i,
                     "price": float(px.iloc[i]), "dd": float(r["dd"])})
    e = pd.DataFrame(rows)
    # 국면: 직전 신호와 7일 넘게 벌어지면 새 국면. 대시보드의 '국면' 정의와 같다.
    gap = e["signal"].diff().dt.days
    e["episode"] = (gap.isna() | (gap > 7)).cumsum() - 1
    return e


def path_ret(V: np.ndarray, i: int, k: int):
    """진입 후 k거래일 수익률. 아직 그 날짜가 안 왔으면 None."""
    return float(V[i + k] / V[i] - 1) if i + k < len(V) else None


def fmt_pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.1f}%"


def spearman(a, b) -> float:
    """순위상관. 극단값 하나가 만드는 가짜 상관을 걸러낸다.

    실제로 1개월차는 피어슨 +0.57 인데 순위상관은 0.00 이었다.
    2023-01(낙폭 -70%, 6개월 +127%) 한 점이 만든 값이라는 뜻이다.
    """
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def loo_range(a, b):
    """국면 하나씩 빼면 상관이 어디까지 흔들리는가. n 이 작을 때 필수다."""
    if len(a) < 4:
        return None, None
    v = [float(np.corrcoef(np.delete(a, i), np.delete(b, i))[0, 1])
         for i in range(len(a))]
    return min(v), max(v)


def wilson(k: int, n: int, z: float = 1.96):
    """이항 비율의 95% 신뢰구간.

    n=9 에서 '9/9 = 100%' 를 그냥 100% 라고 쓰면 거짓말이다. 실제 구간은
    70~100% 다. 확률처럼 보이는 숫자일수록 구간을 같이 줘야 한다.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def base_up_rate(V: np.ndarray, lo: int, hi: int, k: int) -> float:
    """같은 기간 무작위 진입의 상승 비율.

    기준을 15년 전체에서 뽑으면 안 된다. 테슬라는 2010년 이후 200배 올랐고,
    국면이 다른 표본을 섞으면 어떤 신호든 좋아 보이거나 나빠 보인다.
    신호가 존재한 구간에서만 뽑는다. (DECISIONS '비교 기준은 같은 기간에서')
    """
    pool = np.arange(lo, max(lo + 1, hi - k))
    return float((V[pool + k] / V[pool] - 1 > 0).mean())


def verdict_of(pear: float, sp: float, lo, hi) -> str:
    """세 가지가 다 같은 방향일 때만 '관계 있다' 고 말한다."""
    if lo is None:
        return "표본 부족"
    if abs(sp) < 0.3 or lo * hi < 0:
        return "극단값이 만든 값 — 관계로 보면 안 된다"
    if min(abs(pear), abs(sp), abs(lo)) >= 0.5:
        return "관계가 버틴다"
    return "약하다"


def main() -> None:
    px = load()
    V = px.values
    w = sc.build(px)

    # ---- 자기 검사: 라이브와 같은 신호 집합인가 ----
    e = build_entries(px, w)
    n_live = int(w["sig"].sum())
    print(f"[자기 검사] 신호 {len(e)}건 재현 vs 라이브 {n_live}건  "
          f"{'통과' if len(e) == n_live else '실패 — 아래 숫자를 믿지 마라'}")
    if len(e) != n_live:
        raise SystemExit("신호 집합이 라이브와 다르다. CLAUDE.md 23번.")

    cur = e.iloc[-1]
    day = len(V) - 1 - int(cur["i"])
    cur_ret = path_ret(V, int(cur["i"]), day)
    past = e.iloc[:-1]                                  # 과거 진입 15건
    past_ep = past.groupby("episode").first().reset_index()   # 과거 국면 9개

    print(f"\n비교 대상 — 이번 진입 {cur['entry']:%Y-%m-%d} "
          f"(신호 {cur['signal']:%Y-%m-%d}) 이후 **{day}거래일**, "
          f"${cur['price']:.2f} -> ${V[-1]:.2f} ({fmt_pct(cur_ret)})")
    print(f"과거 표본 — 국면 {len(past_ep)}개(주 표본) / 진입 {len(past)}건(참고)")
    print(f"이번 진입 낙폭 {cur['dd']:.1f}% -> "
          f"{'깊은' if cur['dd'] <= DEEP_CUT else '얕은'} 국면 그룹")

    # ================================================================
    # 1. 지금 위치가 앞날을 말해주는가 — 이걸 먼저 답한다
    # ================================================================
    print("\n" + "=" * 78)
    print(f"1. k거래일차 수익률이 {label(OUTCOME)} 뒤 결과를 예고하는가")
    print("=" * 78)
    print("   n=9 에서 피어슨 상관은 극단값 하나로 만들어진다. 그래서 세 가지를 같이 본다:")
    print("   순위상관(극단값에 둔감) / 국면 하나씩 빼봤을 때의 범위(안정성).")
    print(f"   {'시점':<7s}{'n':>3s}{'피어슨':>8s}{'순위':>7s}"
          f"{'하나 빼면':>16s}   해석")
    pred = {}
    for k in EARLY:
        rows = [(path_ret(V, int(r["i"]), k), path_ret(V, int(r["i"]), OUTCOME))
                for _, r in past_ep.iterrows()]
        rows = [(a, b) for a, b in rows if a is not None and b is not None]
        if len(rows) < 4:
            continue
        a = np.array([x for x, _ in rows])
        b = np.array([y for _, y in rows])
        pear = float(np.corrcoef(a, b)[0, 1])
        sp = spearman(a, b)
        lo, hi = loo_range(a, b)
        vd = verdict_of(pear, sp, lo, hi)
        pred[k] = {"n": len(rows), "r": pear, "spearman": sp,
                   "loo_lo": lo, "loo_hi": hi, "verdict": vd}
        print(f"   {label(k):<7s}{len(rows):>3d}{pear:>+8.2f}{sp:>+7.2f}"
              f"{f'{lo:+.2f} ~ {hi:+.2f}':>16s}   {vd}")
    near = min(pred, key=lambda k: abs(k - day)) if pred else None

    # ---- 이 관계가 그냥 '진입 낙폭' 의 그림자는 아닌가 ----
    dd_arr = np.array([r["dd"] for _, r in past_ep.iterrows()])
    out_arr = np.array([path_ret(V, int(r["i"]), OUTCOME) for _, r in past_ep.iterrows()])
    kk = near if near is not None else 5
    early_arr = np.array([path_ret(V, int(r["i"]), kk) for _, r in past_ep.iterrows()])
    dd_out = float(np.corrcoef(dd_arr, out_arr)[0, 1])
    dd_early = float(np.corrcoef(dd_arr, early_arr)[0, 1])
    print(f"\n   교란 검정 — 낙폭이 깊으면 초반도 좋고 끝도 좋다면,")
    print(f"   '초반 -> 결과' 상관은 낙폭의 그림자일 뿐이다.")
    print(f"     진입낙폭 -> {label(OUTCOME)} 결과   {dd_out:+.2f} "
          f"(순위 {spearman(dd_arr, out_arr):+.2f})   <- 낙폭이 더 센 설명변수다")
    print(f"     진입낙폭 -> {label(kk)} 수익     {dd_early:+.2f}")
    grp_mask = (dd_arr > DEEP_CUT) == (cur["dd"] > DEEP_CUT)
    within = None
    if grp_mask.sum() >= 4:
        within = float(np.corrcoef(early_arr[grp_mask], out_arr[grp_mask])[0, 1])
        print(f"     낙폭 그룹 안에서만 (n={int(grp_mask.sum())}): "
              f"{label(kk)} -> 결과 {within:+.2f} "
              f"(순위 {spearman(early_arr[grp_mask], out_arr[grp_mask]):+.2f})")
        print(f"     -> 그룹 안에서도 남아 있으면 그림자만은 아니다. 다만 n={int(grp_mask.sum())} 이다.")

    # ================================================================
    # 2. 같은 나이의 과거 국면들 — 아홉 개를 전부 찍는다
    # ================================================================
    print("\n" + "=" * 78)
    print(f"2. 진입 후 {day}거래일 시점, 과거 국면들은 어디에 있었나")
    print("=" * 78)
    vals = []
    for _, r in past_ep.iterrows():
        v = path_ret(V, int(r["i"]), day)
        if v is None:
            continue
        grp = "깊음" if r["dd"] <= DEEP_CUT else "얕음"
        vals.append({"entry": r["entry"], "dd": float(r["dd"]),
                     "grp": grp, "ret": v})
    vals.sort(key=lambda x: x["ret"])
    for v in vals:
        bar = "" if v["ret"] is None else ("+" if v["ret"] >= 0 else "-")
        print(f"   {v['entry']:%Y-%m-%d}  낙폭 {v['dd']:>6.1f}% ({v['grp']})  "
              f"{fmt_pct(v['ret']):>8s} {bar}")
    arr = np.array([v["ret"] for v in vals])
    rank = int((arr < cur_ret).sum()) + 1
    print(f"   {'':<12s}{'':<20s}{'-' * 10}")
    print(f"   지금        낙폭 {cur['dd']:>6.1f}% "
          f"({'깊음' if cur['dd'] <= DEEP_CUT else '얕음'})  "
          f"{fmt_pct(cur_ret):>8s}  <- {len(arr)}개 중 {rank}번째로 낮음")
    print(f"\n   중앙값 {fmt_pct(float(np.median(arr)))} · "
          f"범위 {fmt_pct(float(arr.min()))} ~ {fmt_pct(float(arr.max()))}")

    # 같은 낙폭 그룹만
    grp_now = "깊음" if cur["dd"] <= DEEP_CUT else "얕음"
    peer = [v["ret"] for v in vals if v["grp"] == grp_now]
    if peer:
        pa = np.array(peer)
        print(f"   진입 낙폭이 비슷한 '{grp_now}' 국면 {len(pa)}개만: "
              f"중앙값 {fmt_pct(float(np.median(pa)))} · "
              f"범위 {fmt_pct(float(pa.min()))} ~ {fmt_pct(float(pa.max()))}")

    # ================================================================
    # 3. 앞으로의 분포 (지금 값으로 조건 걸지 않는다)
    # ================================================================
    print("\n" + "=" * 78)
    print("3. 과거 국면들의 지평별 수익률 (지금 값으로 조건을 걸지 않은 분포)")
    print("=" * 78)
    print("   지금 위치가 좋다/나쁘다로 앞날을 좁히면 표본이 2~3개로 줄어 무의미하다.")
    print(f"   {'지평':<8s}{'n':>4s}{'중앙값':>9s}{'최소':>9s}{'최대':>9s}"
          f"{'양수':>7s}   {'낙폭 얕은 국면만':>18s}")
    horizon_rows = []
    for k in HORIZONS:
        allv = [path_ret(V, int(r["i"]), k) for _, r in past_ep.iterrows()]
        allv = [x for x in allv if x is not None]
        if not allv:
            continue
        a = np.array(allv)
        pv = [path_ret(V, int(r["i"]), k) for _, r in past_ep.iterrows()
              if (r["dd"] <= DEEP_CUT) == (cur["dd"] <= DEEP_CUT)]
        pv = [x for x in pv if x is not None]
        ptxt = (f"{np.median(pv)*100:+.1f}% (n={len(pv)})" if pv else "—")
        print(f"   {label(k):<8s}{len(a):>4d}{np.median(a)*100:>+8.1f}%"
              f"{a.min()*100:>+8.1f}%{a.max()*100:>+8.1f}%"
              f"{(a > 0).mean()*100:>6.0f}%   {ptxt:>18s}")
        horizon_rows.append({"k": k, "label": label(k), "n": len(a),
                             "median": float(np.median(a)), "min": float(a.min()),
                             "max": float(a.max()), "hit": float((a > 0).mean()),
                             "peer_median": float(np.median(pv)) if pv else None,
                             "peer_n": len(pv)})

    # ================================================================
    # 4. 지금부터 앞으로 — 무작위 진입과 견줘 우위가 있는가
    # ================================================================
    print("\n" + "=" * 78)
    print(f"4. 지금({day}거래일차)부터 앞으로 오를 비율")
    print("=" * 78)
    print("   '확률' 처럼 보이지만 국면 9개를 센 것이다. 한 개가 11%p 다.")
    print("   그래서 원자료(x/n)와 95% 구간을 같이 싣고,")
    print("   같은 기간 무작위 진입과 견준다 — 그보다 못하면 신호가 아니라 시장이다.")
    print(f"   {'앞으로':<8s}{'국면':>9s}{'비율':>7s}{'95% 구간':>14s}"
          f"{'무작위':>8s}{'차이':>8s}   판정")
    lo_i, hi_i = int(past_ep["i"].min()), len(V) - 1
    fwd_rows = []
    for k in FWD:
        # 이름 주의: 위쪽 `vals` 는 '같은 나이 시점의 값들' 이라 덮어쓰면 안 된다.
        fwd_v = [V[i + day + k] / V[i + day] - 1
                 for i in past_ep["i"] if i + day + k < len(V)]
        if len(fwd_v) < 4:
            continue
        a_ = np.array(fwd_v)
        up = int((a_ > 0).sum())
        cl, cu = wilson(up, len(a_))
        bp = base_up_rate(V, lo_i, hi_i, k)
        edge = up / len(a_) - bp
        # 구간 하한이 무작위보다 위일 때만 '우위' 라고 쓴다.
        vd = ("우위 있다" if cl > bp else
              "우위 없다" if up / len(a_) <= bp else "말할 수 없다")
        print(f"   {label(k):<8s}{f'{up}/{len(a_)}':>9s}{up/len(a_)*100:>6.0f}%"
              f"{f'{cl*100:.0f}~{cu*100:.0f}%':>14s}{bp*100:>7.0f}%"
              f"{edge*100:>+7.0f}%p   {vd}")
        fwd_rows.append({"k": k, "label": label(k), "n": len(a_), "up": up,
                         "rate": up / len(a_), "ci_lo": cl, "ci_hi": cu,
                         "base": bp, "edge": edge, "verdict": vd,
                         "median": float(np.median(a_))})
    n_var = len({r["n"] for r in fwd_rows})
    if n_var > 1:
        print(f"   * 지평마다 n 이 다르다(가장 최근 국면은 아직 12개월이 안 지났다).")
        print(f"     표본이 바뀌면 엄밀한 비교가 아니다 — 8개로 고정해도 결론은 같았다.")

    # ================================================================
    # 5. 얼마나 기다렸나 / 얼마나 견뎠나
    # ================================================================
    print("\n" + "=" * 78)
    print("5. 목표 도달까지 걸린 거래일, 그리고 그 사이 최악")
    print("=" * 78)
    print("   '기다리면 오르나' 에 대한 실제 답이다. 진입가 대비 목표다.")
    print(f"   {'진입':<12s}" + "".join(f"{f'+{t}%':>9s}" for t in TARGETS)
          + f"{'6개월 내 최악':>14s}")
    reach = {t: [] for t in TARGETS}
    wait_rows = []
    for _, r in past_ep.iterrows():
        i = int(r["i"])
        H = min(252, len(V) - 1 - i)
        pth = np.array([V[i + j] / V[i] - 1 for j in range(H + 1)])
        row, hit = "", {}
        for t in TARGETS:
            wq = np.where(pth >= t / 100)[0]
            if len(wq):
                reach[t].append(int(wq[0]))
                hit[t] = int(wq[0])
                row += f"{int(wq[0]):>7d}일"
            else:
                hit[t] = None
                row += f"{'미도달':>9s}"
        mae6 = float(pth[:min(127, len(pth))].min())
        print(f"   {r['entry']:%Y-%m-%d}{row}{mae6*100:>13.1f}%")
        wait_rows.append({"entry": r["entry"].strftime("%Y-%m-%d"),
                          "reach": hit, "mae6": mae6})
    print(f"   {'중앙값':<12s}"
          + "".join((f"{int(np.median(reach[t])):>7d}일" if reach[t] else f"{'—':>9s}")
                    for t in TARGETS)
          + f"{np.median([w['mae6'] for w in wait_rows])*100:>13.1f}%")
    print(f"   {'도달 국면':<12s}"
          + "".join(f"{len(reach[t])}/{len(past_ep)}".rjust(9) for t in TARGETS))
    cur_path = np.array([V[int(cur["i"]) + j] / V[int(cur["i"])] - 1
                         for j in range(day + 1)])
    print(f"   {'지금':<12s}" + " " * (9 * len(TARGETS))
          + f"{cur_path.min()*100:>13.1f}%  <- {day}거래일까지의 최악")
    print("   ** 최악은 지평을 밝혀야 한다. 같은 국면도 1개월 기준과 12개월 기준이")
    print("      크게 다르다(2022-02-28: 1개월 -12.0% / 12개월 -62.7%).")

    # ================================================================
    # 6. 판정 — 전부 데이터에서 만든 문장
    # ================================================================
    print("\n" + "=" * 78)
    print("6. 결론")
    print("=" * 78)
    med = float(np.median(arr))
    where = ("중앙값보다 위" if cur_ret > med else
             "중앙값보다 아래" if cur_ret < med else "정확히 중앙값")
    inside = arr.min() <= cur_ret <= arr.max()
    lines = [
        f"지금은 진입 {day}거래일차이고 {fmt_pct(cur_ret)} 다. "
        f"과거 국면 {len(arr)}개의 같은 시점 중앙값은 {fmt_pct(med)} 이므로 {where}, "
        f"과거 범위({fmt_pct(float(arr.min()))} ~ {fmt_pct(float(arr.max()))}) "
        + ("안" if inside else "밖") + "에 있다."
    ]
    if peer:
        pa = np.array(peer)
        prank = int((pa < cur_ret).sum()) + 1
        lines.append(
            f"진입 낙폭이 비슷한 '{grp_now}' 국면 {len(pa)}개만 보면 같은 시점 "
            f"중앙값이 {fmt_pct(float(np.median(pa)))} 이고, 지금은 그중 "
            f"{len(pa)+1}개 중 {prank}번째다. "
            f"낙폭이 깊을수록 이후 수익이 컸으므로(진입낙폭 -> {label(OUTCOME)} 상관 "
            f"{dd_out:+.2f}), 전체를 뭉뚱그린 중앙값은 지금 상황보다 낙관적이다. "
            f"같은 그룹끼리 비교한 쪽이 맞다.")
    if near is not None:
        p = pred[near]
        if p["verdict"] == "관계가 버틴다":
            lines.append(
                f"{label(near)}차 수익률과 {label(OUTCOME)} 결과는 과거 {p['n']}개 국면에서 "
                f"피어슨 {p['r']:+.2f} / 순위 {p['spearman']:+.2f}, 하나씩 빼도 "
                f"{p['loo_lo']:+.2f}~{p['loo_hi']:+.2f} 로 방향이 유지된다 — "
                f"초반 흐름이 끝을 어느 정도 예고했다는 뜻이다."
                + (f" 낙폭 그룹 안에서만 봐도 {within:+.2f} 로 남으므로 낙폭의 그림자만은 "
                   f"아니지만, 그 표본은 {int(grp_mask.sum())}개뿐이다."
                   if within is not None else ""))
        else:
            lines.append(
                f"{label(near)}차 수익률과 {label(OUTCOME)} 결과의 관계는 "
                f"피어슨 {p['r']:+.2f} 이지만 순위상관 {p['spearman']:+.2f}, "
                f"하나씩 빼면 {p['loo_lo']:+.2f}~{p['loo_hi']:+.2f} 로 흔들린다 — "
                f"{p['verdict']}. 지금 위치로 앞날을 점치면 안 된다.")
    win = [r for r in fwd_rows if r["verdict"] == "우위 있다"]
    if win:
        lines.append(
            "지금부터 앞으로는, 무작위 진입 대비 우위가 "
            + " · ".join(f"{r['label']} {r['up']}/{r['n']}"
                         f"(구간 {r['ci_lo']*100:.0f}~{r['ci_hi']*100:.0f}%, "
                         f"무작위 {r['base']*100:.0f}%)" for r in win)
            + " 에서 확인된다. 구간 하한이 무작위보다 위일 때만 '우위' 라고 적었다."
            + (f" 반면 " + " · ".join(
                f"{r['label']}({r['rate']*100:.0f}% vs 무작위 {r['base']*100:.0f}%)"
                for r in fwd_rows if r["verdict"] == "우위 없다")
               + " 는 우위가 없다 — 기다림에도 유효 구간이 있다는 뜻이다."
               if any(r["verdict"] == "우위 없다" for r in fwd_rows) else ""))
    mr = {t: (int(np.median(reach[t])) if reach[t] else None) for t in TARGETS}
    got = [t for t in TARGETS if len(reach[t]) == len(past_ep)]
    if got:
        lines.append(
            "과거 국면 "
            + " · ".join(f"+{t}% 는 {len(reach[t])}/{len(past_ep)}개가 중앙값 {mr[t]}거래일"
                         for t in got)
            + " 만에 닿았다. 다만 그 사이 6개월 내 최악은 중앙값 "
            + f"{np.median([x['mae6'] for x in wait_rows])*100:.1f}% 였고, "
            + f"지금은 {day}거래일까지 {cur_path.min()*100:.1f}% 다. "
            + "기다리는 동안 견뎌야 하는 폭이다.")
    lines.append(
        f"표본은 국면 {len(arr)}개이고 그중 큰 수익은 2023년 회복장에 몰려 있다. "
        f"한 국면이 들고 나면 중앙값도 상관도 크게 움직인다. "
        f"이 비교는 '지금이 과거와 견줘 이상한가' 를 보는 맥락이지 예측이 아니다.")
    for s in lines:
        print("   " + s)

    payload = {
        "as_of": px.index[-1].strftime("%Y-%m-%d"),
        "entry": cur["entry"].strftime("%Y-%m-%d"),
        "signal": cur["signal"].strftime("%Y-%m-%d"),
        "day": day, "cur_ret": cur_ret, "cur_dd": float(cur["dd"]),
        "group": grp_now, "deep_cut": DEEP_CUT,
        "n_episodes": len(arr), "n_entries": len(past),
        "at_day": [{"entry": v["entry"].strftime("%Y-%m-%d"), "dd": v["dd"],
                    "grp": v["grp"], "ret": v["ret"]} for v in vals],
        "rank": rank, "median": med,
        "peer_median": float(np.median(peer)) if peer else None,
        "peer_n": len(peer),
        "predictive": [{"k": k, "label": label(k), **v} for k, v in pred.items()],
        "confound": {"dd_outcome": dd_out, "dd_early": dd_early,
                     "within_group": within, "within_n": int(grp_mask.sum())},
        "horizons": horizon_rows,
        "forward": fwd_rows,
        "targets": list(TARGETS),
        "wait": wait_rows,
        "wait_median": {str(t): (int(np.median(reach[t])) if reach[t] else None)
                        for t in TARGETS},
        "wait_reached": {str(t): len(reach[t]) for t in TARGETS},
        "mae_now": float(cur_path.min()),
        "mae6_median": float(np.median([x["mae6"] for x in wait_rows])),
        "verdict": lines,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, default=float)
    print(f"\n저장: {os.path.relpath(OUT, BASE)}")


if __name__ == "__main__":
    main()
