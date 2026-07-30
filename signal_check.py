"""주간 신호 판정 — 수집부터 판정까지 한 번에. GitHub Actions 에서 이 파일만 돌린다.

판정 주기가 왜 '주간'인가 (일간을 시험했고 실패했다):
  일간 판정은 검증을 통과하지 못했다. 최선의 정제(미래를 쓰는 룩어헤드 정제)를
  적용해도 6개월 초과수익 -1.9%, p=0.53 이다. 필터 문제가 아니라 일간 단위 자체가
  소스 노이즈에 묻힌다. 주간(금요일 판정)만 사건 9개에서 유의했다.

정제가 왜 '인과적'이어야 하는가:
  백테스트에 쓴 clean_holdings.py 의 왕복 검출은 '이후 구간의 중앙값'을 본다.
  실전에서는 그날 미래를 모른다. 검증한 것과 배포한 것이 다른 물건이 되면 안 되므로
  여기서는 당일까지의 정보만 쓰는 필터로 통일했다.

필터 파라미터에 대한 정직한 경고:
  AUM 급변 임계를 8%/10%/12% 로 바꾸면 같은 규칙의 6개월 초과수익이
  +45.6% / +14.7% / +20.2% (p=0.007 / 0.164 / 0.101) 로 흔들린다.
  전부 양수지만 유의성은 파라미터에 좌우된다.
  성적이 제일 좋은 값을 고르면 그게 과적합이므로, 사전 근거로 골랐다:
  대형 ETF 의 AUM 이 하루에 10% 넘게 움직이면서 그 종목 주가는 6% 미만이라면
  시장 요인으로 설명되지 않는다. 성적과 무관하게 이 값을 고정한다.
"""

import glob
import json
import os
import sys
import time
import urllib.request
from calendar import monthrange
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
# 아크가 테슬라를 담은 ETF 전부. ARKK+ARKQ 만 보면 전체 보유의 83% 만 보게 된다.
#   ARKK/ARKQ/ARKW 는 전 기간 상시, ARKX 는 2026-03 부터, CTRU 는 2021-12~2022-03.
# 중간에 나타나거나 사라지는 펀드가 있으므로 '레벨을 합쳐서 차분'하면 안 된다.
# 신규 편입일에 그 펀드의 보유량 전체가 대량 매수로 잡힌다.
# -> 펀드별로 먼저 차분한 뒤 합산한다. 신규 편입/청산은 자동으로 0 이 된다.
FUNDS = ("ARKK", "ARKQ", "ARKW", "ARKX", "CTRU")

AUM_JUMP = 0.10      # AUM 이 이만큼 튀고
PX_CALM = 0.06       # 주가는 이만큼도 안 움직였으면 -> 소스 오류
RE_BAND = 0.18       # 원래 수준의 이 범위로 돌아오면 정상 복귀
MIN_HIST = 52        # 문턱 계산에 필요한 최소 주 수
QUANT = 0.90
# 기술적 필터: 전고점(252일) 대비 이 수준 이하로 빠졌을 때만 매수한다.
# 지표를 '신호'로 쓰면 무효였지만 아크 신호를 '거르는' 데는 효과가 있었다.
# -20% ~ -45% 전 구간이 무필터보다 나았다(고원). 최악이 -28.3% -> -3.5% 로 개선된다.
# 현재 손실 중인 두 포지션(2026-06-01 낙폭 -15%, 2026-07-13 -19%)이 정확히 걸러진다.
DD_FILTER = -30.0

ARK_API = "https://arkfunds.io/api/v2/stock/fund-ownership"
# 전체 이력을 받는다. 점수 구성요소(200일선, 확장창 백분위)는 긴 이력이 있어야
# 값이 안정된다. 2021년부터만 받으면 초기 구간의 백분위가 표본 부족으로 요동친다.
YF = ("https://query1.finance.yahoo.com/v8/finance/chart/TSLA"
      "?period1=0&period2=9999999999&interval=1d&events=split")
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, tries: int = 4) -> bytes:
    delay = 3.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:                                   # noqa: BLE001
            print(f"    재시도 {k+1}/{tries} ({getattr(e,'code',None) or type(e).__name__})",
                  file=sys.stderr)
            if k == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def refresh_ark() -> None:
    """이번 달과 지난 달을 다시 받는다. arkfunds.io 는 월말일에만 500 을 내므로
    각 달을 [1일 ~ 말일-1] 로 잘라 요청한다."""
    os.makedirs(RAW, exist_ok=True)
    today = date.today()
    months = {(today.year, today.month)}
    prev = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    months.add(prev)
    for y, m in sorted(months):
        b = min(date(y, m, monthrange(y, m)[1] - 1), today)
        a = date(y, m, 1)
        if a > b:
            continue
        url = f"{ARK_API}?symbol=TSLA&date_from={a}&date_to={b}&limit=100"
        payload = json.loads(_get(url).decode())
        with open(os.path.join(RAW, f"m_{y}-{m:02d}.json"), "w") as f:
            json.dump(payload, f)
        print(f"  ARK {y}-{m:02d}: {len(payload.get('data', []))}일")
        time.sleep(1.2)


def refresh_prices() -> pd.Series:
    raw = _get(YF)
    with open(os.path.join(BASE, "data", "tsla_full.json"), "wb") as f:
        f.write(raw)
    r = json.loads(raw)["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    print(f"  주가: {len(px)}일 ({px.index[0]:%Y-%m}~), 최종 {px.index[-1]:%Y-%m-%d} ${px.iloc[-1]:.2f}")
    return px


def score_state(px: pd.Series, sh: pd.Series) -> dict:
    """종합 점수를 계산한다 — 표시용이며 판정에는 쓰지 않는다.

    검증 결과 점수와 이후 수익률은 단조 관계가 아니었고(README 참조),
    아크를 빼면 정보가 사라졌다(70점 이상 6개월 초과: 포함 +8.1% / 제외 -2.8%).
    기술적 지표는 신호를 강화하는 것이 아니라 희석한다.
    그래서 화면에는 '지금 무엇이 켜져 있는가'를 보여주는 용도로만 싣는다.
    """
    from score_model import components          # 순환 임포트 방지: 함수 안에서 부른다
    ark_w = sh.resample("W-FRI").last().dropna().diff()
    c = components(px, ark_w)
    cols = ["ark", "rsi", "dd", "macd", "ma", "bb"]
    c = c.dropna(subset=cols)
    if c.empty:
        return {}
    cur = c.iloc[-1]
    return {
        "week": c.index[-1].strftime("%Y-%m-%d"),
        "total": round(float(cur[cols].mean()) * 100, 1),
        "components": {k: round(float(cur[k]) * 100, 1) for k in cols},
        "history": [{"d": d.strftime("%Y-%m-%d"),
                     "s": round(float(r[cols].mean()) * 100, 1)}
                    for d, r in c.tail(60).iterrows()],
    }


def load_raw() -> pd.DataFrame:
    rows = {}
    for p in glob.glob(os.path.join(RAW, "*.json")):
        for day in json.load(open(p)).get("data", []):
            for o in day.get("ownership", []):
                if o["fund"] in FUNDS:
                    rows[(o["date"], o["fund"])] = o          # (날짜,펀드) 중복 제거
    d = pd.DataFrame(rows.values())
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["fund", "date"]).reset_index(drop=True)


def causal_clean(g: pd.DataFrame) -> pd.DataFrame:
    """AUM 이 주가로 설명되지 않는 크기로 튄 구간을 버린다. 당일까지의 정보만 쓴다."""
    g = g.sort_values("date").reset_index(drop=True)
    a, p = g["aum"].values, g["close"].values
    keep = np.ones(len(g), bool)
    anchor = a[0]
    for i in range(1, len(g)):
        moved = abs(np.log(a[i] / anchor))
        px_move = abs(np.log(p[i] / p[i - 1])) if p[i - 1] > 0 else 0.0
        if moved > AUM_JUMP and px_move < PX_CALM:
            keep[i] = False
        elif moved < RE_BAND or keep[i]:
            anchor = a[i]
    return g[keep]


def tech_state(px: pd.Series) -> dict:
    """신호 시점의 기술적 지표 상태를 '기록만' 한다. 판정에는 쓰지 않는다.

    지표를 합류시키면 신호가 더 확실해진다는 가설은 검증에 실패했다(README 참조).
    지표 단독은 전부 무효였고, 합류 강도와 수익률의 상관은 판정 창을 바꾸면
    +0.27~+0.55 로 요동쳤다. n=7 에서는 노이즈다.
    근거가 n=3~4 인 규칙을 판정에 넣는 것은 과적합을 코드에 새기는 일이므로,
    지금은 계측만 해두고 사건이 쌓이면 그때 답한다.
    """
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    m20, s20 = px.rolling(20).mean(), px.rolling(20).std()
    last = -1
    f = lambda v: None if pd.isna(v) else round(float(v), 2)
    return {
        "rsi": f(rsi.iloc[last]),
        "macd": f(macd.iloc[last]), "macd_signal": f(macd_sig.iloc[last]),
        "ma50": f(ma50.iloc[last]), "ma200": f(ma200.iloc[last]),
        "bb_low": f((m20 - 2 * s20).iloc[last]),
        "flags": {
            "rsi_oversold": bool(rsi.iloc[last] < 30),
            "macd_above_signal": bool(macd.iloc[last] > macd_sig.iloc[last]),
            "below_ma200": bool(px.iloc[last] < ma200.iloc[last]) if not pd.isna(ma200.iloc[last]) else None,
            "below_ma50": bool(px.iloc[last] < ma50.iloc[last]) if not pd.isna(ma50.iloc[last]) else None,
            "below_bb": bool(px.iloc[last] < (m20 - 2 * s20).iloc[last]),
        },
    }


def build_daily(px: pd.Series) -> pd.DataFrame:
    """정제까지 끝난 일별 합산 보유량. 판정은 주간이지만 표시는 일별로 한다."""
    d = load_raw()
    d["close"] = d["date"].map(px)
    d = d.dropna(subset=["close"])
    # 분할 보정: 날짜가 아니라 내재 주가로 판정 (ARK 는 분할 당일에도 분할 전 수치를 보고했다)
    ratio = (d["market_value"] / d["shares"]) / d["close"]
    d["shares_adj"] = np.where(ratio > 2, d["shares"] * 3, d["shares"])
    d["aum"] = d["market_value"] / (d["weight"] / 100)
    cleaned = pd.concat([causal_clean(g) for _, g in d.groupby("fund")])
    wide = cleaned.pivot(index="date", columns="fund", values="shares_adj")
    cols = [f for f in FUNDS if f in wide.columns]
    # min_count=1 : 그날 존재하는 펀드만 더한다(없는 펀드를 0 으로 치지 않는다)
    out = pd.DataFrame({"shares": wide[cols].sum(axis=1, min_count=1)}).dropna()
    out["close"] = px.reindex(out.index).ffill()
    out.attrs["wide"] = wide[cols]
    return out


def build(px: pd.Series) -> pd.DataFrame:
    """주간 판정용. 일간은 검증에 실패했으므로(README 참조) 금요일 기준으로 묶는다.

    문턱은 '절대 주식 수'가 아니라 '보유 대비 %'로 잡는다.
    아크의 규모가 시대마다 다르다. 2021년 ARKK 는 AUM $16.5B 에 테슬라 12M주였고
    지금은 $5.6B 에 2.3M주다. 절대 주식 수로 문턱을 잡으면 2021년의 대량 매수가
    확장창 분위수를 영구히 끌어올려, 이후 몇 년간 신호가 거의 뜨지 않는다
    (실측: 절대 기준 사건 3개 vs 비율 기준 8개).
    단위를 바로잡는 것이지 성적에 맞추는 것이 아니다.
    """
    daily = build_daily(px)
    wide = daily.attrs["wide"]
    wk = wide.resample("W-FRI").last()
    # 펀드별로 차분한 뒤 합산 — 신규 편입/청산이 매매로 잡히지 않는다
    net = wk.diff().sum(axis=1, min_count=1)
    base = wk.shift(1).sum(axis=1, min_count=1)
    w = pd.DataFrame({"shares": wk.sum(axis=1, min_count=1), "net": net}).dropna(subset=["net"])
    w["netpct"] = (net / base * 100).reindex(w.index)
    w["thr_pct"] = w["netpct"].expanding(MIN_HIST).quantile(QUANT).shift(1)  # 과거만 사용
    # 전고점 대비 낙폭 (주가만으로 계산되므로 룩어헤드가 없다)
    dd = (px / px.rolling(252, min_periods=60).max() - 1) * 100
    w["dd"] = dd.reindex(w.index, method="ffill")
    w["thr_ok"] = w["netpct"] >= w["thr_pct"]
    w["dd_ok"] = w["dd"] <= DD_FILTER
    w["sig"] = w["thr_ok"] & w["dd_ok"]
    # 화면 표시는 주식 수가 직관적이므로 문턱을 주식 수로 환산해 함께 싣는다.
    w["thr"] = w["thr_pct"] / 100 * w["shares"].shift(1)
    return w.dropna(subset=["thr_pct"])


def _chart_series(px: pd.Series) -> dict:
    """대시보드 차트용 시계열. 파일 크기를 줄이려 3거래일 간격으로 솎는다."""
    lo = pd.Timestamp("2021-06-01")
    p = px[px.index >= lo]
    dd = (px / px.rolling(252, min_periods=60).max() - 1) * 100
    d = dd.reindex(p.index)
    step = 3
    return {"t": [x.strftime("%Y-%m-%d") for x in p.index[::step]],
            "p": [round(float(v), 2) for v in p.values[::step]],
            "dd": [None if pd.isna(v) else round(float(v), 1) for v in d.values[::step]]}


def _rsi14(px: pd.Series) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _signal_runs(w: pd.DataFrame, px: pd.Series) -> list:
    """연속(7일 이내) 신호를 한 묶음으로 정리한다.

    각 신호의 매수일·청산일·수익까지 함께 담는다. 화면에서 국면 단위로
    보여주려면 신호일(금)과 매수일(다음 거래일)이 둘 다 필요하다 —
    하나만 보여주면 간격이 7일이 아닌 것처럼 보여 오해가 생긴다."""
    sg = list(w[w["sig"]].index)
    if not sg:
        return []
    runs, cur = [], [sg[0]]
    for t in sg[1:]:
        if (t - cur[-1]).days <= 7:
            cur.append(t)
        else:
            runs.append(cur)
            cur = [t]
    runs.append(cur)
    V = px.values
    # position_tracker 와 같은 규칙: 아크 -20% AND 낙폭 -20% 회복
    from position_tracker import ARK_DROP, DD_RECOVER, MAX_HOLD
    H = build_daily(px)["shares"].reindex(px.index).ffill().values
    DDv = ((px / px.rolling(252, min_periods=60).max() - 1) * 100).values

    def exit_of(e):
        for j in range(e + 1, min(e + MAX_HOLD + 1, len(V))):
            if H[j] <= H[e] * (1 - ARK_DROP) and DDv[j] >= DD_RECOVER:
                return j, "아크 -20% + 낙폭 회복"
        return (e + MAX_HOLD, "상한 도달") if e + MAX_HOLD < len(V) else (None, None)

    # 신호 시점의 종합 점수도 함께 싣는다 (수익률과 r=+0.54 로 상관이 있다)
    try:
        from score_model import components
        _sh = build_daily(px)["shares"]
        _c = components(px, _sh.resample("W-FRI").last().dropna().diff())
        _cols = ["ark", "rsi", "dd", "macd", "ma", "bb"]
        _score = _c[_cols].mean(axis=1) * 100
    except Exception:
        _score = None

    out = []
    for rn in runs:
        entries, rets = [], []
        for t in rn:
            k = px.index.searchsorted(t, side="right")
            if k >= len(px):
                continue
            x, why = exit_of(k)
            ret = (V[x] if x is not None else V[-1]) / V[k] - 1
            rets.append(float(ret))
            entries.append({
                "signal": t.strftime("%Y-%m-%d"),
                "date": px.index[k].strftime("%Y-%m-%d"),
                "price": round(float(V[k]), 2),
                "exit": None if x is None else px.index[x].strftime("%Y-%m-%d"),
                "exit_price": None if x is None else round(float(V[x]), 2),
                "days": int((x if x is not None else len(V) - 1) - k),
                "ret": round(float(ret), 6),
                "score": (None if _score is None or t not in _score.index
                          or pd.isna(_score[t]) else round(float(_score[t]), 0)),
                "why": why, "open": x is None})
        if not entries:
            continue
        out.append({"start": rn[0].strftime("%Y-%m-%d"),
                    "end": rn[-1].strftime("%Y-%m-%d"),
                    "weeks": len(rn), "entries": entries,
                    "avg_price": round(float(np.mean([e["price"] for e in entries])), 2),
                    "run_ret": round(float(np.mean(rets)), 6),
                    "open": any(e["open"] for e in entries)})
    return out


def main() -> None:
    started = datetime.now(timezone.utc)
    print(f"실행 시각(UTC) {started:%Y-%m-%d %H:%M} — Actions 스케줄은 수십 분 밀릴 수 있다")
    refresh_ark()
    px = refresh_prices()
    w = build(px)

    cur = w.iloc[-1]
    on = bool(cur["sig"])
    prev = w[w["sig"]]
    past_sig = prev.index[prev.index < w.index[-1]]

    # 대시보드가 '오늘'을 보여줄 수 있도록 일별 최근값도 같이 싣는다.
    # 판정은 주간이지만, 표시까지 주간으로 묶으면 화면이 최대 6일 묵는다.
    daily = build_daily(px)
    recent = daily.tail(30)

    state = {
        "checked_utc": started.isoformat(timespec="seconds"),
        "week": w.index[-1].strftime("%Y-%m-%d"),
        "signal": on,
        "net": int(cur["net"]),
        "threshold": int(cur["thr"]),
        "gap": int(cur["net"] - cur["thr"]),
        "net_pct": round(float(cur["netpct"]), 2),
        "threshold_pct": round(float(cur["thr_pct"]), 2),
        "dd_now": round(float(cur["dd"]), 1),
        "dd_filter": DD_FILTER,
        "thr_ok": bool(cur["thr_ok"]),
        "dd_ok": bool(cur["dd_ok"]),
        "shares": int(cur["shares"]),
        "price": round(float(px.iloc[-1]), 2),
        "price_date": px.index[-1].strftime("%Y-%m-%d"),
        "drawdown_from_ath": round(float(px.iloc[-1] / px.max() - 1) * 100, 1),
        "ath": round(float(px.max()), 2),
        "ath_date": px.idxmax().strftime("%Y-%m-%d"),
        # 현재 판정 주보다 '앞선' 신호 중 가장 최근 것.
        # [-2] 를 쓰면 현재 주가 OFF 일 때 가장 최근 신호를 건너뛴다(실제로 그랬다).
        "prev_signal_week": (past_sig[-1].strftime("%Y-%m-%d") if len(past_sig) else None),
        "prev_signal_weeks_ago": (int(len(w.loc[past_sig[-1]:]) - 1) if len(past_sig) else None),
        # 최근 신호의 '진입일/진입가' 까지 싣는다. 신호 확정일과 매수일이 다르므로
        # (공시가 장 마감 후라 다음 거래일 종가 진입) 둘 다 보여줘야 오해가 없다.
        "prev_signal_entry": (px.index[px.index.searchsorted(past_sig[-1], side="right")]
                              .strftime("%Y-%m-%d")
                              if len(past_sig) and
                              px.index.searchsorted(past_sig[-1], side="right") < len(px)
                              else None),
        "prev_signal_price": (round(float(px.iloc[px.index.searchsorted(past_sig[-1], side="right")]), 2)
                              if len(past_sig) and
                              px.index.searchsorted(past_sig[-1], side="right") < len(px)
                              else None),
        "signal_weeks_total": int(w["sig"].sum()),
        # 신호 묶음: 7일 이내로 이어진 신호는 한 국면으로 본다.
        # "연속으로 뜨는 일이 흔한가" 를 화면에서 바로 볼 수 있게 싣는다.
        "runs": _signal_runs(w, px),
        # 차트용 주가·낙폭 시계열 (신호가 시작된 2021-06 이후, 3일 간격으로 솎음)
        "chart": _chart_series(px),
        "holdings_date": daily.index[-1].strftime("%Y-%m-%d"),
        "holdings": int(daily["shares"].iloc[-1]),
        "holdings_chg": int(daily["shares"].diff().iloc[-1]),
        "recent": [{"d": d.strftime("%Y-%m-%d"), "s": int(r["shares"]),
                    "c": (None if pd.isna(r["close"]) else round(float(r["close"]), 2))}
                   for d, r in recent.iterrows()],
        # 판정에는 쓰지 않는다. 사건이 쌓였을 때 답하기 위한 기록이다.
        "tech": tech_state(px),
        # 이번 주 점수. 이번 주가 신호가 아니면 참고값일 뿐이며
        # 수익률과 연결되지 않는다 — 화면·알림에서 그 점을 명시한다.
        "score": score_state(px, daily["shares"]),
        "weekly": [{"d": d.strftime("%Y-%m-%d"), "net": int(r["net"]),
                    "thr": int(r["thr"]), "sig": bool(r["sig"])}
                   for d, r in w.tail(16).iterrows()],
    }
    path = os.path.join(BASE, "data", "signal_state.json")
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)

    print("\n" + "=" * 60)
    print(f"주 {state['week']}  순매수 {state['net']:+,}주 / 문턱 {state['threshold']:+,}주")
    print(f"차이 {state['gap']:+,}주  ->  문턱 {'통과' if state['thr_ok'] else '미달'}")
    print(f"낙폭 {state['dd_now']:.1f}% (기준 {DD_FILTER}%) -> "
          f"{'통과' if state['dd_ok'] else '미달'}")
    print(f"==> {'*** 매수 신호 ON ***' if on else '신호 OFF'}")
    print(f"TSLA ${state['price']} ({state['price_date']}), 사상최고 대비 {state['drawdown_from_ath']}%")
    print("=" * 60)

    # GitHub Actions 로 결과 전달
    if out := os.environ.get("GITHUB_OUTPUT"):
        with open(out, "a") as f:
            f.write(f"signal={'true' if on else 'false'}\n")
            f.write(f"week={state['week']}\n")
            f.write(f"net={state['net']}\n")
            f.write(f"threshold={state['threshold']}\n")
            f.write(f"price={state['price']}\n")
            f.write(f"drawdown={state['drawdown_from_ath']}\n")
            f.write(f"dd_now={state['dd_now']}\ndd_filter={DD_FILTER}\n")
            f.write(f"thr_ok={'true' if state['thr_ok'] else 'false'}\n")
            f.write(f"dd_ok={'true' if state['dd_ok'] else 'false'}\n")
            # 주간 상태 알림(하트비트)에 필요한 값들
            f.write(f"net_pct={state['net_pct']}\n")
            f.write(f"thr_pct={state['threshold_pct']}\n")
            f.write(f"progress={round(state['net_pct'] / state['threshold_pct'] * 100) if state['threshold_pct'] else 0}\n")
            f.write(f"prev_week={state['prev_signal_week'] or '-'}\n")
            f.write(f"prev_ago={state['prev_signal_weeks_ago'] if state['prev_signal_weeks_ago'] is not None else '-'}\n")
            f.write(f"holdings_date={state['holdings_date']}\n")
            f.write(f"holdings={state['holdings']}\n")
            f.write(f"score={state.get('score', {}).get('total', '-')}\n")
            # 직전 '신호' 의 점수. 이번 주 점수와 구분해야 한다 —
            # 신호가 아닌 주의 점수는 수익률과 연결되지 않는다.
            _pv = None
            for _r in reversed(state.get("runs", [])):
                for _e in reversed(_r.get("entries", [])):
                    if _e.get("score") is not None:
                        _pv = _e["score"]
                        break
                if _pv is not None:
                    break
            f.write(f"prev_score={_pv if _pv is not None else '-'}\n")
            f.write(f"ark_score={state.get('score', {}).get('components', {}).get('ark', '-')}\n")
            # 신호 확률은 signal_probability.py 스텝이 직접 내보낸다.
            # 여기서 읽으면 실행 순서상 하루 전 값이 나간다.


if __name__ == "__main__":
    main()
