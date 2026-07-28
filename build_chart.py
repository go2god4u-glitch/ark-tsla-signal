"""수집한 원본 JSON + 야후 주가로 TSLA / ARK 차트를 만든다.

핵심 주의사항 3가지:

1. 중복 제거. data/raw/ 에는 두 세대의 파일이 섞여 있다.
   - `YYYY-MM-DD_YYYY-MM-DD.json` : 1차 수집(30일 청크), 2021-07 ~ 2023-02
   - `m_YYYY-MM.json`             : 2차 수집(월 단위), 2021-07 ~ 현재
   기간이 통째로 겹치므로 그냥 이어붙이면 앞 19개월의 보유수가 정확히 2배가 된다.
   그 모양이 "ARK 가 2023년에 절반을 팔았다"처럼 보여서 눈으로는 절대 못 잡는다.
   그래서 (날짜, 펀드) 를 키로 dict 에 넣어 한 행만 남긴다.

2. `totals` 금지. 응답의 totals 는 ARKW/ARKX 까지 더한 값이다.
   ARKW 의 TSLA 포지션은 작지 않아서 섞이면 그래프가 통째로 틀어진다.
   ARKK/ARKQ 만 명시적으로 골라 합산한다.

3. 액면분할. 2022-08-25 에 3:1 분할이 있었다. ARK 는 '그날 실제 주식 수'를
   보고하므로 분할 전 숫자는 분할 후 기준으로 ×3 해야 연속적인 선이 된다.
   야후 종가는 이미 분할 반영(split-adjusted)된 값이다.
   단, 날짜로 자르면 안 된다. ARK 는 분할 당일(2022-08-25)에도 분할 전
   주식 수를 그대로 보고했다. 날짜 기준으로 자르면 그 하루가 1/3 로 꺼진다.
   -> 행마다 market_value/shares(내재 주가)를 야후 종가와 비교해
      어느 기준으로 보고된 값인지 직접 판정한다. 추측하지 않는다.

4. 소스 데이터 이상치. arkfunds.io 는 가끔 비중은 정상(~10%)인데 주식 수만
   하루 이틀 튀는 행을 준다(예: 2026-06-15 ARKK 3.29M, 앞뒤는 1.7~2.1M).
   market_value/weight = 펀드 전체 AUM 인데, 이 값이 하루 만에 20% 넘게
   뛰었다 돌아온다는 건 실제로는 불가능하다(그만한 자금유입이 없다).
   그래서 AUM 기준으로 이상치를 걸러낸다. 보간하지 않고 그냥 버린다.
"""

import glob
import json
import os
from datetime import date, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
FUNDS = ("ARKK", "ARKQ")
SPLIT_DATE = date(2022, 8, 25)   # 3:1
SPLIT_RATIO = 3

# dataviz 레퍼런스 팔레트(라이트 모드). 파랑=ARKK, 주황=ARKQ 로 전 패널 고정.
C_ARKK = "#2a78d6"
C_ARKQ = "#eb6834"
C_TOTAL = "#4a3aa7"
C_PRICE = "#3f3e3b"
C_GRID = "#dcdbd6"
C_TEXT = "#0b0b0b"
C_MUTED = "#7a7973"


def load_ark() -> pd.DataFrame:
    """raw JSON 전부를 읽어 (date, fund) 중복 없는 long 포맷으로 만든다."""
    rows: dict[tuple[str, str], dict] = {}
    files = sorted(glob.glob(os.path.join(RAW, "*.json")))
    for path in files:
        with open(path) as f:
            payload = json.load(f)
        for day in payload.get("data", []):
            for o in day.get("ownership", []):     # totals 는 읽지 않는다
                if o["fund"] not in FUNDS:
                    continue
                rows[(o["date"], o["fund"])] = o   # 같은 키면 덮어쓴다 = 중복 제거

    df = pd.DataFrame(rows.values())
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "fund"]).reset_index(drop=True)

    per_date = df.groupby("date").size()
    assert per_date.max() <= len(FUNDS), f"날짜당 행이 {per_date.max()}개 - 중복 제거 실패"
    print(f"[ARK] 파일 {len(files)}개 -> {df['date'].nunique()}일 "
          f"({df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d})")
    return df


def load_price() -> pd.DataFrame:
    with open(os.path.join(BASE, "data", "tsla_yahoo.json")) as f:
        r = json.load(f)["chart"]["result"][0]
    px = pd.DataFrame({
        "date": [datetime.fromtimestamp(t).date() for t in r["timestamp"]],
        "close": r["indicators"]["quote"][0]["close"],
    }).dropna()
    px["date"] = pd.to_datetime(px["date"])
    print(f"[주가] {len(px)}일 ({px['date'].min():%Y-%m-%d} ~ {px['date'].max():%Y-%m-%d}), "
          f"분할반영 종가")
    return px


def adjust_split(ark: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """행마다 분할 기준을 판정해 주식 수를 현재 기준으로 환산한다.

    ARK 가 보고한 market_value / shares = 그날의 '실제 주가'다.
    이 값이 야후의 분할반영 종가의 3배면 분할 전 기준으로 보고된 행이고,
    1배면 이미 분할 후 기준이다. 날짜가 아니라 이 비율로 판정하므로
    분할 당일처럼 소스가 하루 늦게 반영한 경우도 정확히 잡힌다.
    """
    m = ark.merge(px[["date", "close"]], on="date", how="left")
    ratio = (m["market_value"] / m["shares"]) / m["close"]

    pre_like = ratio > 2          # 3배 근처
    post_like = ratio < 1.5       # 1배 근처
    bad = ~(pre_like | post_like) & ratio.notna()
    assert not bad.any(), f"분할 기준 판정 불가 행 {bad.sum()}개: {m[bad]['date'].tolist()[:5]}"

    m["shares_adj"] = m["shares"].where(~pre_like, m["shares"] * SPLIT_RATIO)
    n_pre = int(pre_like.sum())
    print(f"[분할보정] ×{SPLIT_RATIO} 적용 {n_pre}행 / 그대로 {int(post_like.sum())}행 "
          f"(마지막 보정일 {m[pre_like]['date'].max():%Y-%m-%d}, 분할일 {SPLIT_DATE})")
    return m


def drop_bad_rows(m: pd.DataFrame) -> pd.DataFrame:
    """소스의 주식 수 이상치를 버린다.

    market_value / (weight/100) = 그 펀드의 전체 AUM. 실제 AUM 은 하루에
    20% 넘게 움직일 수 없다(주가가 아니라 자금 규모다). 5일 중앙값 대비
    그 이상 벗어난 행은 소스 오류로 보고 버린다 — 보간은 하지 않는다.
    """
    m = m.sort_values(["fund", "date"]).copy()
    m["aum"] = m["market_value"] / (m["weight"] / 100)
    med = m.groupby("fund")["aum"].transform(
        lambda s: s.rolling(5, center=True, min_periods=3).median())
    bad = (m["aum"] / med - 1).abs() > 0.20
    if bad.any():
        d = m[bad]
        print(f"[이상치] AUM 급변 {int(bad.sum())}행 제거: "
              f"{', '.join(f'{r.date:%Y-%m-%d}/{r.fund}' for r in d.itertuples())}")
    return m[~bad]


def main() -> None:
    ark = load_ark()
    px = load_price()

    m = drop_bad_rows(adjust_split(ark, px))

    wide_sh = m.pivot(index="date", columns="fund", values="shares_adj")
    wide_wt = m.pivot(index="date", columns="fund", values="weight")
    # 두 펀드가 다 있는 날만 쓴다. 한쪽만 있으면 합계가 그날만 푹 꺼진다.
    both = wide_sh[list(FUNDS)].dropna().index
    n_drop = len(wide_sh) - len(both)
    if n_drop:
        print(f"[결측] 한쪽 펀드만 있는 날 {n_drop}일 제외")
    wide_sh, wide_wt = wide_sh.loc[both], wide_wt.loc[both]
    combined = wide_sh[list(FUNDS)].sum(axis=1)

    out = pd.DataFrame({
        "ARKK_shares_adj": wide_sh["ARKK"], "ARKQ_shares_adj": wide_sh["ARKQ"],
        "combined_shares_adj": combined,
        "ARKK_weight_pct": wide_wt["ARKK"], "ARKQ_weight_pct": wide_wt["ARKQ"],
    }).join(px.set_index("date")["close"].rename("TSLA_close"))
    out.to_csv(os.path.join(BASE, "data", "tsla_ark_merged.csv"))

    plt.rcParams.update({
        "font.family": "AppleGothic",      # 한글 깨짐 방지 (macOS)
        "axes.unicode_minus": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "text.color": C_TEXT, "axes.labelcolor": C_MUTED,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    })

    # 축을 하나만 쓰기 위해 이중축 대신 3단 패널로 나눈다(x축 공유).
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 11.5), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1, 1], "hspace": 0.26})

    ax1.plot(combined.index, combined / 1e6, color=C_TOTAL, lw=2)
    ax1.fill_between(combined.index, combined / 1e6, color=C_TOTAL, alpha=0.07)
    ax1.set_ylabel("보유 주식 수 (백만 주)")
    ax1.set_title("ARKK + ARKQ 합산 테슬라 보유 주식 수  (2022-08 3:1 분할 반영)",
                  loc="left", fontsize=13, pad=10, color=C_TEXT)

    ax2.plot(px["date"], px["close"], color=C_PRICE, lw=2)
    ax2.set_ylabel("TSLA 종가 (USD)")
    ax2.set_title("테슬라 주가 (분할 반영 종가)", loc="left", fontsize=13,
                  pad=10, color=C_TEXT)

    ax3.plot(wide_wt.index, wide_wt["ARKK"], color=C_ARKK, lw=2, label="ARKK")
    ax3.plot(wide_wt.index, wide_wt["ARKQ"], color=C_ARKQ, lw=2, label="ARKQ")
    ax3.set_ylabel("펀드 내 테슬라 비중 (%)")
    ax3.set_title("펀드별 테슬라 비중", loc="left", fontsize=13, pad=10, color=C_TEXT)
    leg = ax3.legend(loc="upper right", frameon=False, ncol=2)
    for t, c in zip(leg.get_texts(), (C_ARKK, C_ARKQ)):
        t.set_color(C_TEXT)

    # 마지막 값 직접 라벨 — 범례에 의존하지 않고 각 선의 끝을 읽게 한다.
    # ARKK/ARKQ 는 현재 값이 거의 붙어 있어서, 큰 쪽을 위 작은 쪽을 아래로 민다.
    k_last, q_last = wide_wt["ARKK"].iloc[-1], wide_wt["ARKQ"].iloc[-1]
    dy_k, dy_q = (9, -9) if k_last >= q_last else (-9, 9)
    for ax, series, color, fmt, dy in (
            (ax1, combined / 1e6, C_TOTAL, "{:.2f}M", 0),
            (ax2, px.set_index("date")["close"], C_PRICE, "${:.0f}", 0),
            (ax3, wide_wt["ARKK"], C_ARKK, "ARKK {:.1f}%", dy_k),
            (ax3, wide_wt["ARKQ"], C_ARKQ, "ARKQ {:.1f}%", dy_q)):
        ax.annotate(fmt.format(series.iloc[-1]),
                    xy=(series.index[-1], series.iloc[-1]),
                    xytext=(8, dy), textcoords="offset points",
                    va="center", fontsize=10, color=color, fontweight="bold")

    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(C_GRID)
        ax.axvline(pd.Timestamp(SPLIT_DATE), color=C_MUTED, lw=1, ls=":", alpha=0.7)
        ax.margins(x=0.06)

    ax1.text(pd.Timestamp(SPLIT_DATE), ax1.get_ylim()[1] * 0.97, " 3:1 분할",
             fontsize=9, color=C_MUTED, va="top")

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("ARK 의 테슬라 보유 추이 vs 테슬라 주가 (최근 5년)",
                 x=0.075, ha="left", fontsize=17, fontweight="bold", y=0.977)
    fig.text(0.075, 0.947,
             f"{combined.index.min():%Y-%m-%d} ~ {combined.index.max():%Y-%m-%d}"
             "   ·   보유 데이터 arkfunds.io, 주가 Yahoo Finance"
             "   ·   각 달의 말일 하루는 데이터 소스 오류로 결측",
             ha="left", fontsize=10, color=C_MUTED)

    fig.subplots_adjust(top=0.885, left=0.075, right=0.93, bottom=0.05)
    png = os.path.join(BASE, "tsla_ark_holdings.png")
    fig.savefig(png, dpi=160)
    print(f"\n저장: {png}")
    print(f"저장: {os.path.join(BASE, 'data', 'tsla_ark_merged.csv')}")

    first, last = combined.iloc[0], combined.iloc[-1]
    print(f"\n합산 보유 주식 수: {first/1e6:.2f}M ({combined.index[0]:%Y-%m-%d})"
          f" -> {last/1e6:.2f}M ({combined.index[-1]:%Y-%m-%d})"
          f"  [{(last/first-1)*100:+.1f}%]  최고 {combined.max()/1e6:.2f}M"
          f" ({combined.idxmax():%Y-%m-%d})")


if __name__ == "__main__":
    main()
