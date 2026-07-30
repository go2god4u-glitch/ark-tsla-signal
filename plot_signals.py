"""테슬라 주가 위에 매수 신호 시점을 표시한다.

사용자 질문: 신호가 나온 주가가 시간에 따라 우상향하는가?
-> 신호 진입가에 추세선을 그어 눈으로 확인할 수 있게 한다.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

C_PX, C_SIG, C_EXIT, C_TREND = "#2b3138", "#0f7a52", "#c0392f", "#2a78d6"
C_GRID, C_INK, C_MUTED = "#dcdbd6", "#0f1419", "#7a7973"

def main():
    r = json.load(open("data/tsla_full.json"))["chart"]["result"][0]
    idx = pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize()
    px = pd.Series(r["indicators"]["quote"][0]["close"], index=idx).dropna().sort_index()
    st = json.load(open("data/signal_state.json"))

    sigs = [(pd.Timestamp(e["date"]), e["price"], e["exit"], e["exit_price"], e["open"])
            for run in st["runs"] for e in run["entries"]]
    sd = [s[0] for s in sigs]; sp = [s[1] for s in sigs]

    plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "text.color": C_INK, "axes.labelcolor": C_MUTED,
                         "xtick.color": C_MUTED, "ytick.color": C_MUTED})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    # 신호가 시작된 2022 이후만 (그 전은 알고리즘이 작동할 데이터가 없다)
    lo = pd.Timestamp("2021-06-01")
    p = px[px.index >= lo]
    ax.plot(p.index, p.values, color=C_PX, lw=1.2, zorder=2)
    ax.set_yscale("log")
    ax.set_ylabel("TSLA 종가 (USD, 로그축)")

    for d, pr, xd, xp, live in sigs:
        ax.scatter([d], [pr], s=90, color=C_SIG, zorder=5,
                   edgecolor="white", linewidth=1.4)
        if not live and xd:
            ax.scatter([pd.Timestamp(xd)], [xp], s=55, marker="v", color=C_EXIT,
                       zorder=5, edgecolor="white", linewidth=1)
            ax.plot([d, pd.Timestamp(xd)], [pr, xp], color=C_SIG,
                    lw=0.9, alpha=0.35, zorder=3)

    # 신호 진입가 추세선 (로그 공간에서 회귀 -> 연평균 변화율)
    x = mdates.date2num(sd)
    b, a = np.polyfit(x, np.log(sp), 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(mdates.num2date(xs), np.exp(a + b * xs), color=C_TREND, lw=2,
            ls="--", zorder=4, label=f"신호 진입가 추세  연 {(np.exp(b*365)-1)*100:+.0f}%")
    ax.scatter([], [], s=90, color=C_SIG, label=f"매수 신호 ({len(sigs)}회)")
    ax.scatter([], [], s=55, marker="v", color=C_EXIT, label="청산")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("테슬라 주가와 매수 신호 시점", loc="left", fontsize=15,
                 fontweight="bold", pad=28)

    # 아래: 신호 시점의 낙폭
    dd = (px / px.rolling(252, min_periods=60).max() - 1) * 100
    d2 = dd[dd.index >= lo]
    ax2.plot(d2.index, d2.values, color=C_MUTED, lw=1)
    ax2.axhline(-30, color=C_SIG, ls=":", lw=1.2)
    ax2.text(d2.index[-1], -30, "진입 문턱 -30% ", color=C_SIG, fontsize=9,
             va="bottom", ha="right")
    for d, pr, *_ in sigs:
        ax2.scatter([d], [dd[d]], s=45, color=C_SIG, zorder=5,
                    edgecolor="white", linewidth=1)
    ax2.set_ylabel("252일 고점 대비 낙폭 (%)")
    ax2.grid(axis="y", color=C_GRID, lw=0.7)
    ax2.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax2.spines[s_].set_visible(False)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    df_y = pd.DataFrame({"d": sd, "p": sp})
    for y, g in df_y.groupby(df_y["d"].dt.year):
        ax.annotate(f"{y}\n평균 ${g['p'].mean():.0f}",
                    xy=(g["d"].mean(), g["p"].mean()), xytext=(0, -34),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=C_SIG, alpha=0.85)

    first, last = sp[0], sp[-1]
    fig.text(0.077, 0.952,
             f"신호 {len(sigs)}회 · 진입가 ${first:.0f}({sd[0]:%Y-%m}) → ${last:.0f}({sd[-1]:%Y-%m})"
             f" · 추세 연 {(np.exp(b*365)-1)*100:+.0f}%"
             f"   ·   같은 기간 주가는 연 {((px.iloc[-1]/px[px.index>=sd[0]].iloc[0])**(365/(px.index[-1]-sd[0]).days)-1)*100:+.0f}%",
             fontsize=10.5, color=C_MUTED)
    fig.subplots_adjust(top=0.885, left=0.077, right=0.975, bottom=0.06)
    fig.savefig("signal_chart.png", dpi=160)
    print(f"저장: signal_chart.png")
    print(f"\n신호 진입가 추세: 연 {(np.exp(b*365)-1)*100:+.1f}%")
    print(f"  첫 신호 ${first:.2f} ({sd[0]:%Y-%m-%d}) -> 마지막 ${last:.2f} ({sd[-1]:%Y-%m-%d})")
    print(f"\n연도별 신호 진입가:")
    df = pd.DataFrame({"date": sd, "price": sp})
    for y, g in df.groupby(df["date"].dt.year):
        print(f"  {y}: {len(g):>2d}회  ${g['price'].min():>7.2f} ~ ${g['price'].max():>7.2f}"
              f"  (평균 ${g['price'].mean():>7.2f})")

if __name__ == "__main__":
    main()
