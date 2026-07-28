"""arkfunds.io 의 '파일 전체 스케일링' 오류를 걸러낸다.

관측된 오류의 정체:
  2025-08-12 ~ 08-18 사이 ARKK 의 보고 주식 수가 2.34M -> 3.99M -> 2.34M 로
  왕복한다. 같은 기간 비중은 10.2~10.9% 로 평평하다. 즉 테슬라 한 종목이 아니라
  펀드 파일 전체가 부풀려졌다. 내재 AUM(= market_value / weight)으로 보면
  7.3B -> 12.65B -> 7.1B 다. 대형 ETF 가 일주일 만에 $5.5B 를 얻었다 잃을 수 없다.

왜 중앙값 필터로는 못 잡나:
  왜곡이 4관측 이상 이어지면 중앙값 창이 왜곡에 같이 끌려간다.
  실제로 5관측 중앙값 기준 20% 필터는 이 구간을 통과시켰다.

그래서 '되돌림'을 직접 본다:
  진짜 매집은 새 수준으로 올라가서 머문다 -> 이후 구간의 중앙값도 함께 올라간다.
  가짜 스파이크는 원래 수준으로 되돌아온다 -> 앞 구간과 뒤 구간 중앙값 '양쪽'에서
  같은 방향으로 벗어난다. 이 조건으로만 버린다. 추세는 살고 왕복은 죽는다.
"""

import json
import os
import glob

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
FUNDS = ("ARKK", "ARKQ")
SPLIT = pd.Timestamp("2022-08-25")

K = 7           # 앞뒤로 볼 관측 수
BAND = 0.12     # 양쪽 중앙값에서 12% 넘게 벗어나면 의심
PASSES = 3      # 넓은 왜곡은 바깥쪽부터 한 겹씩 벗겨진다


def load_raw() -> pd.DataFrame:
    rows = {}
    for path in glob.glob(os.path.join(BASE, "data", "raw", "*.json")):
        with open(path) as f:
            payload = json.load(f)
        for day in payload.get("data", []):
            for o in day.get("ownership", []):
                if o["fund"] in FUNDS:
                    rows[(o["date"], o["fund"])] = o
    d = pd.DataFrame(rows.values())
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["fund", "date"]).reset_index(drop=True)


def flag_roundtrips(s: pd.Series) -> pd.Series:
    """앞뒤 중앙값 양쪽에서 같은 방향으로 벗어난 관측을 True 로 표시한다."""
    bad = pd.Series(False, index=s.index)
    work = s.copy()
    for _ in range(PASSES):
        prev_med = work.shift(1).rolling(K, min_periods=3).median()
        next_med = work.shift(-1)[::-1].rolling(K, min_periods=3).median()[::-1]
        dev_p = work / prev_med - 1
        dev_n = work / next_med - 1
        hit = ((dev_p.abs() > BAND) & (dev_n.abs() > BAND)
               & (np.sign(dev_p) == np.sign(dev_n)) & ~bad)
        if not hit.any():
            break
        bad |= hit.fillna(False)
        work = work.mask(bad)          # 걸러낸 값은 다음 패스의 중앙값에서 제외
        work = work.interpolate(limit_direction="both")
    return bad


def main() -> None:
    d = load_raw()

    # 분할 보정: 날짜가 아니라 내재 주가로 판정 (build_chart.py 와 동일한 근거)
    px = pd.read_json(os.path.join(BASE, "data", "tsla_yahoo.json"))
    r = json.load(open(os.path.join(BASE, "data", "tsla_yahoo.json")))["chart"]["result"][0]
    close = pd.Series(r["indicators"]["quote"][0]["close"],
                      index=pd.to_datetime(pd.Series(r["timestamp"]), unit="s").dt.normalize())
    close = close.dropna()
    d["close"] = d["date"].map(close)
    ratio = (d["market_value"] / d["shares"]) / d["close"]
    d["shares_adj"] = np.where(ratio > 2, d["shares"] * 3, d["shares"])

    d["aum"] = d["market_value"] / (d["weight"] / 100)

    out = []
    for fund, g in d.groupby("fund"):
        g = g.sort_values("date").reset_index(drop=True)
        bad = flag_roundtrips(g["aum"])
        print(f"[{fund}] 왕복 스파이크 {int(bad.sum())}행 제거")
        if bad.any():
            for dt, a in zip(g.loc[bad, "date"], g.loc[bad, "aum"]):
                print(f"    {dt:%Y-%m-%d}  AUM ${a/1e9:.2f}B")
        g = g[~bad]
        out.append(g)

    d = pd.concat(out)
    wide_sh = d.pivot(index="date", columns="fund", values="shares_adj")
    wide_wt = d.pivot(index="date", columns="fund", values="weight")
    aum_k = d[d["fund"] == "ARKK"].set_index("date")["aum"]

    keep = wide_sh[list(FUNDS)].dropna().index
    print(f"[결측] 한쪽 펀드만 남은 날 {len(wide_sh) - len(keep)}일 제외")

    res = pd.DataFrame({
        "ARKK_shares_adj": wide_sh.loc[keep, "ARKK"],
        "ARKQ_shares_adj": wide_sh.loc[keep, "ARKQ"],
        "combined_shares_adj": wide_sh.loc[keep, list(FUNDS)].sum(axis=1),
        "ARKK_weight_pct": wide_wt.loc[keep, "ARKK"],
        "ARKQ_weight_pct": wide_wt.loc[keep, "ARKQ"],
        "ARKK_aum": aum_k.reindex(keep),
    })
    res["TSLA_close"] = close.reindex(res.index)
    res.to_csv(os.path.join(BASE, "data", "holdings_clean.csv"))

    c = res["combined_shares_adj"]
    print(f"\n관측일 {len(res)}  {res.index.min():%Y-%m-%d} ~ {res.index.max():%Y-%m-%d}")
    print(f"합산 보유 {c.iloc[0]/1e6:.2f}M -> {c.iloc[-1]/1e6:.2f}M  최고 {c.max()/1e6:.2f}M")
    dd = c.diff().abs()
    print(f"일변화 중앙값 {dd.median():,.0f}주 / 최대 {dd.max():,.0f}주 "
          f"({c.diff().abs().idxmax():%Y-%m-%d})")
    print("저장: data/holdings_clean.csv")


if __name__ == "__main__":
    main()
