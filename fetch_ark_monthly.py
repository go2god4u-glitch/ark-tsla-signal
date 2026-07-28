"""ARK 의 TSLA 보유 이력 수집기 (월 단위 버전).

왜 월 단위인가:
  arkfunds.io 의 fund-ownership 엔드포인트는 '월말일'을 포함한 요청에 대해
  영구적으로 500 을 낸다. 하루짜리 요청이든 30일 범위 요청이든 마찬가지라,
  범위 안에 월말이 한 번이라도 끼면 그 청크 전체가 날아간다.
  (실측: 2023-02-28 / 2023-06-30 / 2025-01-31 / 2026-06-30 전부 500,
   같은 달의 1일~말일-1 범위는 전부 200)

  그래서 각 달을 [1일 ~ 말일-1] 로 잘라 요청한다. 각 달의 마지막 하루만
  결측이 되는데, 5년 시계열에서는 무시할 수 있는 손실이다.

응답의 `totals` 는 ARKW/ARKX 까지 합산한 값이므로 절대 쓰지 않는다.
필요한 펀드(ARKK/ARKQ)만 골라 쓰는 것은 build_chart.py 의 몫이다.
"""

import json
import os
import sys
import time
import urllib.request
from calendar import monthrange
from datetime import date

API = "https://arkfunds.io/api/v2/stock/fund-ownership"
SYMBOL = "TSLA"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

LIMIT = 100  # 한 달 영업일은 최대 23일. 100 이면 truncation 걱정이 없다.
MAX_RETRIES = 4


def fetch(date_from: date, date_to: date) -> dict:
    url = (f"{API}?symbol={SYMBOL}&date_from={date_from}&date_to={date_to}"
           f"&limit={LIMIT}")
    delay = 3.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            if "data" not in payload:
                raise ValueError(f"'data' 키 없음: {str(payload)[:200]}")
            if len(payload["data"]) >= LIMIT:
                raise ValueError(f"limit({LIMIT}) 도달 - truncation 의심")
            return payload
        except Exception as e:  # noqa: BLE001
            print(f"    재시도 {attempt}/{MAX_RETRIES} "
                  f"({getattr(e, 'code', None) or type(e).__name__}) {delay:.0f}s",
                  file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(f"{date_from}~{date_to} 실패: {e}") from e


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    today = date.today()
    failed = []

    for y, m in months(date(2021, 7, 1), today):
        last = monthrange(y, m)[1]
        a = date(y, m, 1)
        b = date(y, m, last - 1)          # 월말 하루를 의도적으로 제외한다
        b = min(b, today)
        if a > b:
            continue
        path = os.path.join(OUT_DIR, f"m_{y}-{m:02d}.json")
        if os.path.exists(path):
            print(f"{y}-{m:02d} 건너뜀(이미 있음)")
            continue
        try:
            payload = fetch(a, b)
        except RuntimeError as e:
            print(f"{y}-{m:02d} !! {e}", file=sys.stderr)
            failed.append(f"{y}-{m:02d}")
            continue
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"{y}-{m:02d} 날짜 {len(payload['data']):>2}개 저장")
        time.sleep(1.2)

    print(f"수집 완료. 실패한 달: {failed or '없음'}")


if __name__ == "__main__":
    main()
