"""ARK Invest 의 TSLA 보유 이력을 arkfunds.io 에서 긁어오는 수집기.

전략:
  - 30일 단위 청크로 잘라 요청한다. rate limit 에 걸리면 지수 백오프로 재시도.
  - 청크별 원본 JSON 을 data/raw/ 에 그대로 저장한다 -> 중단되어도 이어받기 가능.
  - `limit` 파라미터는 '날짜 개수'를 자른다. 청크 길이보다 넉넉히 준 뒤,
    받은 날짜 수가 limit 에 닿았는지 검사해 조용한 truncation 을 잡아낸다.

주의: 응답의 `totals` 필드는 ARKK/ARKQ 뿐 아니라 ARKW/ARKX 까지 합산한 값이다.
      이 스크립트는 totals 를 절대 읽지 않는다. 필요한 펀드만 직접 골라 합산한다.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

API = "https://arkfunds.io/api/v2/stock/fund-ownership"
SYMBOL = "TSLA"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

CHUNK_DAYS = 30
CHUNK_LIMIT = 100  # 30 캘린더일 안의 영업일은 최대 ~23일. 100이면 여유가 충분하다.
MAX_RETRIES = 6


def fetch(date_from: date, date_to: date, retries: int = MAX_RETRIES) -> dict:
    url = (
        f"{API}?symbol={SYMBOL}"
        f"&date_from={date_from.isoformat()}&date_to={date_to.isoformat()}"
        f"&limit={CHUNK_LIMIT}"
    )
    delay = 3.0
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            if "data" not in payload:
                raise ValueError(f"'data' 키 없음: {str(payload)[:200]}")
            # limit 에 닿았다면 잘렸을 수 있다 -> 실패로 간주하고 알린다.
            if len(payload["data"]) >= CHUNK_LIMIT:
                raise ValueError(f"응답 날짜 수가 limit({CHUNK_LIMIT})에 도달 - truncation 의심")
            return payload
        except Exception as e:  # noqa: BLE001 - 네트워크/파싱/한도 전부 재시도 대상
            last_err = e
            code = getattr(e, "code", None)
            print(f"    재시도 {attempt}/{retries} ({code or type(e).__name__}) "
                  f"{delay:.0f}s 대기", file=sys.stderr)
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2, 90)
    raise RuntimeError(f"{date_from}~{date_to} 수집 실패: {last_err}")


def fetch_daywise(date_from: date, date_to: date, dead: list) -> dict:
    """청크 통째 요청이 실패하면 하루씩 받아서 다시 붙인다.

    arkfunds.io 는 특정 날짜(예: 2022-10-31)에 대해 영구적으로 500 을 낸다.
    그 하루 때문에 청크 전체를 버리지 않도록, 죽은 날짜만 골라내고 나머지는 살린다.
    """
    print(f"    -> 청크 실패, 날짜별 재시도로 전환", file=sys.stderr)
    merged = []
    cur = date_from
    while cur <= date_to:
        if cur.weekday() < 5:  # 주말은 애초에 데이터가 없다
            try:
                one = fetch(cur, cur, retries=2)
                merged.extend(one["data"])
            except RuntimeError:
                print(f"    !! {cur} 영구 실패 - 결측 처리", file=sys.stderr)
                dead.append(cur.isoformat())
            time.sleep(1.0)
        cur += timedelta(days=1)
    return {"symbol": SYMBOL, "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(), "data": merged}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    end = date.today()
    start = end - timedelta(days=365 * 5 + 2)  # 5년 + 여유

    chunks = []
    cur = start
    while cur <= end:
        chunks.append((cur, min(cur + timedelta(days=CHUNK_DAYS - 1), end)))
        cur += timedelta(days=CHUNK_DAYS)

    print(f"기간 {start} ~ {end} / 청크 {len(chunks)}개")
    dead: list = []
    for i, (a, b) in enumerate(chunks, 1):
        path = os.path.join(OUT_DIR, f"{a.isoformat()}_{b.isoformat()}.json")
        if os.path.exists(path):
            print(f"[{i:>3}/{len(chunks)}] {a}~{b} 건너뜀(이미 있음)")
            continue
        try:
            payload = fetch(a, b)
        except RuntimeError:
            payload = fetch_daywise(a, b, dead)
        with open(path, "w") as f:
            json.dump(payload, f)
        n = len(payload["data"])
        print(f"[{i:>3}/{len(chunks)}] {a}~{b} 날짜 {n:>2}개 저장")
        time.sleep(1.5)  # rate limit 예방용 간격

    if dead:
        with open(os.path.join(OUT_DIR, "..", "dead_dates.json"), "w") as f:
            json.dump(sorted(dead), f, indent=2)
        print(f"영구 실패 날짜 {len(dead)}개: {sorted(dead)}")
    print("수집 완료")


if __name__ == "__main__":
    main()
