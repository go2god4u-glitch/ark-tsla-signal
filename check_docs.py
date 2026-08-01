"""문서와 코드가 어긋나지 않는지 대조한다.

규칙을 여러 번 바꾸면서 문서 곳곳에 옛 값이 남았다. 사람이 눈으로 찾으면 놓친다.
코드의 상수를 읽어 현재 사양 문서에 그 값이 실제로 적혀 있는지 확인한다.

'현재 사양' 문서만 검사한다. DECISIONS·CLAUDE 는 정정 이력이므로
옛 값이 남아 있는 것이 정상이다.
"""
import re
import sys

# 전체 사양을 담는 문서 / 요약만 담는 문서를 구분한다.
# docs/README.md 는 한 줄 요약이므로 상한 같은 세부값을 요구하지 않는다.
FULL_DOCS = ["README.md", "docs/ALGORITHM.md", "docs/HOW_IT_WORKS.md"]
SUMMARY_DOCS = ["docs/README.md"]
SPEC_DOCS = FULL_DOCS + SUMMARY_DOCS

def const(path, name):
    m = re.search(rf'^{name}\s*=\s*(-?[\d.]+)', open(path).read(), re.M)
    return float(m.group(1)) if m else None


def main():
    vals = {
        "매수 문턱 분위수": const("signal_check.py", "QUANT"),
        "매수 낙폭 필터": const("signal_check.py", "DD_FILTER"),
        "규칙 A 아크 감소": const("position_tracker.py", "ARK_DROP"),
        "규칙 A 낙폭 회복": const("position_tracker.py", "DD_RECOVER"),
        "규칙 B 대량매도": const("position_tracker.py", "BIG_SELL"),
        "보유 상한": const("position_tracker.py", "MAX_HOLD"),
    }
    # 문서에 나타나야 할 표기
    want = {
        "매수 문턱 분위수": ["90분위", "상위 10%"],
        "매수 낙폭 필터": ["-30%"],
        "규칙 A 아크 감소": ["-20%"],
        "규칙 A 낙폭 회복": ["-20%"],
        "규칙 B 대량매도": ["-7%"],
        "보유 상한": ["504"],
    }
    # 남아 있으면 안 되는 옛 표현 (현재 사양 문서 기준)
    stale = [
        (r"6개월 뒤 매도", "폐기된 6개월 고정 규칙"),
        (r"매도는 시간", "폐기된 요약"),
        (r"RSI ?70 을 넘었다가", "폐기된 RSI 이탈 규칙"),
        (r"매도\s*=\s*RSI", "폐기된 RSI 매도"),
    ]

    print("코드 상수:")
    for k, v in vals.items():
        print(f"  {k:<18s} {v}")
    print()

    bad = 0
    for doc in SPEC_DOCS:
        txt = open(doc).read()
        miss = [k for k, pats in want.items()
                if not any(p in txt for p in pats)]
        old = [d for pat, d in stale if re.search(pat, txt)]
        full = doc in FULL_DOCS
        if full and miss:
            print(f"  ⚠ {doc}: 누락 {miss}")
            bad += 1
        if old:
            print(f"  ⚠ {doc}: 옛 표현 {old}")
            bad += 1
        if not (full and miss) and not old:
            print(f"  ✓ {doc}")

    bad += check_data_files()

    print()
    if bad:
        print(f"불일치 {bad}건")
        sys.exit(1)
    print("문서와 코드 일치")


def check_data_files() -> int:
    """data/*.json 이 전부 파싱되는가.

    리베이스 충돌을 풀다가 `data/tsla_full.json` 에 충돌 마커(<<<<<<<)를 남긴 채
    커밋하고 푸시한 적이 있다. 사이트가 읽는 원본이라 대시보드가 통째로 죽는다.
    파이썬은 그 파일을 안 열어보면 모르고, 커밋도 조용히 통과한다.
    푸시 전 검사에 넣어 두면 다시는 못 지나간다.
    """
    import glob
    import json
    import os

    base = os.path.dirname(os.path.abspath(__file__))
    print()
    bad, files = 0, sorted(glob.glob(os.path.join(base, "data", "*.json")))
    for p in files:
        raw = open(p, encoding="utf-8", errors="replace").read()
        name = os.path.relpath(p, base)
        if "<<<<<<<" in raw or ">>>>>>>" in raw:
            print(f"  ⚠ {name}: 병합 충돌 마커가 남아 있다")
            bad += 1
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ⚠ {name}: JSON 파싱 실패 — {e}")
            bad += 1
    if not bad:
        print(f"  ✓ data/*.json {len(files)}개 전부 정상")
    return bad


if __name__ == "__main__":
    main()
