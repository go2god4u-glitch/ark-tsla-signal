"""대시보드를 실제로 렌더해서 그림으로 본다.

왜 필요한가:
  범례 아이콘이 세로로 쌓인 적이 있다. 전역 `svg { width:100% }` 가
  14px 아이콘에도 걸려 하나가 가로 전체를 먹었기 때문이다.
  코드만 읽어서는 안 보인다 — CSS 는 계산해봐야 아는 것이고,
  계산은 브라우저가 한다. 눈으로 안 봤기 때문에 그대로 배포됐다.

  화면을 바꿨으면 이 스크립트를 돌리고 나온 PNG 를 실제로 열어봐라.
  "코드상 맞다" 는 렌더링 검증이 아니다.

사용:
  python shot.py                 # 전체 페이지
  python shot.py 1370 1420       # y 구간만 잘라 확대 (범례 등 작은 것 확인용)

의존:
  playwright 의 chrome-headless-shell 을 그대로 빌려 쓴다. 별도 설치 없이
  이미 캐시에 있으면 동작한다. 없으면 어디서 받는지 알려주고 끝낸다.
"""

import glob
import os
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = 8901
OUT = os.path.join(BASE, "_shot")


def find_shell() -> str | None:
    pats = [
        os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/"
                           "chrome-headless-shell-*/chrome-headless-shell"),
        os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/"
                           "chrome-headless-shell-*/chrome-headless-shell"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",):
        if os.path.exists(p):
            return p
    return None


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        # 캐시가 남으면 방금 고친 것을 못 본다. 검증용 서버이므로 항상 새로 준다.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    shell = find_shell()
    if not shell:
        print("헤드리스 크롬을 못 찾았다.\n"
              "  npx playwright install chromium   (또는 Chrome 설치)")
        raise SystemExit(1)

    os.makedirs(OUT, exist_ok=True)
    srv = ThreadingHTTPServer(
        ("127.0.0.1", PORT), partial(Quiet, directory=BASE))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)

    png = os.path.join(OUT, "dash.png")
    subprocess.run(
        [shell, "--headless", "--disable-gpu", "--no-sandbox",
         "--hide-scrollbars", "--window-size=1280,2400",
         f"--screenshot={png}", "--virtual-time-budget=6000",
         f"http://127.0.0.1:{PORT}/index.html"],
        capture_output=True, check=True)
    srv.shutdown()
    print(f"저장: {png}")

    if len(sys.argv) >= 3:
        try:
            from PIL import Image
        except ImportError:
            print("자르기는 pillow 가 필요하다: pip install pillow")
            return
        y0, y1 = int(sys.argv[1]), int(sys.argv[2])
        im = Image.open(png)
        crop = im.crop((0, y0, im.width, y1))
        k = max(1, 1280 // max(1, crop.width))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        path = os.path.join(OUT, f"crop_{y0}_{y1}.png")
        crop.save(path)
        print(f"저장: {path}")

    print("\n이제 이 PNG 를 **열어서 봐라**. 여는 것까지가 검증이다.")


if __name__ == "__main__":
    main()
