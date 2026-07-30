"""로컬 미리보기 서버 — 배포 전에 브라우저로 직접 확인한다.

charset 을 붙이는 이유: 기본 SimpleHTTPRequestHandler 는 text/html 에
charset 을 안 붙여서 한글이 깨진다. GitHub Pages 는 붙여주므로
그대로 두면 로컬에서만 깨져 헷갈린다.

캐시를 끄는 이유: data/*.json 을 고치고 새로고침해도 옛 값이 보이면
고친 것을 확인할 수 없다.
"""
import http.server
import socketserver

PORT = 8899

class H(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        t = super().guess_type(path)
        return t + "; charset=utf-8" if t in ("text/html", "text/plain") else t

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
        print(f"미리보기: http://127.0.0.1:{PORT}/")
        print("  종료: Ctrl+C")
        httpd.serve_forever()
