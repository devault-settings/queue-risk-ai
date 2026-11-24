"""A local web server for the demo page, for browsers that block `file://` data files.

    python demo/serve.py        # then open http://localhost:8000
"""
import functools
import http.server
import webbrowser
from pathlib import Path

PORT = 8000
HERE = Path(__file__).resolve().parent

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Tell the browser not to cache: after a rerun of export_data.py, a plain reload must pick
    up the new data.js (a cached old data.js with a newer index.html breaks the page)."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


handler = functools.partial(NoCacheHandler, directory=str(HERE))
with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"Serving {HERE.name}/ at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    httpd.serve_forever()
