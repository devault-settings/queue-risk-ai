"""A local web server for the demo page, for browsers that block `file://` data files.

    python demo/serve.py        # then open http://localhost:8000
"""
import functools
import http.server
import webbrowser
from pathlib import Path

PORT = 8000
HERE = Path(__file__).resolve().parent

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"Serving {HERE.name}/ at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    httpd.serve_forever()
