"""Serve the dashboard (static files) on http://localhost:8080. The simulation itself comes from serve.py on :8000."""
import functools, http.server, sys
from pathlib import Path
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
print(f"dashboard on http://localhost:{port}"); http.server.ThreadingHTTPServer(("0.0.0.0", port), h).serve_forever()
