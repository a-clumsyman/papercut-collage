from http.server import BaseHTTPRequestHandler
import json
from .cut import cut

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > 12000000:
                raise ValueError('Photo is too large. Choose another photo.')
            data = cut(json.loads(self.rfile.read(size)))
            body = json.dumps(data).encode()
            status = 200 if 'error' not in data else 400
        except Exception as exc:
            body = json.dumps({'error':str(exc)}).encode()
            status = 400
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
