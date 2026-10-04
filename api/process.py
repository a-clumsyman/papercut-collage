"""POST /api/process  ->  paper pieces for the collage.

The browser has already cropped the photo to the sheet and run the style model. Request JSON:
  styled     base64 JPEG/PNG of the stylised head-and-shoulders crop (sheet aspect, max side <= 1600)
  person     base64 PNG mask of the figure in that crop (white = person), any size
  landmarks  478 face-mesh points, normalised [x, y] to the crop
  style      "realistic" | "ghibli" | "toon"
  difficulty "easy" | "medium" (optional, default easy)
"""
import base64, binascii, io, json
from http.server import BaseHTTPRequestHandler

import numpy as np
from PIL import Image

from .collage import cut_styled, DIFF
from .stylize import STYLES

MAX_BODY = 12_000_000
MAX_SIDE = 1600
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE  # refuse decompression bombs early


class UserError(ValueError):
    pass


def _image(b64, mode):
    try:
        raw = base64.b64decode(b64, validate=False)
        im = Image.open(io.BytesIO(raw))
        if max(im.size) > MAX_SIDE:
            raise UserError('Photo is too big. Please use a smaller photo.')
        return np.asarray(im.convert(mode))
    except (binascii.Error, OSError, Image.DecompressionBombError):
        raise UserError('Could not read the photo. Please try another one.')


def run(payload):
    if not isinstance(payload, dict) or not payload.get('styled') or not payload.get('person'):
        raise UserError('Missing photo data.')
    lm = payload.get('landmarks')
    if not isinstance(lm, list) or len(lm) < 468:
        raise UserError('No face found. Try a brighter, face-forward photo.')
    style = payload.get('style', 'ghibli')
    if style not in STYLES:
        raise UserError('Unknown style.')
    difficulty = payload.get('difficulty', 'easy')
    if difficulty not in DIFF:
        difficulty = 'easy'
    rgb = _image(payload['styled'], 'RGB')
    person = _image(payload['person'], 'L')
    if person.shape != rgb.shape[:2]:
        person = np.asarray(Image.fromarray(person).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
    person = person > 127
    if person.mean() < .03:
        raise UserError("Couldn't find you in the photo. Try a plain background and good light.")
    return cut_styled(rgb, person, [p[:2] for p in lm], style, difficulty)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        status = 200
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > MAX_BODY:
                raise UserError('Photo is too large. Choose another photo.')
            body = run(json.loads(self.rfile.read(size)))
        except UserError as exc:
            status, body = 400, {'error': str(exc)}
        except json.JSONDecodeError:
            status, body = 400, {'error': 'Bad request.'}
        except Exception:  # never leak internals to the page
            status, body = 500, {'error': 'Something went wrong while cutting. Please try again.'}
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
