"""Neural style step: photo -> stylised image, with free open-source models that run on CPU.

  realistic  animegan2 "face_paint_512_v2" (MIT)       smooth painted portrait, true colours
  ghibli     AnimeGANv3 "Ghibli_c1" (free, non-commercial) Studio-Ghibli-like portrait
  toon       animegan2 "celeba_distill" (MIT)          flat comic shading with ink lines

Models live in api/models/ (7-9 MB each) and are loaded once per process.
"""
import os
import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

STYLES = {
    'realistic': dict(file='animegan2_face_paint_512_v2.onnx', layout='nchw', side=768, mult=32),
    'ghibli':    dict(file='AnimeGANv3_large_Ghibli_c1_e299.onnx', layout='nhwc', side=1024, mult=8),
    'toon':      dict(file='animegan2_celeba_distill.onnx', layout='nchw', side=512, mult=32),
}

_sessions = {}


def session(style):
    if style not in _sessions:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        _sessions[style] = ort.InferenceSession(os.path.join(MODELS_DIR, STYLES[style]['file']), opts,
                                                providers=['CPUExecutionProvider'])
    return _sessions[style]


def stylize(rgb, style):
    """rgb: HxWx3 uint8 -> HxWx3 uint8 of the same size."""
    cfg = STYLES[style]
    sess = session(style)
    h, w = rgb.shape[:2]
    s = min(1.0, cfg['side'] / max(h, w))
    m = cfg['mult']
    nh, nw = max(m, int(h * s) // m * m), max(m, int(w * s) // m * m)
    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1.0
    if cfg['layout'] == 'nchw':
        x = x.transpose(2, 0, 1)
    y = sess.run(None, {sess.get_inputs()[0].name: x[None]})[0][0]
    if cfg['layout'] == 'nchw':
        y = y.transpose(1, 2, 0)
    y = ((y + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
