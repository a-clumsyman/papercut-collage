"""Shared rendering + scoring for papercut results (any engine).

A result is the JSON the API returns: {width, height, pieces:[{d, L, n, ...}]}, plus a
`frame` we attach here: (x0, y0, s) mapping result coords -> source photo coords:
    src = (x0 + x / s, y0 + y / s)
"""
import json, re
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

RAMP_GREEN = ['#1e5c38', '#3d8a4e', '#79c47c', '#c9ecc4']
PAPER_BG = '#eaf2f5'
A4_MM = (190.0, 277.0)  # printable area


def hex2rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def app_colors(result, ramp=RAMP_GREEN):
    """Exactly the app's pieceColor(): bucket L between min/max into 4 ramp steps."""
    Ls = [p['L'] for p in result['pieces']]
    lo, hi = min(Ls), max(Ls)
    out = []
    for p in result['pieces']:
        t = (p['L'] - lo) / (hi - lo) if hi > lo else .5
        out.append(ramp[max(0, min(3, int(np.floor(t * 4))))])
    return out


def loops_of(d):
    loops = []
    for sub in re.findall(r'M[^M]*', d):
        nums = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', sub)]
        pts = np.array(nums, float).reshape(-1, 2)
        if len(pts) >= 3: loops.append(pts)
    return loops


def piece_mask(d, shape, scale=1.0, offset=(0, 0)):
    """Even-odd fill of one piece's path."""
    m = np.zeros(shape, np.uint8)
    for lp in loops_of(d):
        t = np.zeros(shape, np.uint8)
        q = np.round(((lp - offset) * scale) * 8).astype(np.int32)
        cv2.fillPoly(t, [q], 1, lineType=cv2.LINE_8, shift=3)
        m ^= t
    return m.astype(bool)


def render(result, colors=None, scale=1.0, bg=PAPER_BG):
    colors = colors or app_colors(result)
    w, h = int(round(result['width'] * scale)), int(round(result['height'] * scale))
    img = np.zeros((h, w, 3), np.uint8); img[:] = hex2rgb(bg)
    for p, c in zip(result['pieces'], colors):
        img[piece_mask(p['d'], (h, w), scale)] = hex2rgb(c)
    return img


def lum(rgb):
    rgb = rgb.astype(float); return .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]


# ---------------- likeness ----------------
FEATURES = {
    'eyes+brows': [33, 133, 159, 145, 362, 263, 386, 374, 70, 105, 107, 336, 334, 300],
    'nose': [6, 197, 4, 1, 2, 98, 327, 64, 294],
    'mouth': [61, 291, 0, 17, 13, 14, 40, 270, 91, 321],
}


def head_box(lm_px, W, H):
    x0, y0 = lm_px.min(0); x1, y1 = lm_px.max(0)
    fw, fh = x1 - x0, y1 - y0
    b = [x0 - .30 * fw, y0 - .45 * fh, x1 + .30 * fw, y1 + .25 * fh]
    return [int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3]))]


def likeness(result, frame, cache_dir):
    """Compare the collage to the photo inside the head region.

    Both images are grey, background-flattened, blurred to 'paper resolution' and
    z-normalised, so the score measures WHERE light and dark fall (the likeness),
    not which exact paper colours were chosen.
    """
    photo = np.asarray(Image.open(f'{cache_dir}/rgb.png').convert('RGB'))
    cat = np.asarray(Image.open(f'{cache_dir}/cat.png'))
    lm = np.array(json.load(open(f'{cache_dir}/lm.json'))['landmarks']) * [photo.shape[1], photo.shape[0]]
    H, W = cat.shape
    x0, y0, s = frame
    # render collage into source-photo space
    colors = result.get('_colors') or app_colors(result)
    coll = np.zeros((H, W, 3), np.uint8); coll[:] = hex2rgb(PAPER_BG)
    cover = np.zeros((H, W), bool)
    for p, c in zip(result['pieces'], colors):
        m = piece_mask(p['d'], (H, W), scale=1 / s, offset=(-x0 * s, -y0 * s))
        coll[m] = hex2rgb(c); cover |= m
    bx0, by0, bx1, by1 = head_box(lm, W, H)
    fw = lm[:, 0].max() - lm[:, 0].min()
    k = 256.0 / (bx1 - bx0)
    person = cat[by0:by1, bx0:bx1] > 0
    G = lum(photo[by0:by1, bx0:bx1]); R = lum(coll[by0:by1, bx0:bx1])
    G = np.where(person, G, np.median(G[~person]) if (~person).any() else G.mean())
    sz = (256, int(round((by1 - by0) * k)))
    G = cv2.resize(G, sz, interpolation=cv2.INTER_AREA); R = cv2.resize(R, sz, interpolation=cv2.INTER_AREA)
    sig = 0.012 * fw * k
    G = cv2.GaussianBlur(G, (0, 0), sig); R = cv2.GaussianBlur(R, (0, 0), sig)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    Gz, Rz = z(G), z(R)
    rng = max(Gz.max(), Rz.max()) - min(Gz.min(), Rz.min())
    _, smap = ssim(Gz, Rz, data_range=rng, full=True, gaussian_weights=True, sigma=2.0)
    head = float(smap.mean())
    feats = {}
    for name, idx in FEATURES.items():
        p = (lm[idx] - [bx0, by0]) * k
        fx0, fy0 = p.min(0) - .08 * fw * k; fx1, fy1 = p.max(0) + .08 * fw * k
        sl = smap[int(max(0, fy0)):int(fy1), int(max(0, fx0)):int(fx1)]
        feats[name] = float(sl.mean()) if sl.size else 0.0
    feat = float(np.mean(list(feats.values())))
    pm = cat[by0:by1, bx0:bx1] > 0; cm = cover[by0:by1, bx0:bx1]
    iou = float((pm & cm).sum() / max(1, (pm | cm).sum()))
    score = .4 * head + .4 * feat + .2 * iou
    return {'likeness': round(100 * score, 1), 'head_ssim': round(head, 3), 'feature_ssim': round(feat, 3),
            'silhouette_iou': round(iou, 3), **{k2: round(v, 3) for k2, v in feats.items()}}


# ---------------- cuttability ----------------
def cuttability(result, min_width_mm=4.0):
    """Scale the sheet to fit A4 and measure every piece at 5 px/mm."""
    w, h = result['width'], result['height']
    mm_per_unit = min(A4_MM[0] / w, A4_MM[1] / h)
    ppm = 5.0
    sc = mm_per_unit * ppm
    shape = (int(h * sc) + 2, int(w * sc) + 2)
    r = int(round(min_width_mm * ppm / 2))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    areas, thin_px, total_px, holes, fiddly_pieces, cut_len = [], 0, 0, 0, 0, 0.0
    for p in result['pieces']:
        m = piece_mask(p['d'], shape, sc).astype(np.uint8)
        a = int(m.sum()); total_px += a
        areas.append(a / ppm ** 2)
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk)
        thin = a - int(opened.sum()); thin_px += thin
        if a == 0 or thin / a > .03: fiddly_pieces += 1
        loops = loops_of(p['d']); holes += max(0, len(loops) - 1)
        for lp in loops:
            cut_len += float(np.sum(np.hypot(*np.diff(np.vstack([lp, lp[:1]]), axis=0).T))) * mm_per_unit
    areas = np.array(areas)
    return {'pieces': len(areas), 'tiny_pieces(<1cm2)': int((areas < 100).sum()),
            'smallest_mm2': round(float(areas.min()), 1), 'fiddly_pieces': fiddly_pieces,
            'thin_area_%': round(100 * thin_px / max(1, total_px), 1), 'holes': holes,
            'cut_length_m': round(cut_len / 1000, 2),
            'face_width_mm': None}
