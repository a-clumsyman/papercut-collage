"""Papercut collage engine v3: stacked tone layers from the photo's real light and shadow.

Idea
----
A paper-collage portrait looks like the person when the light and dark shapes fall where
they fall in the photo (that is what stencil art and papercut portraits rely on). So
instead of assembling a face from template parts, we:

  1. crop to head + shoulders so the face fills the sheet,
  2. flatten the photo to a smooth tone image (texture removed, edges kept),
  3. split it into 4 paper tones with thresholds tuned on the FACE,
  4. build stacked layers: layer 0 = whole silhouette (lightest paper), layer k = every
     area at least as dark as tone k. Each darker layer is glued on top of the last,
     so small cutting errors are hidden and pieces never have to meet edge-to-edge,
  5. make every layer cuttable in real millimetres: no part narrower than the minimum
     width, no crumbs, no interior holes (a lighter hole becomes its own piece on top),
  6. search a few settings and keep the most photo-like result within the piece budget.

Everything is numpy + OpenCV, so it runs as a Vercel Python function.
"""
import math
import cv2
import numpy as np

# Paper sheet (portrait A4 printable area) and working resolution
SHEET_MM = (190.0, 277.0)
PPM = 4.0  # working pixels per millimetre -> 760 x 1108 px sheet

TONES_L = [215, 150, 90, 35]  # layer 0 (lightest) .. layer 3 (darkest); app maps these to ramp colours

DIFFICULTY = {
    #          min width  min piece  feature zones (eyes/brows/nose/mouth)  piece budget
    'easy':   dict(min_w=4.0, min_a=100.0, feat_w=2.5, feat_a=25.0, max_pieces=30, min_pieces=12),
    'medium': dict(min_w=3.0, min_a=60.0, feat_w=2.0, feat_a=15.0, max_pieces=60, min_pieces=25),
}

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_R = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
NOSE = [6, 197, 195, 5, 4, 1, 2, 98, 327, 64, 294, 129, 358, 49, 279]


# ---------------------------------------------------------------- helpers
def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_poly(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts) * 8).astype(np.int32)], 1, shift=3)
    return m.astype(bool)


def hull_mask(shape, pts, grow=1.0):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    h = cv2.convexHull(((pts - c) * grow + c).astype(np.float32)).reshape(-1, 2)
    return fill_poly(shape, h)


def fill_small_holes(m, max_area):
    """Fill holes (background pockets fully inside m) smaller than max_area."""
    inv = (~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m.copy()
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, a = st[i]
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if not touches and a < max_area:
            out[lab == i] = True
    return out


def drop_small(m, min_area, protect=None, min_area_protected=None):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= min_area:
            keep[i] = True
        elif protect is not None and a >= min_area_protected and protect[lab == i].mean() > .5:
            keep[i] = True
    return keep[lab]


def multi_otsu(values, weights, classes=4, bins=64):
    """Exhaustive 3-threshold Otsu on a weighted 64-bin histogram (no scikit-image needed)."""
    hist, edges = np.histogram(values, bins=bins, range=(0, 256), weights=weights)
    p = hist / max(hist.sum(), 1e-9)
    idx = np.arange(bins)
    P = np.concatenate([[0], np.cumsum(p)])
    S = np.concatenate([[0], np.cumsum(p * idx)])

    def var(a, b):  # between-class term for bins [a, b)
        w = P[b] - P[a]
        return 0.0 if w <= 0 else (S[b] - S[a]) ** 2 / w
    best, bt = -1, None
    for t1 in range(1, bins - 2):
        v1 = var(0, t1)
        for t2 in range(t1 + 1, bins - 1):
            v12 = v1 + var(t1, t2)
            for t3 in range(t2 + 1, bins):
                v = v12 + var(t2, t3) + var(t3, bins)
                if v > best:
                    best, bt = v, (t1, t2, t3)
    return [edges[t] for t in bt]  # ascending luminance thresholds


def smooth_contour(c, sigma):
    """Gaussian-smooth a closed contour (N,2) and resample."""
    c = c.astype(float)
    n = len(c)
    if n < 8 or sigma <= 0:
        return c
    k = int(3 * sigma) | 1
    k = min(k, (n // 2) * 2 - 1)
    pad = np.vstack([c[-k:], c, c[:k]])
    g = cv2.getGaussianKernel(2 * k + 1, sigma).ravel()
    xs = np.convolve(pad[:, 0], g, 'same')[k:k + n]
    ys = np.convolve(pad[:, 1], g, 'same')[k:k + n]
    return np.stack([xs, ys], 1)


def mask_to_path(m, px_per_unit, smooth_px, eps_px):
    """Single-component hole-free mask -> SVG path (sheet units = mm) + label point."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    c = smooth_contour(c, smooth_px)
    c = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), eps_px, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    pts = (c + .5) / px_per_unit
    d = 'M' + ' L'.join('%.1f,%.1f' % (x, y) for x, y in pts) + ' Z'
    # label at the point farthest from the edge (always inside, even for C shapes)
    dt = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    ly, lx = np.unravel_index(np.argmax(dt), dt.shape)
    return d, (lx + .5) / px_per_unit, (ly + .5) / px_per_unit, float(dt.max()) / px_per_unit


# ---------------------------------------------------------------- stages
def crop_head_shoulders(rgb, person, lm_px):
    """Crop (and scale) so the head+shoulders fill a portrait A4 sheet."""
    H, W = person.shape
    face = lm_px[FACE_OVAL]
    fx0, fy0 = face.min(0); fx1, fy1 = face.max(0)
    fw, fh = fx1 - fx0, fy1 - fy0
    # hair top: highest person pixel in the column band above the face
    band = person[:, int(max(0, fx0)):int(min(W, fx1))]
    rows = np.where(band.any(1))[0]
    top = rows.min() if len(rows) else fy0 - .5 * fh
    top = max(top, fy0 - 1.0 * fh)
    y0 = top - .10 * fh
    y1 = fy1 + 1.05 * fh               # well into the shoulders
    ch = y1 - y0
    cw = max(ch * SHEET_MM[0] / SHEET_MM[1], 2.3 * fw)
    ch = cw * SHEET_MM[1] / SHEET_MM[0]
    # The photo must cover the sheet's bottom edge, or the body ends in a flat cut / streaks.
    # First use headroom (move the crop up), then zoom in (never tighter than 1.7 face widths).
    if y0 + ch > H:
        y0 = max(0.0, H - ch) if H - ch >= 0 else y0
        if y0 + ch > H:
            y0 = max(0.0, min(y0, top - .06 * fh))
            ch = max(H - y0, 1.7 * fw * SHEET_MM[1] / SHEET_MM[0])
            cw = ch * SHEET_MM[0] / SHEET_MM[1]
    x0 = (fx0 + fx1) / 2 - cw / 2
    if cw <= W:
        x0 = min(max(x0, 0.0), W - cw)
    s = SHEET_MM[0] * PPM / cw       # source px -> working px
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    size = (int(round(SHEET_MM[0] * PPM)), int(round(SHEET_MM[1] * PPM)))
    rgb_c = cv2.warpAffine(rgb, M, size, flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    per_c = cv2.warpAffine(person.astype(np.float32), M, size, flags=cv2.INTER_LINEAR, borderValue=0)
    lm_c = (lm_px - [x0, y0]) * s
    return rgb_c, per_c, lm_c, (float(x0), float(y0), float(s))


def tone_image(rgb, person_soft, lm, smooth_mm, boost=0.0):
    Y = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (255 / 255.0)
    # silhouette: soft mask refined a little, then cleaned
    sil = cv2.GaussianBlur(person_soft, (0, 0), 1.5 * PPM) > .5
    face_core = hull_mask(sil.shape, lm[FACE_OVAL], .8)
    n, lab = cv2.connectedComponents(sil.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[face_core & sil]); ids = ids[ids > 0]
    if len(ids):
        sil = np.isin(lab, ids)  # drop other people, microphones, flags...
    sil = fill_small_holes(sil, 400 * PPM * PPM)
    # edge-preserving flatten, then smooth at paper scale
    Yu = Y.astype(np.uint8)
    for _ in range(2):
        Yu = cv2.bilateralFilter(Yu, 0, 18, 1.2 * PPM)
    Yf = cv2.GaussianBlur(Yu.astype(np.float32), (0, 0), smooth_mm * PPM)
    if boost:
        # local contrast at feature scale: eyes, brows, nostrils and mouth pop out of the face
        # the way an artist would exaggerate them, whatever the overall exposure
        fill = np.where(sil, Yf, np.median(Yf[sil]))
        Yf = np.clip(Yf + boost * (Yf - cv2.GaussianBlur(fill, (0, 0), 7 * PPM)), 0, 255)
    # gentle local-contrast lift inside the face so low-contrast photos still get features
    face = hull_mask(Y.shape, lm[FACE_OVAL])
    return Yf, sil, face


def build_layers(Yf, sil, face, feat, thr, P, extra=None):
    """thr: 3 ascending luminance thresholds. Returns list of cleaned boolean masks, layer 0..3."""
    rw = P['min_w'] * PPM / 2
    rf = P['feat_w'] * PPM / 2
    min_a = P['min_a'] * PPM * PPM
    feat_a = P['feat_a'] * PPM * PPM
    dark = [sil]
    for k, t in enumerate(sorted(thr, reverse=True)):  # lighter threshold -> layer 1
        raw = sil & (Yf < t)
        if extra is not None:
            raw |= extra[k + 1]
        # everywhere: close slits, open slivers at the full min width
        big = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, disk(rw * .75))
        big = cv2.morphologyEx(big, cv2.MORPH_OPEN, disk(rw)).astype(bool)
        # feature zones: allow finer detail (eyes, brows, nostrils, mouth line)
        fine = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, disk(rf * .75))
        fine = cv2.morphologyEx(fine, cv2.MORPH_OPEN, disk(rf)).astype(bool)
        m = np.where(feat, fine, big) & sil
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(rf)).astype(bool)
        m = drop_small(m, min_a, protect=feat, min_area_protected=feat_a)
        m = fill_small_holes(m, min_a)
        dark.append(m)
    return dark


NOSTRILS = [98, 327, 2, 64, 294, 240, 460, 97, 326, 19, 94]


def feature_marks(Yf, sil, face, lm, shape):
    """Locally-adaptive feature shapes: eyes, brows, nostrils and the mouth line, found with a
    threshold computed INSIDE each feature's own neighbourhood. This is what an artist does:
    the eyes are drawn dark relative to the cheek next to them, whatever the overall exposure.
    Returns {layer_index: mask} to be merged into the tone layers."""
    out = {1: np.zeros(shape, bool), 2: np.zeros(shape, bool), 3: np.zeros(shape, bool)}
    face_med = float(np.median(Yf[face & sil])) if (face & sil).any() else 128.0
    spec = [  # landmarks, ROI growth, darkness quantile inside ROI, minimum layer
        (EYE_L, 1.7, .30, 3), (EYE_R, 1.7, .30, 3),
        (BROW_L, 1.5, .40, 2), (BROW_R, 1.5, .40, 2),
        (NOSTRILS, 1.15, .22, 2), (LIPS, 1.12, .40, 2),
    ]
    for idx, grow, q, layer in spec:
        roi = hull_mask(shape, lm[idx], grow) & sil
        if roi.sum() < 30:
            continue
        v = Yf[roi]
        t = float(np.quantile(v, q))
        lo, hi = float(np.median(v[v < t])) if (v < t).any() else t, float(np.median(v[v >= t]))
        if hi - lo < max(8.0, .07 * face_med):  # no real feature contrast here: draw nothing
            continue
        out[layer] |= roi & (Yf < t)
    # Eyes: the aperture itself, measured from the mesh (so heavy lids, wide eyes and squints all
    # keep their own shape), made at least 3 mm tall so it can be cut. Darkest paper.
    for ring in (EYE_L, EYE_R):
        e = lm[ring].copy(); c = e.mean(0)
        hgt = np.ptp(e[:, 1]); need = 3.0 * PPM
        sy = max(1.0, need / max(hgt, 1e-3)) * 1.05
        e = (e - c) * [1.08, sy] + c
        out[3] |= fill_poly(shape, e)
    return out


def layers_to_pieces(layers, target, P):
    """Split each layer into hole-free pieces. A remaining hole becomes a lighter piece placed
    right after its layer (a 'highlight'), so no piece ever needs an interior cut."""
    pieces = []  # (tone_index, mask)
    min_a = P['min_a'] * PPM * PPM
    for k, m in enumerate(layers):
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        highlights = []
        for i in range(1, n):
            comp = lab == i
            solid = fill_small_holes(comp, 10 ** 9)
            pieces.append((k, solid))
            holes = solid & ~comp
            if holes.any():
                hn, hl = cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)
                for j in range(1, hn):
                    h = hl == j
                    if h.sum() < min_a:
                        continue
                    tone = int(np.clip(np.round(np.median(target[h])), 0, max(0, k - 1)))
                    # shrink slightly so the highlight sits inside, with the dark rim visible
                    highlights.append((tone, h))
        pieces += highlights
    return pieces


def render_tones(pieces, shape):
    img = np.full(shape, -1, np.int8)
    for tone, m in pieces:
        img[m] = tone
    return img


def ssim_score(A, B, mask, weights):
    """Face-weighted SSIM between two float images (z-normalised in the mask)."""
    def z(a):
        v = a[mask]; return (a - v.mean()) / (v.std() + 1e-6)
    A, B = z(A), z(B)
    C1, C2 = (0.01 * 6) ** 2, (0.03 * 6) ** 2
    g = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5 * PPM)
    mA, mB = g(A), g(B)
    sA, sB, sAB = g(A * A) - mA ** 2, g(B * B) - mB ** 2, g(A * B) - mA * mB
    S = ((2 * mA * mB + C1) * (2 * sAB + C2)) / ((mA ** 2 + mB ** 2 + C1) * (sA + sB + C2))
    return float((S * weights)[mask].sum() / weights[mask].sum())


# ---------------------------------------------------------------- main entry
def _prepare(rgb, person, lm_px):
    rgb_c, per_c, lm, frame = crop_head_shoulders(rgb, person, lm_px)
    shape = per_c.shape
    feat = np.zeros(shape, bool)
    for idx, g in ((EYE_L, 1.8), (EYE_R, 1.8), (BROW_L, 1.35), (BROW_R, 1.35), (LIPS, 1.35), (NOSE, 1.25)):
        feat |= hull_mask(shape, lm[idx], g)
    return rgb_c, per_c, lm, frame, feat


def _candidates(Yf, sil, face, feat):
    """Candidate tone splits. (a) weighted multi-Otsu over the person, face-weighted;
    (b) face quantiles: the FACE gets three tones of its own (highlight / mid / shadow),
    and the darkest paper is kept for hair, pupils, deep shadow and dark clothes."""
    wts = np.where(feat, 8.0, np.where(face, 4.0, 1.0))[sil]
    cands = [[t + sh for t in multi_otsu(Yf[sil], wts)] for sh in (-10, 0, 10)]
    fv = Yf[face & sil]
    if fv.size > 500:
        for q in ((.55, .20, .04), (.45, .15, .03), (.65, .30, .08)):
            a, b, c = np.quantile(fv, q)
            cands.append(sorted([float(c), float(b), float(a)]))
    return cands


def _build(rgb_c, per_c, lm, feat, P, smooth_mm, boost, thr):
    Yf, sil, face = tone_image(rgb_c, per_c, lm, smooth_mm, boost)
    marks = feature_marks(Yf, sil, face, lm, sil.shape)
    target = 3 - np.digitize(Yf, sorted(thr))  # tone index: 0 light..3 dark
    layers = build_layers(Yf, sil, face, feat, thr, P, marks)
    return layers_to_pieces(layers, target, P)


def _search(rgb, person, lm_px, P):
    """Try every (smoothing, contrast, tone split) combination; keep the most photo-like one that
    stays inside the piece budget. Runs at half resolution for speed."""
    rgb_c, per_c, lm, frame, feat = _prepare(rgb, person, lm_px)
    shape = per_c.shape
    Yref, sil0, face0 = tone_image(rgb_c, per_c, lm, 1.0, 0.0)
    ref = np.where(sil0, Yref, np.median(Yref[~sil0]) if (~sil0).any() else 255).astype(np.float32)
    weights = np.where(feat, 4.0, np.where(face0, 2.0, .5)).astype(np.float32)
    box = hull_mask(shape, lm[FACE_OVAL], 1.25) | (sil0 & face0)
    ones = np.ones_like(weights)
    best, tried = None, []
    for smooth_mm, boost in ((0.6, 0.0), (1.0, 0.0), (0.6, 1.0), (1.0, 1.0)):
        Yf, sil, face = tone_image(rgb_c, per_c, lm, smooth_mm, boost)
        marks = feature_marks(Yf, sil, face, lm, shape)
        for ci, thr in enumerate(_candidates(Yf, sil, face, feat)):
            target = 3 - np.digitize(Yf, sorted(thr))
            pieces = layers_to_pieces(build_layers(Yf, sil, face, feat, thr, P, marks), target, P)
            n = len(pieces)
            tones = render_tones(pieces, shape)
            rend = np.where(tones < 0, 250.0, np.take(np.array(TONES_L, float), np.clip(tones, 0, 3))).astype(np.float32)
            # likeness = face detail (weighted to eyes/brows/nose/mouth) + whole-figure tone order
            sc = .75 * ssim_score(ref, rend, box, weights) + .25 * ssim_score(ref, rend, sil0 | box, ones)
            obj = sc - 0.01 * (max(0, n - P['max_pieces']) + max(0, P['min_pieces'] - n))
            tried.append(dict(smooth=smooth_mm, boost=boost, split=ci, pieces=n, score=round(sc, 3)))
            if best is None or obj > best[0]:
                best = (obj, dict(smooth=smooth_mm, boost=boost, split=ci, thr=[float(t) for t in thr]))
    return best[1], tried


def make_collage(rgb, person, landmarks, difficulty='easy', search_ppm=2.0, final_ppm=4.0):
    """rgb: HxWx3 uint8, person: HxW bool/float (segmenter foreground),
    landmarks: 478 normalised [x, y]. Returns API JSON (sheet units are millimetres)."""
    global PPM
    P = DIFFICULTY[difficulty]
    H, W = person.shape
    lm_px = np.asarray(landmarks, float)[:, :2] * [W, H]
    person = person.astype(np.float32)
    try:
        PPM = search_ppm
        cfg, tried = _search(rgb, person, lm_px, P)
        PPM = final_ppm
        rgb_c, per_c, lm, frame, feat = _prepare(rgb, person, lm_px)
        pieces = _build(rgb_c, per_c, lm, feat, P, cfg['smooth'], cfg['boost'], cfg['thr'])
        out = []
        for tone, m in pieces:
            r = mask_to_path(m, PPM, smooth_px=0.6 * PPM, eps_px=0.25 * PPM)
            if r is None:
                continue
            d, cx, cy, inner = r
            out.append({'L': float(TONES_L[tone]), 'tone': tone, 'd': d, 'area': round(float(m.sum() / PPM ** 2), 1),
                        'cx': round(cx, 1), 'cy': round(cy, 1)})
        x0, y0, s = frame
        frame_out = [x0, y0, s / PPM]  # sheet mm -> source px: src = x0 + x / frame[2]
    finally:
        PPM = final_ppm
    # stacking order: lighter layers first (already), then top-to-bottom inside a layer is kept
    for i, p in enumerate(out):
        p['n'] = str(i + 1)
        p['hex'] = '#%02x%02x%02x' % ((int(p['L']),) * 3)
    return {'width': SHEET_MM[0], 'height': SHEET_MM[1], 'units': 'mm', 'count': len(out), 'pieces': out,
            'frame': frame_out, 'config': cfg, 'tried': tried, 'difficulty': difficulty}
