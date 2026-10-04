"""Stylised image -> cuttable paper pieces.

1. Crop to head + shoulders on an A4 sheet (shared with engine.py).
2. Run the chosen style model on the crop (stylize.py).
3. Pick the paper colours: k-means in Lab over the figure, with the face sampled more densely so
   skin tones and features get their own papers. The background gets one paper of its own.
4. Label every pixel with its paper, smooth the label map, then turn it into STACKED pieces:
   big papers are glued first and smaller ones on top; each piece extends a little under the
   pieces above it (no white gaps), tiny crumbs are merged into their neighbours, holes that a
   later piece covers are filled, and uncovered holes become their own piece on top.
   Everything is at least `min_w` mm wide (narrower around the eyes and mouth), so it can be
   cut with scissors.
Coordinates in the result are millimetres on the A4 printable area (190 x 277).
"""
import cv2
import numpy as np

from . import engine as E
from .stylize import stylize, STYLES

PPM = 4.0  # working pixels per mm (760 x 1108 px sheet)

DIFF = {
    #           min width  min piece area  feature min width/area  papers (incl. background)
    'easy':   dict(min_w=4.0, min_a=140.0, feat_w=2.5, feat_a=20.0, underlap=1.5, k=7),
    'medium': dict(min_w=3.0, min_a=70.0, feat_w=2.0, feat_a=12.0, underlap=1.2, k=10),
}


# ----------------------------------------------------------------- helpers
def lab_of(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def hex_of(lab):
    rgb = cv2.cvtColor(np.float32(lab).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)
    r, g, b = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return '#%02x%02x%02x' % (r, g, b)


def kmeans(samples, k, seed=0):
    samples = np.ascontiguousarray(samples, np.float32)
    k = max(1, min(k, len(samples)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, lbl, cen = cv2.kmeans(samples, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    return cen, lbl.ravel()


def smooth_labels(lab_img, k, sigma):
    """Soft 'mode' filter: blur each label's one-hot plane, take the argmax."""
    best = np.full(lab_img.shape, -1.0, np.float32)
    out = np.zeros(lab_img.shape, np.int16)
    for c in range(k):
        p = cv2.GaussianBlur((lab_img == c).astype(np.float32), (0, 0), sigma)
        upd = p > best
        out[upd] = c
        best[upd] = p[upd]
    return out


def absorb_small(labels, k, min_px, protect=None, min_px_protected=None):
    """Regions smaller than min_px are merged into the neighbouring label they touch most."""
    labels = labels.copy()
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed = False
        rounds += 1
        for c in range(k):
            m = (labels == c).astype(np.uint8)
            n, cl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                a = st[i, cv2.CC_STAT_AREA]
                limit = min_px
                comp = cl == i
                if protect is not None and protect[comp].mean() > .5:
                    limit = min_px_protected
                if a >= limit:
                    continue
                ring = cv2.dilate(comp.astype(np.uint8), E.disk(1.5)).astype(bool) & ~comp
                nb = labels[ring]
                nb = nb[nb != c]
                if nb.size:
                    labels[comp] = np.bincount(nb).argmax()
                    changed = True
    return labels


def enforce_min_width(labels, k, rw, feat=None, rf=None):
    """Open every label at the minimum width (finer inside feature zones); pixels that fall out
    are re-assigned to the nearest surviving label by distance."""
    H, W = labels.shape
    opened = np.zeros((k,) + labels.shape, bool)
    for c in range(k):
        m = (labels == c).astype(np.uint8)
        big = cv2.morphologyEx(m, cv2.MORPH_OPEN, E.disk(rw)).astype(bool)
        if feat is not None:
            fine = cv2.morphologyEx(m, cv2.MORPH_OPEN, E.disk(rf)).astype(bool)
            big = np.where(feat, fine, big)
        opened[c] = big
    keep = opened.any(0)
    out = labels.copy()
    # nearest surviving label for the orphans
    if (~keep).any():
        best = np.full(labels.shape, np.inf, np.float32)
        for c in range(k):
            if not opened[c].any():
                continue
            dt = cv2.distanceTransform((~opened[c]).astype(np.uint8), cv2.DIST_L2, 3)
            upd = (~keep) & (dt < best)
            out[upd] = c
            best[upd] = dt[upd]
    # inside kept areas use the opened label (there may be overlaps: pick the smallest-area owner last)
    for c in np.argsort([-opened[c].sum() for c in range(k)]):
        out[opened[c]] = c
    return out


def stack_pieces(labels, order, P, feat):
    """labels: paper index per pixel; order: paper indices bottom -> top. Returns [(paper, mask)]."""
    shape = labels.shape
    rw = P['min_w'] * PPM / 2
    min_a = P['min_a'] * PPM * PPM
    feat_a = P['feat_a'] * PPM * PPM
    under = P['underlap'] * PPM
    H, W = shape
    pieces = []
    above = np.zeros(shape, bool)
    masks = {}
    for i in range(len(order) - 1, -1, -1):
        c = order[i]
        own = labels == c
        masks[c] = (own, above.copy())
        above = above | own
    for c in order:
        own, above_c = masks[c]
        if not own.any():
            continue
        # underlap: extend beneath what is glued on top of it
        m = own | (above_c & cv2.dilate(own.astype(np.uint8), E.disk(under)).astype(bool))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, E.disk(rw * .6)).astype(bool)
        # holes: fill the ones a later piece covers, and the tiny ones
        inv = (~m).astype(np.uint8)
        hn, hl, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        leftover = []
        for j in range(1, hn):
            x, y, w, h, a = st[j]
            if x == 0 or y == 0 or x + w == W or y + h == H:
                continue
            hole = hl == j
            if a < min_a or above_c[hole].mean() > .8:
                m |= hole
            else:
                leftover.append(hole)
        cn, cl, cst, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for j in range(1, cn):
            comp = cl == j
            lim = feat_a if feat[comp].mean() > .5 else min_a
            if cst[j, cv2.CC_STAT_AREA] >= lim * .6:
                pieces.append((c, E.fill_small_holes(comp, 10 ** 9)))
        for hole in leftover:  # visible hole: becomes its own piece on top
            ids = labels[hole]
            pieces.append((int(np.bincount(ids).argmax()), hole))
    return pieces


def ellipse_mask(shape, c, ax, ay):
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(round(c[0] * 8)), int(round(c[1] * 8))), (max(1, int(ax * 8)), max(1, int(ay * 8))), 0, 0, 360, 1, -1, shift=3)
    return m.astype(bool)


def thick_polyline(shape, pts, thick):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(np.asarray(pts) * 8).astype(np.int32)], False, 1, max(1, int(round(thick))), lineType=cv2.LINE_8, shift=3)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, E.disk(thick / 3)).astype(bool)


def min_height_poly(pts, need):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    sy = max(1.0, need / max(np.ptp(pts[:, 1]), 1e-3))
    return (pts - c) * [1.0, sy] + c


UPPER_LID_L = [33, 246, 161, 160, 159, 158, 157, 173, 133]
UPPER_LID_R = [263, 466, 388, 387, 386, 385, 384, 398, 362]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
IRIS_L, IRIS_R = 468, 473
IRIS_RING_L, IRIS_RING_R = [469, 470, 471, 472], [474, 475, 476, 477]


def feature_pieces(lab, lm, sil, skinL, mm, style):
    """Eyes, brows and mouth as explicit shapes sized from this face, coloured from the stylised
    image. Returns [(name, lab_colour, mask)] in gluing order."""
    shape = sil.shape
    out = []
    med = lambda m, fb: np.median(lab[m], 0) if m.sum() > 12 else np.float32(fb)
    anime = style != 'realistic'
    for side, (brow, ring, lid, ic, iring) in enumerate((
            (E.BROW_L, E.EYE_L, UPPER_LID_L, IRIS_L, IRIS_RING_L),
            (E.BROW_R, E.EYE_R, UPPER_LID_R, IRIS_R, IRIS_RING_R))):
        bm = E.fill_poly(shape, min_height_poly(lm[brow], 2.6 * mm))
        bl = med(E.fill_poly(shape, lm[brow]), (30, 8, 15))
        if bl[0] < skinL - 8:  # only if the brow is actually darker than the skin
            out.append(('Eyebrow', bl, bm))
        ring_m = E.fill_poly(shape, lm[ring])
        ringL = np.quantile(lab[ring_m][:, 0], .85) if ring_m.sum() > 12 else skinL
        if ringL < .62 * skinL:  # covered eye (patch / dark glasses): leave what the model drew
            continue
        eye = min_height_poly(lm[ring], (4.6 if anime else 3.6) * mm)
        if anime:
            c = eye.mean(0); eye = (eye - c) * [1.04, 1.15] + c
        eye_m = E.fill_poly(shape, eye)
        white = lab[ring_m]
        wl = np.median(white[white[:, 0] >= np.quantile(white[:, 0], .6)], 0) if ring_m.sum() > 12 else np.float32([90, 0, 0])
        wl = np.float32([max(90.0, skinL + 15), wl[1] * .25, wl[2] * .25])
        out.append(('Eye white', wl, eye_m))
        cen = lm[ic]
        rad = float(np.mean([np.hypot(*(lm[j] - cen)) for j in iring]))
        rad = max(rad * (1.1 if anime else 1.0), 1.8 * mm)
        rad = min(rad, .46 * np.ptp(eye[:, 1]))
        iris_m = ellipse_mask(shape, cen, rad, rad * (1.1 if anime else 1.0)) & cv2.dilate(eye_m.astype(np.uint8), E.disk(.5 * mm)).astype(bool)
        il = med(E.hull_mask(shape, lm[iring], 1.0), (30, 6, 18)); il[0] = min(il[0], wl[0] - 30)
        out.append(('Iris', il, iris_m))
        lid_pts = lm[lid].copy(); lid_pts[:, 1] -= (.8 if anime else .4) * mm
        ll = np.float32([max(8, il[0] - 12), il[1] * .5, il[2] * .5])
        out.append(('Lash line', ll, thick_polyline(shape, lid_pts, (2.4 if anime else 2.0) * mm)))
        if anime:
            cl = cen + np.array([-.35 * rad if side == 0 else .35 * rad, -.4 * rad])
            r_ = max(1.2 * mm, .28 * rad)
            out.append(('Catch-light', np.float32([98, 0, 0]), ellipse_mask(shape, cl, r_, r_)))
    lips_m = E.fill_poly(shape, min_height_poly(lm[E.LIPS], 3.0 * mm))
    inner_m = E.fill_poly(shape, lm[LIPS_INNER])
    out.append(('Lips', med(lips_m & ~inner_m, (55, 30, 20)), lips_m))
    inner = lm[LIPS_INNER]
    if np.ptp(inner[:, 1]) > 2.2 * mm and inner_m.sum() > 12:
        Li = lab[inner_m][:, 0]
        dark = inner_m & (lab[..., 0] < np.quantile(Li, .5))
        out.append(('Mouth', med(dark, (20, 15, 10)), inner_m))
        teeth = inner_m & (lab[..., 0] > max(np.quantile(Li, .55), skinL))
        teeth = cv2.morphologyEx(teeth.astype(np.uint8), cv2.MORPH_OPEN, E.disk(.7 * mm)).astype(bool)
        teeth = E.drop_small(teeth, 5 * mm * mm)
        if teeth.any():
            tl = med(teeth, (88, 0, 5)); tl[0] = max(tl[0], skinL + 15)
            out.append(('Teeth', tl, teeth))
    return out


# ----------------------------------------------------------------- main
SHEET_PX = (int(E.SHEET_MM[0] * PPM), int(E.SHEET_MM[1] * PPM))  # (w, h) = 760 x 1108


def make_collage(rgb, person, landmarks, style='ghibli', difficulty='easy', return_image=False):
    """Whole photo in, pieces out (crop + style model on the server). Used by the eval harness;
    the web app crops and runs the style model in the browser, then calls cut_styled()."""
    H, W = person.shape
    E.PPM = PPM
    lm_px = np.asarray(landmarks, float)[:, :2] * [W, H]
    rgb_c, per_c, lm, frame = E.crop_head_shoulders(rgb, person.astype(np.float32), lm_px)
    x0, y0, s = frame
    sty = stylize(rgb_c, style)
    res = cut_styled(sty, per_c > .5, lm / [rgb_c.shape[1], rgb_c.shape[0]], style, difficulty)
    res['frame'] = [x0, y0, s / PPM]
    if return_image:
        res['_styled'] = sty
    return res


def cut_styled(styled, person, landmarks, style='ghibli', difficulty='easy'):
    """styled: stylised head-and-shoulders crop (sheet aspect 190:277), HxWx3 uint8.
    person: HxW bool mask of the figure in that crop. landmarks: 478 [x, y] normalised to the crop."""
    assert style in STYLES
    P = DIFF[difficulty]
    E.PPM = PPM
    w, h = SHEET_PX
    sty = cv2.resize(styled, (w, h), interpolation=cv2.INTER_AREA if styled.shape[1] > w else cv2.INTER_CUBIC)
    per_c = cv2.resize(person.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lm = np.asarray(landmarks, float)[:, :2] * [w, h]
    shape = per_c.shape

    # ---- masks
    sil = cv2.GaussianBlur(per_c, (0, 0), 1.2 * PPM) > .5
    face_core = E.hull_mask(shape, lm[E.FACE_OVAL], .8)
    n, cl = cv2.connectedComponents(sil.astype(np.uint8), connectivity=8)
    ids = np.unique(cl[face_core & sil]); ids = ids[ids > 0]
    if len(ids):
        sil = np.isin(cl, ids)
    sil = E.fill_small_holes(sil, 10 ** 9)
    sil = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_OPEN, E.disk(P['min_w'] * PPM / 2)).astype(bool)
    face = E.hull_mask(shape, lm[E.FACE_OVAL], 1.05)
    feat = np.zeros(shape, bool)
    for idx, g in ((E.EYE_L, 1.9), (E.EYE_R, 1.9), (E.BROW_L, 1.35), (E.BROW_R, 1.35), (E.LIPS, 1.3), (E.NOSTRILS, 1.3)):
        feat |= E.hull_mask(shape, lm[idx], g)

    # ---- paper colours
    lab = lab_of(cv2.GaussianBlur(sty, (0, 0), 0.6 * PPM))
    rng = np.random.default_rng(0)
    fig = lab[sil]
    fac = lab[face & sil]
    fea = lab[feat & sil]
    pick = lambda a, n_: a[rng.choice(len(a), min(len(a), n_), replace=False)] if len(a) else a[:0]
    samples = np.vstack([pick(fig, 6000), pick(fac, 6000), pick(fea, 3000)])
    k = P['k'] - 1
    cen, _ = kmeans(samples, k)
    # merge papers closer than a just-noticeable step, then re-fit once
    def merge(cen, thr):
        cen = list(cen)
        done = False
        while not done and len(cen) > 2:
            done = True
            for i in range(len(cen)):
                for j in range(i + 1, len(cen)):
                    if np.linalg.norm(cen[i] - cen[j]) < thr:
                        cen[i] = (cen[i] + cen[j]) / 2; cen.pop(j); done = False; break
                if not done: break
        return np.array(cen, np.float32)
    cen = merge(cen, 7.0)
    if style == 'toon':  # the comic model is pale: give the papers back their colour
        cen[:, 1:] *= 1.5
        lab[..., 1:] *= 1.5
    # the features need a light paper (eye whites, teeth) and a dark one (iris, lashes)
    skinL = float(np.median(lab[face & sil][:, 0])) if (face & sil).any() else 60.0
    feats = feature_pieces(lab, lm, sil, skinL, PPM, style)
    for name, col, _ in feats:
        if not len(cen) or np.min(np.linalg.norm(cen - col, axis=1)) > (8 if name in ('Iris', 'Eye white') else 11):
            cen = np.vstack([cen, col[None]])
    # keep the shopping list short: merge the closest pair until we are within budget
    while len(cen) > P['k'] + 3:
        dm = np.linalg.norm(cen[:, None] - cen[None], axis=-1); np.fill_diagonal(dm, np.inf)
        i, j = np.unravel_index(np.argmin(dm), dm.shape)
        cen[i] = (cen[i] + cen[j]) / 2; cen = np.delete(cen, j, 0)
    k = len(cen)
    d = np.linalg.norm(lab[..., None, :] - cen[None, None], axis=-1)
    labels = np.argmin(d, -1).astype(np.int16)
    labels[~sil] = -1

    # background paper: median of the stylised background (or the sheet itself for realistic)
    bg_lab = np.median(lab[~sil], 0) if (~sil).any() else np.float32([92, 0, 0])
    bg_lab = np.float32([max(74.0, bg_lab[0]), bg_lab[1] * .45, bg_lab[2] * .45])
    if style == 'realistic':
        bg_lab = np.float32([94, -1, -2])

    # ---- cleanup in millimetres
    lbl = smooth_labels(np.where(sil, labels, k), k + 1, 1.1 * PPM)  # k = outside
    lbl[~sil] = k
    lbl = enforce_min_width(lbl, k + 1, P['min_w'] * PPM / 2)
    lbl[~sil] = k
    zone = np.zeros(shape, bool)
    for idx, g in ((E.EYE_L, 1.6), (E.EYE_R, 1.6), (E.LIPS, 1.15)):
        zone |= E.hull_mask(shape, lm[idx], g)
    ring = cv2.dilate(zone.astype(np.uint8), E.disk(2 * PPM)).astype(bool) & ~zone & sil
    if ring.any():
        lbl[zone & sil] = np.bincount(lbl[ring]).argmax()
    lbl = absorb_small(lbl, k + 1, P['min_a'] * PPM * PPM)
    lbl[~sil] = k
    lbl = lbl.astype(np.int16)

    # ---- gluing order: biggest paper first; papers that are mostly features last
    areas = np.array([(lbl == c).sum() for c in range(k)], float)
    featness = np.array([(feat & (lbl == c)).sum() / max(1, areas[c]) for c in range(k)])
    order = sorted(range(k), key=lambda c: (featness[c] > .5, -areas[c]))
    order = [c for c in order if areas[c] > 0]

    pieces = stack_pieces(lbl, order, P, feat)
    for name, col, m in feats:
        if m.any():
            pieces.append((int(np.argmin(np.linalg.norm(cen - col, axis=1))), m & cv2.dilate(sil.astype(np.uint8), E.disk(2 * PPM)).astype(bool)))

    # ---- output
    names = {}
    out = [{'d': 'M0,0 L%.1f,0 L%.1f,%.1f L0,%.1f Z' % (E.SHEET_MM[0], E.SHEET_MM[0], E.SHEET_MM[1], E.SHEET_MM[1]),
            'hex': hex_of(bg_lab), 'paper': 'Background', 'area': round(E.SHEET_MM[0] * E.SHEET_MM[1], 1),
            'cx': round(E.SHEET_MM[0] / 2, 1), 'cy': round(E.SHEET_MM[1] * .08, 1)}]
    for c, m in pieces:
        r = E.mask_to_path(m, PPM, smooth_px=0.5 * PPM, eps_px=0.2 * PPM)
        if r is None:
            continue
        dpath, cx, cy, _ = r
        hx = hex_of(cen[c])
        if hx not in names:
            L_ = cen[c][0]
            tone = 'dark' if L_ < 35 else 'light' if L_ > 72 else 'mid'
            names[hx] = f'Paper {len(names) + 1} ({tone})'
        out.append({'d': dpath, 'hex': hx, 'paper': names[hx], 'area': round(float(m.sum() / PPM ** 2), 1),
                    'cx': round(cx, 1), 'cy': round(cy, 1)})
    for i, p in enumerate(out):
        p['n'] = str(i + 1)
    pal = {}
    for p in out:
        e = pal.setdefault(p['hex'], {'hex': p['hex'], 'name': p['paper'], 'count': 0})
        e['count'] += 1
    return {'width': E.SHEET_MM[0], 'height': E.SHEET_MM[1], 'units': 'mm', 'style': style,
            'difficulty': difficulty, 'count': len(out), 'pieces': out, 'papers': list(pal.values())}
