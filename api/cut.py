"""Server-side piece cutting (v2). Input: levels image, landmarks, meta. Output: pieces JSON."""
import base64, io, json, math
import numpy as np
from PIL import Image
from . import cutter as C


EYE_LI = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
EYE_RI = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LIPS_I = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
BROW_LI = [70,63,105,66,107,55,65,52,53,46]
BROW_RI = [336,296,334,293,300,276,283,282,295,285]
NOSE_PTS = [6,197,195,5,4,1,2,98,327,19,94]
IRIS_L, IRIS_R = 468, 473
IRIS_LR = [469,470,471,472]
IRIS_RR = [474,475,476,477]

def decode_png(b64):
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
    if a.ndim == 3:
        a = a[..., 0]
    return a

def gray_hex(L):
    v = int(round(L))
    return '#%02x%02x%02x' % (v, v, v)

def landmark_pts(landmarks, w, h):
    return [(p[0]*w, p[1]*h) for p in landmarks]

def mask_of_poly(w, h, pts):
    m = np.zeros((h, w), bool)
    ys = [p[1] for p in pts]
    y0 = max(0, int(min(ys))-2); y1 = min(h-1, int(max(ys))+2)
    for y in range(y0, y1+1):
        xs = []
        n = len(pts)
        for i in range(n):
            a = pts[i]; b = pts[(i+1) % n]
            if (a[1] <= y+0.5) != (b[1] <= y+0.5):
                xs.append(a[0] + (y+0.5-a[1])*(b[0]-a[0])/(b[1]-a[1]))
        xs.sort()
        for k in range(0, len(xs)-1, 2):
            xa = max(0, int(round(xs[k]))); xb = min(w-1, int(round(xs[k+1])))
            m[y, xa:xb+1] = True
    return m

def largest_component(mask):
    lab, comps = C.components(mask)
    if not comps: return None
    k = int(np.argmax(comps)) + 1
    return lab == k, comps[k-1]

def region_piece(mask, L, eps, minlen, iters=2, scale_grid=1.0):
    """Trace mask -> list of (d, area, cx, cy) using biggest outer loop + its holes."""
    got = largest_component(mask)
    if got is None: return None
    comp, area_px = got
    loops = C.region_loops(comp)
    if not loops: return None
    loops.sort(key=lambda oh: abs(C.poly_area(oh[0])), reverse=True)
    o, holes = loops[0]
    o2 = C.smooth_loop([(float(x), float(y)) for x, y in o], eps=eps, minlen=minlen, iters=iters)
    hs = [C.smooth_loop([(float(x), float(y)) for x, y in hp], eps=eps, minlen=minlen, iters=iters) for hp in holes]
    hs = [hh for hh in hs if len(hh) >= 3 and abs(C.poly_area(hh)) > 400]
    d = C.path_d([o2] + hs)
    cx, cy = C.centroid(o2)
    return d, abs(C.poly_area(o2)), cx, cy

def poly_piece(pts, L, scale=1.0, eps=1.5, minlen=6.0, iters=1):
    p = C.smooth_loop(pts, scale=scale, eps=eps, minlen=minlen, iters=iters)
    if len(p) < 3: return None
    d = C.path_d([p])
    cx, cy = C.centroid(p)
    return d, abs(C.poly_area(p)), cx, cy

def cut(payload):
    levels = decode_png(payload['photo'])
    landmarks = payload['landmarks']
    meta = payload.get('meta', {})
    h, w = levels.shape
    pts = landmark_pts(landmarks, w, h)
    sc = w / 900.0  # smoothing parameters scale with image size

    # Raw photo pixels (optional): drive the beard and nose from what the
    # camera actually saw, not from landmark geometry.
    lum = chroma = None
    if payload.get('pixels'):
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload['pixels'])))
                         .convert('RGB').resize((w, h), Image.LANCZOS)).astype(float)
        lum = .299*rgb[..., 0] + .587*rgb[..., 1] + .114*rgb[..., 2]
        chroma = rgb.max(axis=2) - rgb.min(axis=2)

    LV_FACE = meta.get('face', 165)
    LV_SHADE = meta.get('shade', 100)
    LV_HAIR = meta.get('hair', 35)
    LV_BEARD = meta.get('beard', 55)
    LV_BROW = meta.get('brow', 35)
    LV_LIPS = meta.get('lips', meta.get('shade', 100))
    LV_CLOTH = meta.get('cloth', None)
    LV_IRIS = 35
    LV_WHITE = 235

    # Display tones: four paper shades per his steer - dark (hair/brows/iris),
    # shadow (beard/lips/nose/cheek shade/neck), face, light (clothes/eye white).
    TONE_DARK = 35
    TONE_SHADOW = 90
    TONE_FACE = 150
    TONE_LIGHT = 215
    TONE_WHITE = 235

    pieces = []
    def add(L, d, area, cx, cy):
        pieces.append({'L': float(L), 'd': d, 'area': float(area), 'cx': float(cx), 'cy': float(cy)})

    # --- big regions, sticking order bottom-up ---
    from scipy.ndimage import binary_fill_holes, binary_dilation
    personM = levels != 245

    # Jaw boundary from the face mesh: U-curve temple -> chin -> temple. The
    # beard and face stop here; below is neck. No more wedge past the chin.
    JAW = (234,93,132,58,172,136,150,149,176,148,152,377,400,379,378,397,365,361,323,454)
    jaw = sorted((pts[i] for i in JAW), key=lambda p: p[0])
    jx = np.array([p[0] for p in jaw]); jy = np.array([p[1] for p in jaw])
    fw = max(1.0, abs(pts[454][0]-pts[234][0]))
    yy, xx = np.mgrid[0:h, 0:w]
    jawY = np.interp(xx, jx, jy)
    belowJaw = yy > jawY + .05*fw

    # Collar line: a shallow curve a short neck-length below the chin, rising
    # toward the sides. The shirt owns everything below it so the neck is a
    # band, never a pointed wedge down the chest.
    chinX, chinY = pts[152]
    dxn = (xx-chinX)/fw
    collarY = chinY + .30*fw + .5*fw*dxn*dxn
    clothM = (levels == LV_CLOTH) if LV_CLOTH is not None else np.zeros((h, w), bool)
    clothM |= personM & (yy > collarY)

    # clothes
    if clothM.sum() > 3000*sc*sc:
        r = region_piece(clothM, TONE_LIGHT, eps=5*sc, minlen=16*sc)
        if r: add(TONE_LIGHT, *r)
    # neck: whatever the person silhouette has below the jaw, minus clothes.
    # Skin tone so it reads as a bust, not a continuation of the beard.
    nm = ((levels == LV_SHADE) | (personM & belowJaw)) & ~clothM
    if nm.sum() > 1500*sc*sc:
        r = region_piece(nm, TONE_FACE, eps=3.5*sc, minlen=12*sc)
        if r: add(TONE_FACE, *r)
    # face (internal holes filled), clipped at the jaw line
    fm = (levels == LV_FACE) | ((levels == LV_BEARD) if LV_BEARD is not None else False)
    fm = fm & ~belowJaw
    if fm.sum() < 2000*sc*sc:
        return {'error': 'bad segmentation'}
    fm_filled = binary_fill_holes(fm)
    r = region_piece(fm_filled, TONE_FACE, eps=3*sc, minlen=10*sc)
    face_c = None
    if r: add(TONE_FACE, *r); face_c = (r[2], r[3])
    # One raised hand overlays the face and neck, never a neck-shaped sliver.
    LV_HAND = meta.get('hand')
    if LV_HAND is not None:
        hand_mask = levels == LV_HAND
        if hand_mask.sum() > max(1800, .025*(abs(pts[454][0]-pts[234][0])**2)):
            r = region_piece(hand_mask, TONE_FACE, eps=4*sc, minlen=14*sc)
            if r: add(TONE_FACE, *r)
    # beard / mustache, split at the lip line, both clipped at the jaw.
    # Photo-driven: Otsu-split the luminance inside the face-mesh jaw band so
    # the piece is HIS beard - real coverage, gaps and edges - not a jaw fill.
    bm = None
    if lum is not None and LV_BEARD is not None:
        noseBaseY = pts[2][1]
        skin = fm & (yy <= noseBaseY)
        medL = float(np.median(lum[skin])) if skin.sum() > 200 else 128.0
        jawreg = fm & (yy > noseBaseY) & (yy < pts[152][1] + .35*fw)
        if jawreg.sum() > 1500*sc*sc:
            vals = lum[jawreg]
            hist, _ = np.histogram(vals, bins=64, range=(0, 256))
            p = hist / max(1, hist.sum()); omega = np.cumsum(p)
            mu = np.cumsum(p*np.arange(64)); mt = mu[-1]
            sig = (mt*omega-mu)**2/(omega*(1-omega)+1e-9)
            t = (float(np.nanargmax(sig))+.5)*4
            from scipy.ndimage import binary_closing, label as _label
            m = jawreg & (lum < t) & (chroma < .85*np.median(chroma[skin])/.6)
            m = binary_closing(m, iterations=max(1, int(round(1.5*sc))))
            lab, n = _label(m)
            if n:
                keep = np.zeros_like(m)
                sizes = np.bincount(lab.ravel()); sizes[0] = 0
                for k in range(1, n+1):
                    if sizes[k] >= 250*sc*sc: keep |= lab == k
                m = keep
            # Beard zone: below the mouth-corners line, or hugging the jaw
            # curve (chin straps). Mid-cheek shadow is never beard.
            mouthY = np.interp(xx, [pts[61][0], pts[291][0]], [pts[61][1], pts[291][1]])
            m &= (yy > mouthY) | (np.abs(yy - jawY) < .18*fw)
            # Separation evidence. A luminance split alone cannot tell a dark
            # beard from smooth cheek skin in shadow - require a real class
            # gap, an absolute darkness anchor vs skin, and hair texture
            # (local luminance std). Weak evidence: no beard piece, never a
            # full-jaw blob.
            if m.sum() > 250*sc*sc:
                from scipy.ndimage import uniform_filter as _uf2
                muB = _uf2(lum, size=max(3, int(round(5*sc))))
                mu2B = _uf2(lum*lum, size=max(3, int(round(5*sc))))
                stdB = np.sqrt(np.maximum(0.0, mu2B-muB*muB))
                medDark = float(np.median(lum[m]))
                medLight = float(np.median(lum[jawreg & ~m])) if (jawreg & ~m).sum() > 200 else medL
                texFrac = float(np.mean(stdB[m] >= 4.0))
                separated = (medLight-medDark >= max(10.0, .08*medL)) and (medDark <= .72*medL) and (texFrac >= .05)
            else:
                separated = False
            if not separated:
                m = np.zeros_like(m)
            bm = m & ~belowJaw
    if bm is None and lum is None:
        bm = ((levels == LV_BEARD) if LV_BEARD is not None else np.zeros_like(fm)) & ~belowJaw
    if bm is None:
        bm = np.zeros_like(fm)
    if LV_BEARD is not None and bm.sum() > 900*sc*sc:
        lipTopY = min(pts[i][1] for i in LIPS_I)
        must = bm.copy(); must[int(lipTopY):,:] = False
        beard = bm.copy(); beard[:int(lipTopY),:] = False
        if beard.sum() > 900*sc*sc:
            r = region_piece(beard, TONE_SHADOW, eps=3*sc, minlen=10*sc)
            if r: add(TONE_SHADOW, *r)
        if must.sum() > 700*sc*sc:
            r = region_piece(must, TONE_SHADOW, eps=2.5*sc, minlen=8*sc)
            if r: add(TONE_SHADOW, *r)
    # hair: dilated a few px so the hairline overlaps the forehead (no white
    # crack), and traced with less smoothing so the real silhouette survives.
    hseed = levels == LV_HAIR
    if lum is not None and hseed.sum() > 4000*sc*sc:
        hL = float(np.median(lum[hseed])); hC = float(np.median(chroma[hseed]))
        from scipy.ndimage import label as _lab2, binary_closing as _bc2
        # local texture gate: real hair has strand-level luminance variance;
        # smooth backgrounds (studio walls, door frames) that merely match
        # hair color do not. Near-seed pixels pass unconditionally so the
        # hairline crack still fills.
        from scipy.ndimage import uniform_filter as _uf
        mu = _uf(lum, size=max(3, int(round(7*sc))))
        mu2 = _uf(lum*lum, size=max(3, int(round(7*sc))))
        locStd = np.sqrt(np.maximum(0.0, mu2-mu*mu))
        from scipy.ndimage import binary_dilation as _d5
        near = _d5(hseed, iterations=max(1, int(round(6*sc))))
        cand = (~binary_fill_holes(fm)) & (np.abs(lum-hL) < max(16.0, .25*hL+8)) & (np.abs(chroma-hC) < 16.0) & ((locStd > 5.0) | near)
        lab2, n2 = _lab2(cand)
        if n2:
            touching = np.unique(lab2[hseed]); touching = touching[touching > 0]
            if len(touching):
                grown = hseed | np.isin(lab2, touching)
                grown = _bc2(grown, iterations=max(1, int(round(2*sc))))
                # never let growth eat far from the seed region
                from scipy.ndimage import binary_dilation as _d4
                halo = _d4(hseed, iterations=max(2, int(round(28*sc))))
                hseed = grown & halo
    hm = binary_dilation(hseed, iterations=max(1, int(round(4*sc))))
    # A tiny seed is a buzz cut or shave, not a hairstyle: the segmenter's
    # temple islands dilate into blobs that read as artifacts. Clean bald is
    # closer to the truth.
    if hseed.sum() > 4000*sc*sc and hm.sum() > 2000*sc*sc:
        r = region_piece(hm, TONE_DARK, eps=2.5*sc, minlen=10*sc)
        if r: add(TONE_DARK, *r)

    # --- features (overlay, no holes needed) ---
    # Nose from the photo's own shading: the darkest structure inside the
    # nose region (nostrils, alar shadow, tip underside) is the cutout.
    side = meta.get('shade_side', 'right')
    def blend(a,b,t): return (a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t)
    nose_done = False
    if lum is not None:
        from scipy.ndimage import binary_dilation as _dil
        noseHull = C.convex_hull([pts[i] for i in [6,197,195,5,4,19,1,2,94,98,327,294,331,49,279,64,240,20,344,360,460]])
        nm = mask_of_poly(w, h, noseHull) & fm
        noseClipY = pts[2][1] + .06*fw
        nm[int(noseClipY):, :] = False
        if nm.sum() > 800*sc*sc:
            nmed = float(np.median(lum[nm]))
            m = nm & (lum < .84*nmed)
            got = largest_component(m)
            if got is not None and 300*sc*sc < got[1] < 8000*sc*sc:
                comp = _dil(got[0], iterations=max(1, int(round(1.5*sc)))) & fm
                r = region_piece(comp, TONE_SHADOW, eps=2.0*sc, minlen=7*sc)
                if r:
                    add(TONE_SHADOW, *r)
                    nose_done = True
    if not nose_done:
        # Landmark fallback: shaded bridge flows into the near-side alar wing.
        if side == 'left':
            upper, bridge, shoulder, wing, rim, nostril = (47,174,51,115,64,97)
        else:
            upper, bridge, shoulder, wing, rim, nostril = (277,399,281,344,294,326)
        plane = [blend(pts[197],pts[bridge],.30),
                 blend(pts[195],pts[shoulder],.38),
                 blend(pts[5],pts[shoulder],.45),
                 pts[wing],pts[rim],pts[nostril],pts[2],
                 pts[4],pts[5],pts[195]]
        r = poly_piece(plane, TONE_SHADOW, scale=1.0, eps=2.0*sc, minlen=9*sc, iters=1)
        if r: add(TONE_SHADOW, *r)
    # Soft cheek shadow on the measured shade side: one broad plane under the
    # cheekbone, so the face stops reading as a flat mask.
    if side == 'left':
        c_out, c_mid, c_jaw = pts[123], pts[205], pts[176]
    else:
        c_out, c_mid, c_jaw = pts[352], pts[425], pts[377]
    cheek_plane = [blend(c_out, c_mid, .25), blend(c_mid, c_out, .15),
                   blend(c_mid, c_jaw, .55), blend(c_jaw, c_out, .45)]
    cheek_done = False
    if lum is not None:
        # A cheek shade only exists if the photo actually shades the cheek:
        # darkest structure inside the cheek ROI, and it must reach the face
        # edge. A disconnected island mid-cheek reads as a scar, not shading.
        cheekROI = mask_of_poly(w, h, [(p[0]*1.6, p[1]*1.15) for p in cheek_plane]) & fm
        skinL = float(np.median(lum[fm])) if fm.sum() > 200 else 128.0
        if cheekROI.sum() > 400*sc*sc:
            cm = cheekROI & (lum < .82*skinL)
            got = largest_component(cm)
            if got is not None and got[1] > 350*sc*sc:
                comp, _ = got
                edge = fm & ~binary_fill_holes(fm) if False else None
                from scipy.ndimage import binary_dilation as _d2, binary_erosion as _e2
                faceEdge = fm & ~_e2(fm, iterations=max(2, int(round(4*sc))))
                if (_d2(comp, iterations=max(1, int(round(2*sc)))) & faceEdge).sum() > 20*sc*sc:
                    comp = _d2(comp, iterations=max(1, int(round(1.5*sc)))) & fm
                    r = region_piece(comp, TONE_SHADOW, eps=2.0*sc, minlen=7*sc)
                    if r:
                        add(TONE_SHADOW, *r)
                        cheek_done = True
    if not cheek_done and lum is None:
        r = poly_piece(cheek_plane, TONE_SHADOW, scale=1.0, eps=1.6*sc, minlen=8*sc, iters=1)
        if r: add(TONE_SHADOW, *r)
    # lips: expression-aware. jawOpen (blendshape from the client, landmark
    # proxy fallback) decides between a single lip piece (closed) and a real
    # open mouth: dark interior, bright teeth from the photo, lip ring on top.
    LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
    bs = payload.get('blendshapes') or {}
    if 'jawOpen' in bs:
        jaw_open = float(bs['jawOpen'])
    else:
        upperIn = np.mean([pts[i][1] for i in [13, 14]])
        lowerIn = np.mean([pts[i][1] for i in [78, 308]])
        jaw_open = max(0.0, (lowerIn-upperIn)/fw) * 1.35
    lips = [pts[i] for i in LIPS_I]
    if jaw_open > .07 and lum is not None:
        innerM = mask_of_poly(w, h, [pts[i] for i in LIPS_INNER]) & fm
        if innerM.sum() > 300*sc*sc:
            skinL2 = float(np.median(lum[fm])) if fm.sum() > 200 else 128.0
            teethM = innerM & (lum > 1.10*skinL2)
            darkM = innerM & ~binary_dilation(teethM, iterations=max(1, int(round(1.5*sc))))
            gotd = largest_component(darkM)
            if gotd is not None and gotd[1] > 250*sc*sc:
                r = region_piece(gotd[0], TONE_DARK, eps=1.5*sc, minlen=5*sc)
                if r: add(TONE_DARK, *r)
            gott = largest_component(teethM)
            if gott is not None and gott[1] > 200*sc*sc:
                r = region_piece(gott[0], TONE_WHITE, eps=1.5*sc, minlen=5*sc)
                if r: add(TONE_WHITE, *r)
            # lip ring: outer lip contour minus the (dilated) inner opening
            from scipy.ndimage import binary_dilation as _d3
            outerM = mask_of_poly(w, h, [(cx0+(p[0]-cx0)*1.12, cy0+(p[1]-cy0)*1.12)
                                         for p in lips
                                         for cx0, cy0 in [(sum(q[0] for q in lips)/len(lips),
                                                           sum(q[1] for q in lips)/len(lips))]])
            ringM = outerM & ~_d3(innerM, iterations=max(1, int(round(1.5*sc)))) & fm
            got = largest_component(ringM)
            if got is not None and got[1] > 200*sc*sc:
                r = region_piece(got[0], TONE_SHADOW, eps=1.5*sc, minlen=5*sc)
                if r: add(TONE_SHADOW, *r)
        else:
            r = poly_piece(lips, TONE_SHADOW, scale=1.10, eps=1.5*sc, minlen=6*sc, iters=1)
            if r: add(TONE_SHADOW, *r)
    else:
        r = poly_piece(lips, TONE_SHADOW, scale=1.12, eps=1.5*sc, minlen=6*sc, iters=1)
        if r: add(TONE_SHADOW, *r)
    # brows
    for idxs in (BROW_LI, BROW_RI):
        brow = [pts[i] for i in idxs]
        r = poly_piece(brow, TONE_DARK, scale=1.14, eps=1.2*sc, minlen=5*sc, iters=1)
        if r: add(TONE_DARK, *r)
    # eyes: white + iris. Aperture measured from the ring itself (the mesh
    # already encodes how open HIS eyes are): heavy lids stay heavy, wide
    # eyes stay wide. A static squash is what made every face the same.
    for eye_i, (ring, ic, iring) in enumerate(((EYE_LI, IRIS_L, IRIS_LR), (EYE_RI, IRIS_R, IRIS_RR))):
        eye = [pts[i] for i in ring]
        ecx = sum(p[0] for p in eye)/len(eye); ecy = sum(p[1] for p in eye)/len(eye)
        xs = [p[0] for p in eye]; ys = [p[1] for p in eye]
        aspect = max(.05, (max(ys)-min(ys)) / max(1e-6, max(xs)-min(xs)))
        sqy = min(1.0, max(.55, aspect/.33))
        eye = [(ecx+(p[0]-ecx)*1.0, ecy+(p[1]-ecy)*sqy) for p in eye]
        r = poly_piece(eye, TONE_WHITE, scale=1.03, eps=0.7*sc, minlen=3.5*sc, iters=1)
        if r: add(TONE_WHITE, *r)
        cen = pts[ic]
        rad = float(np.mean([math.hypot(pts[j][0]-cen[0], pts[j][1]-cen[1]) for j in iring]))
        # Closest point on the eye contour segments, not the vertices: an iris
        # cannot spill outside a narrow eye even when the landmarks are sparse.
        contour = [(cen[0]+(p[0]-cen[0])*1.03, cen[1]+(p[1]-cen[1])*1.03) for p in eye]
        def segdist(a, b):
            vx,vy = b[0]-a[0], b[1]-a[1]
            t = max(0., min(1., ((cen[0]-a[0])*vx+(cen[1]-a[1])*vy)/max(1e-6,vx*vx+vy*vy)))
            return math.hypot(cen[0]-a[0]-t*vx, cen[1]-a[1]-t*vy)
        clearance = min(segdist(contour[k], contour[(k+1)%len(contour)]) for k in range(len(contour)))
        rr = min(rad*1.05, clearance*0.78)
        rr = max(rr, 1.3*sc)
        circ = [(cen[0]+rr*math.cos(t), cen[1]+rr*math.sin(t)) for t in np.linspace(0, 2*np.pi, 12, endpoint=False)]
        r = poly_piece(circ, TONE_DARK, scale=1.0, eps=0.8*sc, minlen=3*sc, iters=0)
        if r: add(TONE_DARK, *r)

    for i, p in enumerate(pieces):
        p['n'] = str(i+1)
        p['hex'] = gray_hex(p['L'])
    return {'width': w, 'height': h, 'count': len(pieces), 'pieces': pieces}
