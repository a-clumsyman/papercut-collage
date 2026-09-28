"""New papercut cutter: level image + landmarks -> simple smooth SVG pieces.
Pure numpy + PIL. Region tracing by crack-following, Chaikin smoothing, RDP simplify.
"""
import base64, io, json, math
import numpy as np
from PIL import Image

# ---------- geometry helpers ----------

def chaikin(pts, iters=2, closed=True):
    pts = [tuple(p) for p in pts]
    for _ in range(iters):
        if len(pts) < 3: break
        out = []
        n = len(pts)
        rng = range(n) if closed else range(n-1)
        if not closed: out.append(pts[0])
        for i in rng:
            a = pts[i]; b = pts[(i+1) % n]
            out.append((0.75*a[0]+0.25*b[0], 0.75*a[1]+0.25*b[1]))
            out.append((0.25*a[0]+0.75*b[0], 0.25*a[1]+0.75*b[1]))
        if not closed: out.append(pts[-1])
        pts = out
    return pts

def rdp(pts, eps, closed=True):
    if len(pts) < 4: return list(pts)
    arr = list(pts)
    if closed:
        # start at point farthest from centroid to reduce seam artifacts
        cx = sum(p[0] for p in arr)/len(arr); cy = sum(p[1] for p in arr)/len(arr)
        k = max(range(len(arr)), key=lambda i: (arr[i][0]-cx)**2 + (arr[i][1]-cy)**2)
        arr = arr[k:] + arr[:k]
    keep = [False]*len(arr); keep[0] = keep[-1] = True
    stack = [(0, len(arr)-1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0+1: continue
        ax, ay = arr[i0]; bx, by = arr[i1]
        dx, dy = bx-ax, by-ay
        L = math.hypot(dx, dy) or 1e-9
        dmax, imax = -1, -1
        for i in range(i0+1, i1):
            px, py = arr[i]
            d = abs(dy*px - dx*py + bx*ay - by*ax) / L
            if d > dmax: dmax, imax = d, i
        if dmax > eps:
            keep[imax] = True
            stack.append((i0, imax)); stack.append((imax, i1))
    return [arr[i] for i in range(len(arr)) if keep[i]]

def min_spacing(pts, minlen, closed=True):
    pts = list(pts)
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for i in range(n):
            a = pts[i]; b = pts[(i+1) % n]
            if math.hypot(b[0]-a[0], b[1]-a[1]) < minlen:
                mid = ((a[0]+b[0])/2, (a[1]+b[1])/2)
                if i == n - 1:
                    pts = [mid] + pts[1:n-1]
                else:
                    pts = pts[:i] + [mid] + pts[i+2:]
                changed = True
                break
    return pts

def poly_area(pts):
    a = 0.0
    for i in range(len(pts)):
        x0,y0 = pts[i]; x1,y1 = pts[(i+1)%len(pts)]
        a += x0*y1 - x1*y0
    return a/2

def path_d(loops):
    """loops: list of point lists. First should be outer (CW), holes CCW under evenodd any order."""
    parts = []
    for pts in loops:
        if len(pts) < 3: continue
        p = "M%.1f,%.1f " % pts[0] + " ".join("L%.1f,%.1f" % q for q in pts[1:]) + " Z"
        parts.append(p)
    return " ".join(parts)

# ---------- region tracing (crack following) ----------

def trace_loops(mask):
    """mask: 2D bool. Returns list of loops; outer loops signed area > 0 convention:
    we return (outer_loops, hole_loops) lists of (x,y) point lists (pixel-corner coords)."""
    h, w = mask.shape
    # directed edges: region interior on the RIGHT -> outer loops clockwise (in image coords y-down => signed area negative)
    edges = {}  # start vertex -> list of end vertices
    m = mask
    # edge key: (x, y, dir)
    es = []
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        if y == 0 or not m[y-1, x]:   es.append(((x, y),   (x+1, y)))     # top edge, dir +x
        if x == w-1 or not m[y, x+1]: es.append(((x+1, y), (x+1, y+1)))   # right edge, dir +y
        if y == h-1 or not m[y+1, x]: es.append(((x+1, y+1), (x, y+1)))   # bottom edge, dir -x
        if x == 0 or not m[y, x-1]:   es.append(((x, y+1), (x, y)))       # left edge, dir -y
    nxt = {}
    for a, b in es:
        nxt.setdefault(a, []).append(b)
    loops = []
    used = set()
    for a, b in es:
        if (a, b) in used: continue
        loop = [a]
        cur, curn = a, b
        used.add((a, b))
        while curn != a:
            loop.append(curn)
            cands = nxt.get(curn, [])
            # choose the edge that continues with the region on the right; prefer right-turn, straight, left-turn
            px, py = cur; cx2, cy2 = curn
            d = (cx2-px, cy2-py)
            order = [(d[1], -d[0]), d, (-d[1], d[0]), (-d[0], -d[1])]  # right, straight, left, back
            chosen = None
            for od in order:
                t = (cx2+od[0], cy2+od[1])
                if (curn, t) in [(c, e) for c, e in [(curn, c2) for c2 in cands]] and (curn, t) not in used:
                    chosen = t; break
            if chosen is None:
                for c2 in cands:
                    if (curn, c2) not in used:
                        chosen = c2; break
            if chosen is None: break
            used.add((curn, chosen))
            cur, curn = curn, chosen
        loops.append(loop)
    return loops

def components(mask):
    from scipy.ndimage import label
    lab, nc = label(mask)
    sizes = np.bincount(lab.ravel())
    comps = [int(sizes[i]) for i in range(1, nc+1)]
    return lab, comps

def region_loops(mask):
    """Trace all loops of a binary mask; group holes with their outer loop.
    Returns list of (outer, [holes])."""
    loops = trace_loops(mask)
    outers, holes = [], []
    for lp in loops:
        a = poly_area(lp)
        # with region-on-right edges: outer loops are CW in y-down coords => signed area > 0? test: y-down, CW => positive
        (outers if a > 0 else holes).append(lp)
    out = []
    holeinfo = [(hp, hp[0]) for hp in holes]
    for o in outers:
        hs = []
        for hp, (hx, hy) in holeinfo:
            if point_in_poly(hx, hy, o):
                hs.append(hp)
        out.append((o, hs))
    return out

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    j = n-1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

def smooth_loop(pts, scale=1.0, eps=2.0, minlen=9.0, iters=2):
    if scale != 1.0:
        cx = sum(p[0] for p in pts)/len(pts); cy = sum(p[1] for p in pts)/len(pts)
        pts = [((p[0]-cx)*scale+cx, (p[1]-cy)*scale+cy) for p in pts]
    pts = chaikin(pts, iters=iters)
    pts = rdp(pts, eps)
    pts = min_spacing(pts, minlen)
    return pts

def centroid(pts):
    a = poly_area(pts)
    if abs(a) < 1e-6:
        return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))
    cx = cy = 0.0
    for i in range(len(pts)):
        x0,y0 = pts[i]; x1,y1 = pts[(i+1)%len(pts)]
        f = x0*y1 - x1*y0
        cx += (x0+x1)*f; cy += (y0+y1)*f
    return (cx/(6*a), cy/(6*a))

def min_thickness(mask):
    """approx min thickness: 2 * min over boundary sample of erosion depth... cheap proxy:
    erode repeatedly; thickness profile. We compute 2*max iterations each pixel survives -> distance transform approx."""
    m = mask.copy()
    depth = np.zeros(m.shape, np.int32)
    it = 0
    while m.any():
        it += 1
        er = m.copy()
        er[1:] &= m[:-1]; er[:-1] &= m[1:]; er[:,1:] &= m[:,:-1]; er[:,:-1] &= m[:,1:]
        depth[er & (depth==0)] = it
        m = er
        if it > 400: break
    bd = mask & (depth>0)
    if not bd.any(): return 0.0
    # min thickness ~ 2 * min distance-to-edge among component pixels is 1... use max depth *2 as 'bulk'
    # better proxy: sample boundary pixels' distance inward is 1; instead report 2*median depth of interior
    interior = depth[depth>0]
    return float(2*np.percentile(interior, 20))


def convex_hull(pts):
    p = sorted(pts, key=lambda q: (q[0], q[1]))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0: lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0: upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]
