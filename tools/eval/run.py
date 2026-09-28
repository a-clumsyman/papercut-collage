"""Score the engine on a folder of prepared portraits (see prep.py).

    python tools/eval/prep.py            # photos/*.jpg -> cache/<name>/{rgb.png,cat.png,lm.json}
    python tools/eval/run.py [easy|medium]
"""
import json, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from api.engine import make_collage  # noqa: E402
import pc_eval as E  # noqa: E402

diff = sys.argv[1] if len(sys.argv) > 1 else 'easy'
rows = {}
for n in sorted(os.listdir(f'{HERE}/cache')):
    c = f'{HERE}/cache/{n}'
    rgb = np.asarray(Image.open(f'{c}/rgb.png').convert('RGB'))
    cat = np.asarray(Image.open(f'{c}/cat.png'))
    lm = json.load(open(f'{c}/lm.json'))['landmarks']
    r = make_collage(rgb, cat > 0, lm, diff)
    rows[n] = {**E.likeness(r, r['frame'], c), **E.cuttability(r)}
    Image.fromarray(E.render(r, scale=4)).save(f'{c}/render_{diff}.png')
    print(n, rows[n]['likeness'], rows[n]['pieces'])
keys = ['likeness', 'feature_ssim', 'pieces', 'tiny_pieces(<1cm2)', 'fiddly_pieces', 'holes']
print({k: round(float(np.mean([v[k] for v in rows.values()])), 2) for k in keys})
