"""Score a style on the prepared portraits (see prep.py).

    python tools/eval/prep.py                 # photos/*.jpg -> cache/<name>/{rgb.png,cat.png,lm.json}
    python tools/eval/run.py [style] [easy|medium]   # style: realistic | ghibli | toon
Needs onnxruntime (the style model runs on the server here; in the app it runs in the browser).
"""
import json, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from api.collage import make_collage  # noqa: E402
import pc_eval as E  # noqa: E402

style = sys.argv[1] if len(sys.argv) > 1 else 'ghibli'
diff = sys.argv[2] if len(sys.argv) > 2 else 'easy'
rows = {}
for n in sorted(os.listdir(f'{HERE}/cache')):
    c = f'{HERE}/cache/{n}'
    rgb = np.asarray(Image.open(f'{c}/rgb.png').convert('RGB'))
    cat = np.asarray(Image.open(f'{c}/cat.png'))
    lm = json.load(open(f'{c}/lm.json'))['landmarks']
    r = make_collage(rgb, cat > 0, lm, style, diff, return_image=True)
    Image.fromarray(r.pop('_styled')).save(f'{c}/{style}_styled.png')
    Image.fromarray(E.render(r, colors=[p['hex'] for p in r['pieces']], scale=4)).save(f'{c}/{style}_{diff}.png')
    rows[n] = {**E.cuttability(r), 'papers': len(r['papers'])}
    print(n, rows[n])
keys = ['pieces', 'tiny_pieces(<1cm2)', 'fiddly_pieces', 'holes', 'papers']
print({k: round(float(np.mean([v[k] for v in rows.values()])), 2) for k in keys})
