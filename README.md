# Papercut Collage

Take a selfie or upload a photo and get a paper-collage portrait in one of three styles —
**Realistic**, **Ghibli-style** or **Toon** — with numbered pieces, A4 cutting templates grouped by
paper colour, and a step-by-step gluing guide.

## How it works

Everything that looks at the photo runs **in the browser**; only the stylised picture goes to the server.

1. **Browser (`app.js`)**
   - MediaPipe finds the face mesh (478 landmarks) and a person mask.
   - The photo is cropped to head and shoulders at the A4 sheet aspect (face ≈ 90 mm wide).
   - A free neural style model runs on the crop with onnxruntime-web (WebGPU when available, WASM otherwise;
     7–9 MB per model, cached after the first use, 2–15 s depending on the device):
     | Style | Model | Licence |
     |---|---|---|
     | Realistic | animegan2-pytorch `face_paint_512_v2` | MIT |
     | Ghibli-style | AnimeGANv3 `Ghibli_c1` (portrait model) | free for non-commercial use |
     | Toon | animegan2-pytorch `celeba_distill` | MIT |
2. **Server (`api/process.py` → `api/collage.py`, OpenCV on Vercel)**
   - picks ~9–12 paper colours (k-means in Lab, the face sampled more densely; a true white and a dark
     paper are reserved for the eyes)
   - builds the eyes, brows, lips, mouth and teeth as explicit shapes sized from this face, coloured from
     the stylised image, so the features never get lost in the quantisation
   - turns the colour map into **stacked** pieces: big papers first, details on top; each piece extends
     1.5 mm under the pieces above it (no gaps), crumbs are merged into their neighbours, covered holes are
     filled and uncovered holes become their own piece — so every piece is a solid shape at least 4 mm wide
     (2.5 mm around the eyes) with nothing to cut out of the middle
3. **Guide**: numbered pieces in gluing order, the paper swatch list (hex colours to match), downloadable
   A4 templates per paper colour, and the styled picture itself.

Sheet coordinates in the API response are millimetres on the A4 printable area (190 × 277).

## Files

```
index.html, main.css, app.js   the web app
models/*.onnx                  the three style models (served as static files)
api/process.py                 Vercel function: validation + JSON in/out
api/collage.py                 paper colours, feature pieces, stacking, cuttability rules
api/engine.py                  shared geometry (crop, masks, contour -> SVG path)
api/stylize.py                 runs the same ONNX models on the server (eval only; not used by the function)
tools/eval/                    scoring harness: prep.py (MediaPipe on test photos), run.py, pc_eval.py
```

## Develop

```
pip install -r requirements.txt            # what the Vercel function needs
pip install onnxruntime mediapipe          # extra, for the eval harness only
python tools/eval/prep.py                  # photos in tools/eval/photos -> tools/eval/cache
python tools/eval/run.py ghibli easy       # pieces, tiny pieces, papers for every test portrait
```

`vercel.json` sets COOP/COEP headers so onnxruntime-web can use threads, and long caching for `/models`.
The function stays under Vercel's 250 MB limit because onnxruntime is **not** installed there.

## Licences

The app code is yours. `models/AnimeGANv3_large_Ghibli_c1_e299.onnx` is from
[TachibanaYoshino/AnimeGANv3](https://github.com/TachibanaYoshino/AnimeGANv3) and is free for
non-commercial use only; ask the author before using it commercially. The two animegan2 models are
exported from [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch) (MIT).
