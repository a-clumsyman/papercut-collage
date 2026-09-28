# Papercut Collage

Take a selfie and get a step-by-step guide for turning it into a paper-collage portrait:
four shades of one paper colour, stacked lightest to darkest, with printable cutting templates.

## How it works

1. **Browser (`app.js`)**: MediaPipe finds the face mesh (478 landmarks) and a person mask. It uploads
   the photo (max 1600 px), the mask and the landmarks.
2. **Server (`api/engine.py`, via `api/process.py` on Vercel)**:
   - crops to head and shoulders so the face fills an A4 sheet
   - flattens the photo's real light and shadow into 4 paper tones, with thresholds tuned on the face
   - builds **stacked layers**: the whole silhouette in the lightest paper, then every darker area glued
     on top. Pieces overlap, so small cutting errors don't show
   - makes every piece cuttable in real millimetres: at least 4 mm wide (2.5 mm around the eyes and
     mouth), no crumbs under 1 cm² outside the features, and **no interior holes** (a lighter hole
     becomes its own piece on top)
   - finds the eyes, brows, nostrils and mouth with thresholds computed locally, and sizes each eye
     from the person's own mesh
   - tries 24 setting combinations and keeps the most photo-like result within the piece budget
     (15–30 pieces for `easy`, up to 60 for `medium`)
3. **Guide**: numbered pieces in gluing order, a highlight for each step, and downloadable A4 cutting
   templates grouped by paper shade.

Sheet coordinates in the API response are in **millimetres** (A4 printable area, 190 × 277).

## Develop

```
pip install -r requirements.txt
python tools/eval/prep.py     # needs mediapipe + models in tools/eval/models, photos in tools/eval/photos
python tools/eval/run.py      # likeness + cuttability scores for every test portrait
```

`tools/eval/pc_eval.py` scores any result for **likeness** (face-weighted structural similarity to
the photo, 0–100) and **cuttability** (piece count, pieces under 1 cm², thin parts under 4 mm, holes).
