"""Run the same MediaPipe models the app uses on each test photo and cache the results.

For every photo we store (at a max side of 1600 px):
  rgb.png        the photo
  cat.png        selfie_multiclass category mask (0 bg,1 hair,2 body skin,3 face skin,4 clothes,5 other)
  lm.json        478 face landmarks (normalised x,y) + blendshapes
"""
import glob, json, os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")

lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=f"{M}/face_landmarker.task"),
    num_faces=1, output_face_blendshapes=True))
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=f"{M}/selfie_multiclass_256x256.tflite"),
    output_category_mask=True))

for f in sorted(glob.glob(f"{HERE}/photos/*.jpg")):
    name = os.path.splitext(os.path.basename(f))[0]
    out = f"{HERE}/cache/{name}"
    os.makedirs(out, exist_ok=True)
    im = Image.open(f).convert("RGB")
    im.thumbnail((1600, 1600), Image.LANCZOS)
    a = np.ascontiguousarray(np.asarray(im))
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=a)
    r = lmk.detect(mpi)
    if not r.face_landmarks:
        print("NO FACE", name); continue
    lm = [[p.x, p.y] for p in r.face_landmarks[0]]
    bs = {c.category_name: c.score for c in r.face_blendshapes[0]} if r.face_blendshapes else {}
    cat = seg.segment(mpi).category_mask.numpy_view().copy()
    im.save(f"{out}/rgb.png")
    Image.fromarray(cat.astype(np.uint8)).save(f"{out}/cat.png")
    json.dump({"landmarks": lm, "blendshapes": bs}, open(f"{out}/lm.json", "w"))
    print(name, a.shape, cat.shape, np.bincount(cat.ravel(), minlength=6))
