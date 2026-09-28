import { FaceLandmarker, ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const show = (id) => {
  for (const s of ["step-pick","step-work","step-result","step-guide","step-err"]) $(s).hidden = s !== id;
};

let landmarker = null, segmenter = null;
async function loadModels(msg) {
  if (landmarker && segmenter) return;
  msg("Loading the cut-out models (one-time download)…");
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
    numFaces: 1, runningMode: "IMAGE" });
  segmenter = await ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" },
    outputCategoryMask: true, runningMode: "IMAGE" });
}

let result = null, guideIdx = 0;

// Monochrome paper ramps (dark -> light). Piece `tone` 0 = lightest paper ... 3 = darkest.
const RAMPS = {
  green:  ['#1e5c38', '#3d8a4e', '#79c47c', '#c9ecc4'],
  blue:   ['#1e3a7a', '#2f6fc4', '#7db4e8', '#cfe7f7'],
  pink:   ['#9c2b4f', '#e0496e', '#f588a5', '#fbd0da'],
  yellow: ['#c98a12', '#eab316', '#f7d442', '#fcefa8'],
};
const SHADE_NAMES = ['lightest', 'light-medium', 'medium-dark', 'darkest'];
let rampName = 'green';
const pieceColor = (p) => RAMPS[rampName][3 - p.tone];

function sheetSVG({ upto = null, numbers = false } = {}) {
  const { width: w, height: h, pieces } = result;
  const fs = (w * 0.028).toFixed(1), sw = (w * 0.004).toFixed(2);
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}mm" height="${h}mm">`;
  s += `<rect width="${w}" height="${h}" fill="#eaf2f5"/>`;
  const label = (p) => `<text x="${p.cx}" y="${p.cy}" dy="0.35em" font-family="system-ui,sans-serif" font-size="${fs}" font-weight="700" text-anchor="middle" fill="#e0432e" stroke="#fff" stroke-width="${sw * 2}" paint-order="stroke">${p.n}</text>`;
  pieces.forEach((p, i) => {
    if (upto === null || i < upto) {
      s += `<path d="${p.d}" fill="${pieceColor(p)}"/>`;
    } else if (i === upto) {
      s += `<path d="${p.d}" fill="${pieceColor(p)}" stroke="#e0432e" stroke-width="${sw}"/>`;
    } else {
      s += `<path d="${p.d}" fill="#ffffff" opacity="0.35"/>`;
    }
  });
  if (numbers) pieces.forEach((p) => { s += label(p); });
  if (upto !== null && pieces[upto]) s += label(pieces[upto]);
  return s + "</svg>";
}

// ---- cutting templates: every piece laid out on its own, grouped by paper shade ----
function bbox(d) {
  const v = d.match(/-?\d+(\.\d+)?/g).map(Number);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i + 1 < v.length; i += 2) {
    x0 = Math.min(x0, v[i]); x1 = Math.max(x1, v[i]); y0 = Math.min(y0, v[i+1]); y1 = Math.max(y1, v[i+1]);
  }
  return { x0, y0, w: x1 - x0, h: y1 - y0 };
}

function templatesSVG() {
  // A4 pages (210 x 297 mm, 10 mm margin). Page 1: finished picture with numbers.
  // Then one or more pages per shade with each piece separated so it can be traced / cut out.
  const PW = 210, PH = 297, M = 10, GAP = 6;
  const pages = [];
  pages.push(`<g transform="translate(${M},${M}) scale(${(PW - 2*M) / result.width})">${sheetSVG({ numbers: true }).replace(/^<svg[^>]*>|<\/svg>$/g, "")}</g>`
    + `<text x="${M}" y="${PH - 4}" font-size="4" font-family="system-ui">Finished picture. Glue pieces in number order: lightest paper first, darkest last.</text>`);
  for (let tone = 0; tone < 4; tone++) {
    const items = result.pieces.filter(p => p.tone === tone).map(p => ({ p, b: bbox(p.d) }))
      .sort((a, b) => b.b.h - a.b.h);
    if (!items.length) continue;
    let page = [], x = M, y = M + 10, rowH = 0;
    const flush = () => {
      if (!page.length) return;
      pages.push(`<rect x="${M}" y="${M}" width="${PW - 2*M}" height="7" fill="${RAMPS[rampName][3 - tone]}"/>`
        + `<text x="${M + 2}" y="${M + 5}" font-size="4.5" font-family="system-ui" font-weight="700" fill="${tone >= 2 ? '#fff' : '#222'}">Cut from the ${SHADE_NAMES[tone]} paper</text>` + page.join(""));
      page = []; x = M; y = M + 10; rowH = 0;
    };
    for (const { p, b } of items) {
      if (x + b.w > PW - M) { x = M; y += rowH + GAP; rowH = 0; }
      if (y + b.h > PH - M && page.length) { flush(); }
      const tx = x - b.x0, ty = y - b.y0;
      page.push(`<g transform="translate(${tx.toFixed(1)},${ty.toFixed(1)})"><path d="${p.d}" fill="${RAMPS[rampName][3 - tone]}" fill-opacity="0.35" stroke="#222" stroke-width="0.4" stroke-dasharray="2 1"/>`
        + `<text x="${p.cx}" y="${p.cy}" dy="0.35em" font-size="5" font-weight="700" text-anchor="middle" font-family="system-ui" fill="#e0432e">${p.n}</text></g>`);
      x += b.w + GAP; rowH = Math.max(rowH, b.h);
    }
    flush();
  }
  const H = PH * pages.length;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${PW}mm" height="${H}mm" viewBox="0 0 ${PW} ${H}">`
    + pages.map((g, i) => `<g transform="translate(0,${i * PH})"><rect width="${PW}" height="${PH}" fill="#fff" stroke="#ccc" stroke-width="0.3"/>${g}</g>`).join("")
    + `</svg>`;
}

function renderGuide() {
  const p = result.pieces[guideIdx];
  $("guide-sheet").innerHTML = sheetSVG({ upto: guideIdx });
  $("guide-label").textContent = `Piece ${p.n} of ${result.count}: cut it from the ${SHADE_NAMES[p.tone]} paper and glue it on top where the red outline is.`;
  $("prev").disabled = guideIdx === 0;
  $("next").textContent = guideIdx === result.count - 1 ? "Last piece ✓" : "Next piece →";
}

async function decodeImage(file) {
  // Chrome often fails the first createImageBitmap right after file pick
  // ("The source image could not be decoded") and succeeds a moment later
  // on the exact same bytes. Retry with backoff, then fall back to the
  // <img> decode path before giving up.
  for (let i = 0; i < 4; i++) {
    try { return await createImageBitmap(file); }
    catch (e) {
      if (i === 3) break;
      await new Promise(r => setTimeout(r, 500 * (i + 1)));
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return await createImageBitmap(img);
  } finally { URL.revokeObjectURL(url); }
}

async function postCut(body) {
  // Retry once only when the server never answered properly (network error / cold start 5xx).
  for (let attempt = 0; attempt < 2; attempt++) {
    let r;
    try { r = await fetch("/api/process", { method: "POST", headers: { "Content-Type": "application/json" }, body }); }
    catch (e) { if (attempt === 0) continue; throw new Error("Can't reach the cutter. Check your connection and try again."); }
    let data = null;
    try { data = await r.json(); } catch (e) { data = null; }
    if (r.ok && data && !data.error) return data;
    if (r.status >= 500 && attempt === 0) continue;
    throw new Error((data && data.error) || "Processing failed. Please try again.");
  }
}

async function processFile(file) {
  show("step-work");
  const msg = (t) => { $("work-msg").textContent = t; };
  try {
    msg("Reading the photo…");
    const bmp = await decodeImage(file);
    const scale = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    cv.getContext("2d").drawImage(bmp, 0, 0, w, h);

    await loadModels(msg);
    msg("Finding you in the photo…");
    const lmRes = landmarker.detect(cv);
    if (!lmRes.faceLandmarks || !lmRes.faceLandmarks.length)
      throw new Error("No face found. Try a brighter, face-forward selfie.");
    const landmarks = lmRes.faceLandmarks[0].map(p => [+p.x.toFixed(5), +p.y.toFixed(5)]);

    // Person mask (everything the segmenter does not call background), sent as a small PNG.
    const seg = segmenter.segment(cv);
    const cat = seg.categoryMask.getAsUint8Array(), mw = seg.categoryMask.width, mh = seg.categoryMask.height;
    const mcv = document.createElement("canvas"); mcv.width = mw; mcv.height = mh;
    const mctx = mcv.getContext("2d"), mid = mctx.createImageData(mw, mh);
    for (let i = 0; i < mw * mh; i++) {
      const v = cat[i] ? 255 : 0;
      mid.data[i*4] = v; mid.data[i*4+1] = v; mid.data[i*4+2] = v; mid.data[i*4+3] = 255;
    }
    mctx.putImageData(mid, 0, 0);
    seg.categoryMask.close();

    msg("Cutting the paper shapes…");
    const body = JSON.stringify({
      pixels: cv.toDataURL("image/jpeg", 0.9).split(",")[1],
      person: mcv.toDataURL("image/png").split(",")[1],
      landmarks, difficulty: "easy",
    });
    const data = await postCut(body);

    result = data; guideIdx = 0;
    $("sheet").innerHTML = sheetSVG({ numbers: true });
    const shades = new Set(data.pieces.map(p => p.tone)).size;
    $("summary").textContent = `${data.count} pieces in ${shades} paper shades. Glue them in number order, lightest paper first.`;
    show("step-result");
  } catch (e) {
    $("err-msg").textContent = String(e.message || e);
    show("step-err");
  }
}

$("file").addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) processFile(e.target.files[0]);
});
$("start-guide").addEventListener("click", () => { guideIdx = 0; renderGuide(); show("step-guide"); });
$("next").addEventListener("click", () => { if (guideIdx < result.count - 1) { guideIdx++; renderGuide(); } });
$("prev").addEventListener("click", () => { if (guideIdx > 0) { guideIdx--; renderGuide(); } });
$("exit-guide").addEventListener("click", () => show("step-result"));
$("redo").addEventListener("click", () => { $("file").value = ""; show("step-pick"); });
$("retry").addEventListener("click", () => { $("file").value = ""; show("step-pick"); });
$("dl").addEventListener("click", () => {
  const blob = new Blob([templatesSVG()], { type: "image/svg+xml" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "papercut-templates.svg";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

// paper color swatches
for (const name of Object.keys(RAMPS)) {
  const b = document.createElement("button");
  b.className = "swatch"; b.title = name; b.dataset.ramp = name;
  b.setAttribute("aria-label", `${name} paper`);
  b.style.background = `linear-gradient(135deg, ${RAMPS[name][1]}, ${RAMPS[name][2]})`;
  b.addEventListener("click", () => {
    rampName = name;
    document.querySelectorAll(".swatch").forEach(x => x.classList.toggle("sel", x === b));
    if (result) $("sheet").innerHTML = sheetSVG({ numbers: true });
  });
  $("swatches").appendChild(b);
}
document.querySelector(".swatch").classList.add("sel");
