import { FaceLandmarker, ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const STEPS = ["step-pick", "step-work", "step-style", "step-result", "step-guide", "step-err"];
const show = (id) => { for (const s of STEPS) $(s).hidden = s !== id; };

// ---------------------------------------------------------------- styles
// All models are free and run in the browser (onnxruntime-web); nothing but the stylised crop
// goes to the server. Ghibli model: AnimeGANv3 (free for non-commercial use). Others: MIT.
const STYLES = {
  realistic: { label: "Realistic", file: "/models/animegan2_face_paint_512_v2.onnx", layout: "nchw", side: 768, mult: 32,
               blurb: "Smooth painted portrait in the photo's own colours." },
  ghibli:    { label: "Ghibli-style", file: "/models/AnimeGANv3_large_Ghibli_c1_e299.onnx", layout: "nhwc", side: 768, mult: 8,
               blurb: "Soft hand-painted anime look with big clear eyes." },
  toon:      { label: "Toon", file: "/models/animegan2_celeba_distill.onnx", layout: "nchw", side: 512, mult: 32,
               blurb: "Flat comic shading, bold colours." },
};
const SHEET = { w: 190, h: 277 };   // mm, A4 printable area (matches the server)

let ort = null;
const ortSessions = {};
async function loadOrt() {
  if (ort) return ort;
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.min.js";
    s.onload = res; s.onerror = () => rej(new Error("Couldn't load the style engine. Check your connection."));
    document.head.appendChild(s);
  });
  ort = window.ort;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  return ort;
}
async function styleSession(name) {
  if (ortSessions[name]) return ortSessions[name];
  await loadOrt();
  const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  try {
    ortSessions[name] = await ort.InferenceSession.create(STYLES[name].file, { executionProviders: providers });
  } catch (e) {
    ortSessions[name] = await ort.InferenceSession.create(STYLES[name].file, { executionProviders: ["wasm"] });
  }
  return ortSessions[name];
}

// Run a style model on a canvas; returns a new canvas of the same size.
async function stylizeCanvas(src, name) {
  const cfg = STYLES[name];
  const sess = await styleSession(name);
  const s = Math.min(1, cfg.side / Math.max(src.width, src.height));
  const nw = Math.max(cfg.mult, Math.floor(src.width * s / cfg.mult) * cfg.mult);
  const nh = Math.max(cfg.mult, Math.floor(src.height * s / cfg.mult) * cfg.mult);
  const cv = document.createElement("canvas"); cv.width = nw; cv.height = nh;
  const cx = cv.getContext("2d"); cx.drawImage(src, 0, 0, nw, nh);
  const d = cx.getImageData(0, 0, nw, nh).data;
  const n = nw * nh, x = new Float32Array(n * 3), chw = cfg.layout === "nchw";
  for (let i = 0; i < n; i++) {
    const r = d[i*4] / 127.5 - 1, g = d[i*4+1] / 127.5 - 1, b = d[i*4+2] / 127.5 - 1;
    if (chw) { x[i] = r; x[n + i] = g; x[2*n + i] = b; } else { x[i*3] = r; x[i*3+1] = g; x[i*3+2] = b; }
  }
  const dims = chw ? [1, 3, nh, nw] : [1, nh, nw, 3];
  const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, dims) });
  const y = out[sess.outputNames[0]].data;
  const od = cx.createImageData(nw, nh);
  for (let i = 0; i < n; i++) {
    const r = chw ? y[i] : y[i*3], g = chw ? y[n + i] : y[i*3+1], b = chw ? y[2*n + i] : y[i*3+2];
    od.data[i*4] = (r + 1) * 127.5; od.data[i*4+1] = (g + 1) * 127.5; od.data[i*4+2] = (b + 1) * 127.5; od.data[i*4+3] = 255;
  }
  cx.putImageData(od, 0, 0);
  const full = document.createElement("canvas"); full.width = src.width; full.height = src.height;
  full.getContext("2d").drawImage(cv, 0, 0, src.width, src.height);
  return full;
}

// ---------------------------------------------------------------- MediaPipe
let landmarker = null, segmenter = null;
async function loadModels(msg) {
  if (landmarker && segmenter) return;
  msg("Loading the face models (one-time download)…");
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
    numFaces: 1, runningMode: "IMAGE" });
  segmenter = await ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" },
    outputCategoryMask: true, runningMode: "IMAGE" });
}

// ---------------------------------------------------------------- crop (same rules as engine.crop_head_shoulders)
const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
function headShouldersCrop(W, H, lmPx, personRowTop) {
  let fx0 = Infinity, fy0 = Infinity, fx1 = -Infinity, fy1 = -Infinity;
  for (const i of FACE_OVAL) { const [x, y] = lmPx[i]; fx0 = Math.min(fx0, x); fx1 = Math.max(fx1, x); fy0 = Math.min(fy0, y); fy1 = Math.max(fy1, y); }
  const fw = fx1 - fx0, fh = fy1 - fy0;
  let top = personRowTop(Math.max(0, fx0), Math.min(W, fx1));
  if (top === null) top = fy0 - .5 * fh;
  top = Math.max(top, fy0 - 1.0 * fh);
  let y0 = top - .10 * fh;
  let ch = (fy1 + 1.05 * fh) - y0;
  let cw = Math.max(ch * SHEET.w / SHEET.h, 2.3 * fw);
  ch = cw * SHEET.h / SHEET.w;
  if (y0 + ch > H) {
    y0 = H - ch >= 0 ? Math.max(0, H - ch) : y0;
    if (y0 + ch > H) {
      y0 = Math.max(0, Math.min(y0, top - .06 * fh));
      ch = Math.max(H - y0, 1.7 * fw * SHEET.h / SHEET.w);
      cw = ch * SHEET.w / SHEET.h;
    }
  }
  let x0 = (fx0 + fx1) / 2 - cw / 2;
  if (cw <= W) x0 = Math.min(Math.max(x0, 0), W - cw);
  return { x0, y0, cw, ch };
}

// ---------------------------------------------------------------- state
let photo = null;      // { crop: canvas (sheet aspect), person: canvas (mask, same size), landmarks: [[x,y]...] normalised to crop }
let styled = {};       // style name -> canvas
let result = null, guideIdx = 0, currentStyle = "ghibli";

async function decodeImage(file) {
  for (let i = 0; i < 4; i++) {
    try { return await createImageBitmap(file); }
    catch (e) { if (i === 3) break; await new Promise(r => setTimeout(r, 500 * (i + 1))); }
  }
  const url = URL.createObjectURL(file);
  try { const img = new Image(); img.src = url; await img.decode(); return await createImageBitmap(img); }
  finally { URL.revokeObjectURL(url); }
}

async function preparePhoto(file) {
  show("step-work");
  const msg = (t) => { $("work-msg").textContent = t; };
  try {
    msg("Reading the photo…");
    const bmp = await decodeImage(file);
    const scale = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
    const W = Math.round(bmp.width * scale), H = Math.round(bmp.height * scale);
    const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
    cv.getContext("2d").drawImage(bmp, 0, 0, W, H);

    await loadModels(msg);
    msg("Finding your face…");
    const lmRes = landmarker.detect(cv);
    if (!lmRes.faceLandmarks || !lmRes.faceLandmarks.length)
      throw new Error("No face found. Try a brighter, face-forward photo.");
    const lmPx = lmRes.faceLandmarks[0].map(p => [p.x * W, p.y * H]);

    const seg = segmenter.segment(cv);
    const cat = seg.categoryMask.getAsUint8Array(), mw = seg.categoryMask.width, mh = seg.categoryMask.height;
    seg.categoryMask.close();
    // mask canvas at photo size
    const mcv = document.createElement("canvas"); mcv.width = mw; mcv.height = mh;
    const mctx = mcv.getContext("2d"), mid = mctx.createImageData(mw, mh);
    for (let i = 0; i < mw * mh; i++) { const v = cat[i] ? 255 : 0; mid.data[i*4] = v; mid.data[i*4+1] = v; mid.data[i*4+2] = v; mid.data[i*4+3] = 255; }
    mctx.putImageData(mid, 0, 0);
    const personRowTop = (xa, xb) => {   // highest person row inside the face's column band
      const ca = Math.floor(xa * mw / W), cb = Math.max(ca + 1, Math.ceil(xb * mw / W));
      for (let y = 0; y < mh; y++) for (let x = ca; x < cb; x++) if (cat[y * mw + x]) return y * H / mh;
      return null;
    };
    const { x0, y0, cw, ch } = headShouldersCrop(W, H, lmPx, personRowTop);

    // crop both to the sheet aspect; the style models like ~768 px tall
    const outH = 1024, outW = Math.round(outH * SHEET.w / SHEET.h);
    const crop = document.createElement("canvas"); crop.width = outW; crop.height = outH;
    const cctx = crop.getContext("2d");
    cctx.fillStyle = "#888"; cctx.fillRect(0, 0, outW, outH);
    // replicate edge pixels where the crop leaves the photo (like BORDER_REPLICATE)
    cctx.imageSmoothingQuality = "high";
    cctx.drawImage(cv, x0, y0, cw, ch, 0, 0, outW, outH);
    const person = document.createElement("canvas"); person.width = outW; person.height = outH;
    const pctx = person.getContext("2d"); pctx.fillStyle = "#000"; pctx.fillRect(0, 0, outW, outH);
    pctx.drawImage(mcv, x0 * mw / W, y0 * mh / H, cw * mw / W, ch * mh / H, 0, 0, outW, outH);
    const landmarks = lmPx.map(([x, y]) => [+((x - x0) / cw).toFixed(5), +((y - y0) / ch).toFixed(5)]);
    photo = { crop, person, landmarks };
    styled = {};
    $("crop-preview").innerHTML = ""; $("crop-preview").appendChild(crop);
    show("step-style");
  } catch (e) {
    $("err-msg").textContent = String(e.message || e);
    show("step-err");
  }
}

async function postCut(body) {
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

async function makeStyle(name) {
  currentStyle = name;
  show("step-work");
  const msg = (t) => { $("work-msg").textContent = t; };
  try {
    if (!styled[name]) {
      msg(`Painting the ${STYLES[name].label} version… (10–20 s on a phone)`);
      await new Promise(r => setTimeout(r, 30));  // let the message paint
      styled[name] = await stylizeCanvas(photo.crop, name);
    }
    msg("Cutting the paper pieces…");
    const body = JSON.stringify({
      styled: styled[name].toDataURL("image/jpeg", 0.92).split(",")[1],
      person: photo.person.toDataURL("image/png").split(",")[1],
      landmarks: photo.landmarks, style: name, difficulty: $("difficulty").value,
    });
    result = await postCut(body); guideIdx = 0;
    $("styled-preview").innerHTML = ""; $("styled-preview").appendChild(styled[name]);
    $("sheet").innerHTML = sheetSVG({ numbers: true });
    renderPapers();
    $("summary").textContent = `${result.count} pieces in ${result.papers.length} paper colours. Glue them in number order: background first, small details last.`;
    document.querySelectorAll("#style-tabs button").forEach(b => b.classList.toggle("sel", b.dataset.style === name));
    show("step-result");
  } catch (e) {
    $("err-msg").textContent = String(e.message || e);
    show("step-err");
  }
}

// ---------------------------------------------------------------- drawing
function sheetSVG({ upto = null, numbers = false } = {}) {
  const { width: w, height: h, pieces } = result;
  const fs = (w * 0.024).toFixed(1), sw = (w * 0.004).toFixed(2);
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}mm" height="${h}mm">`;
  const label = (p) => `<text x="${p.cx}" y="${p.cy}" dy="0.35em" font-family="system-ui,sans-serif" font-size="${fs}" font-weight="700" text-anchor="middle" fill="#e0432e" stroke="#fff" stroke-width="${sw * 2}" paint-order="stroke">${p.n}</text>`;
  pieces.forEach((p, i) => {
    if (upto === null || i < upto) s += `<path d="${p.d}" fill="${p.hex}"/>`;
    else if (i === upto) s += `<path d="${p.d}" fill="${p.hex}" stroke="#e0432e" stroke-width="${sw}"/>`;
    else s += `<path d="${p.d}" fill="#ffffff" opacity="0.35"/>`;
  });
  if (numbers) pieces.forEach((p) => { s += label(p); });
  if (upto !== null && pieces[upto]) s += label(pieces[upto]);
  return s + "</svg>";
}

function renderPapers() {
  const el = $("papers"); el.innerHTML = "";
  for (const p of result.papers) {
    const d = document.createElement("div"); d.className = "paper";
    d.innerHTML = `<span class="chip" style="background:${p.hex}"></span><span>${p.name}<br><small>${p.hex} · ${p.count} piece${p.count > 1 ? "s" : ""}</small></span>`;
    el.appendChild(d);
  }
}

function bbox(d) {
  const v = d.match(/-?\d+(\.\d+)?/g).map(Number);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i + 1 < v.length; i += 2) { x0 = Math.min(x0, v[i]); x1 = Math.max(x1, v[i]); y0 = Math.min(y0, v[i+1]); y1 = Math.max(y1, v[i+1]); }
  return { x0, y0, w: x1 - x0, h: y1 - y0 };
}

// A4 pages: 1) finished picture + paper list, 2+) each paper colour's pieces laid out to cut.
function templatesSVG() {
  const PW = 210, PH = 297, M = 10, GAP = 6;
  const pages = [];
  const legend = result.papers.map((p, i) => `<rect x="${M}" y="${PH - 10 - 7 * (result.papers.length - i)}" width="6" height="5" fill="${p.hex}" stroke="#999" stroke-width="0.2"/>` +
    `<text x="${M + 8}" y="${PH - 6 - 7 * (result.papers.length - i)}" font-size="3.6" font-family="system-ui">${p.name} ${p.hex} — ${p.count} piece${p.count > 1 ? "s" : ""}</text>`).join("");
  const picH = PH - 2 * M - 7 * result.papers.length - 10;
  const picScale = Math.min((PW - 2 * M) / result.width, picH / result.height);
  pages.push(`<g transform="translate(${M},${M}) scale(${picScale})">${sheetSVG({ numbers: true }).replace(/^<svg[^>]*>|<\/svg>$/g, "")}</g>` + legend +
    `<text x="${M + result.width * picScale + 4}" y="${M + 6}" font-size="4" font-family="system-ui">${STYLES[result.style].label} · ${result.count} pieces</text>`);
  for (const paper of result.papers) {
    const items = result.pieces.filter(p => p.hex === paper.hex && p.paper !== "Background").map(p => ({ p, b: bbox(p.d) })).sort((a, b) => b.b.h - a.b.h);
    if (!items.length) continue;
    let page = [], x = M, y = M + 10, rowH = 0;
    const flush = () => {
      if (!page.length) return;
      const dark = parseInt(paper.hex.slice(1, 3), 16) * .3 + parseInt(paper.hex.slice(3, 5), 16) * .59 + parseInt(paper.hex.slice(5, 7), 16) * .11 < 128;
      pages.push(`<rect x="${M}" y="${M}" width="${PW - 2*M}" height="7" fill="${paper.hex}"/>` +
        `<text x="${M + 2}" y="${M + 5}" font-size="4.5" font-family="system-ui" font-weight="700" fill="${dark ? '#fff' : '#222'}">Cut from: ${paper.name} (${paper.hex})</text>` + page.join(""));
      page = []; x = M; y = M + 10; rowH = 0;
    };
    for (const { p, b } of items) {
      if (x + b.w > PW - M) { x = M; y += rowH + GAP; rowH = 0; }
      if (y + b.h > PH - M && page.length) flush();
      page.push(`<g transform="translate(${(x - b.x0).toFixed(1)},${(y - b.y0).toFixed(1)})"><path d="${p.d}" fill="${paper.hex}" fill-opacity="0.3" stroke="#222" stroke-width="0.4" stroke-dasharray="2 1"/>` +
        `<text x="${p.cx}" y="${p.cy}" dy="0.35em" font-size="5" font-weight="700" text-anchor="middle" font-family="system-ui" fill="#e0432e">${p.n}</text></g>`);
      x += b.w + GAP; rowH = Math.max(rowH, b.h);
    }
    flush();
  }
  const H = PH * pages.length;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${PW}mm" height="${H}mm" viewBox="0 0 ${PW} ${H}">` +
    pages.map((g, i) => `<g transform="translate(0,${i * PH})"><rect width="${PW}" height="${PH}" fill="#fff" stroke="#ccc" stroke-width="0.3"/>${g}</g>`).join("") + `</svg>`;
}

function renderGuide() {
  const p = result.pieces[guideIdx];
  $("guide-sheet").innerHTML = sheetSVG({ upto: guideIdx });
  $("guide-label").innerHTML = `Piece ${p.n} of ${result.count}: cut from <span class="chip" style="background:${p.hex}"></span> ${p.paper}` +
    (p.paper === "Background" ? ` — this is the whole sheet; a coloured card works, or skip it.` : ` and glue it where the red outline is.`);
  $("prev").disabled = guideIdx === 0;
  $("next").textContent = guideIdx === result.count - 1 ? "Last piece ✓" : "Next piece →";
}

// ---------------------------------------------------------------- wiring
const onFile = (f) => { if (f && f.type.startsWith("image/")) preparePhoto(f); };
$("file-camera").addEventListener("change", (e) => onFile(e.target.files && e.target.files[0]));
$("file-upload").addEventListener("change", (e) => onFile(e.target.files && e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); onFile(e.dataTransfer.files && e.dataTransfer.files[0]); });
document.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/")); if (it) onFile(it.getAsFile()); });

for (const [name, cfg] of Object.entries(STYLES)) {
  const b = document.createElement("button"); b.className = "style-card"; b.dataset.style = name;
  b.innerHTML = `<strong>${cfg.label}</strong><small>${cfg.blurb}</small>`;
  b.addEventListener("click", () => makeStyle(name));
  $("style-cards").appendChild(b);
  const t = document.createElement("button"); t.className = "tab"; t.dataset.style = name; t.textContent = cfg.label;
  t.addEventListener("click", () => makeStyle(name));
  $("style-tabs").appendChild(t);
}
$("difficulty").addEventListener("change", () => { if (result) makeStyle(currentStyle); });
$("start-guide").addEventListener("click", () => { guideIdx = 0; renderGuide(); show("step-guide"); });
$("next").addEventListener("click", () => { if (guideIdx < result.count - 1) { guideIdx++; renderGuide(); } });
$("prev").addEventListener("click", () => { if (guideIdx > 0) { guideIdx--; renderGuide(); } });
$("exit-guide").addEventListener("click", () => show("step-result"));
const reset = () => { $("file-camera").value = ""; $("file-upload").value = ""; show("step-pick"); };
$("redo").addEventListener("click", reset);
$("retry").addEventListener("click", reset);
$("back-style").addEventListener("click", () => show("step-style"));
$("other-photo").addEventListener("click", reset);
$("dl").addEventListener("click", () => {
  const blob = new Blob([templatesSVG()], { type: "image/svg+xml" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `papercut-${result.style}-templates.svg`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});
$("dl-styled").addEventListener("click", () => {
  const a = document.createElement("a"); a.href = styled[currentStyle].toDataURL("image/png");
  a.download = `portrait-${currentStyle}.png`; a.click();
});
