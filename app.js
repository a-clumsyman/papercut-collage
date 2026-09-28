import { FaceLandmarker, HandLandmarker, ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const show = (id) => {
  for (const s of ["step-pick","step-work","step-result","step-guide","step-err"]) $(s).hidden = s !== id;
};

let landmarker = null, segmenter = null, handmarker = null;
async function loadModels(msg) {
  if (landmarker && segmenter && handmarker) return;
  msg("Loading the cut-out models (one-time download)…");
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
    numFaces: 1, outputFaceBlendshapes: true, runningMode: "IMAGE" });
  handmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" },
    numHands: 2, minHandDetectionConfidence: 0.25,
    minHandPresenceConfidence: 0.25, runningMode: "IMAGE" });
  segmenter = await ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" },
    outputCategoryMask: true, runningMode: "IMAGE" });
}

let result = null, guideIdx = 0;

// Monochrome paper ramps (dark -> light), per his reference images
const RAMPS = {
  green:  ['#1e5c38', '#3d8a4e', '#79c47c', '#c9ecc4'],
  blue:   ['#1e3a7a', '#2f6fc4', '#7db4e8', '#cfe7f7'],
  pink:   ['#9c2b4f', '#e0496e', '#f588a5', '#fbd0da'],
  yellow: ['#c98a12', '#eab316', '#f7d442', '#fcefa8'],
};
let rampName = 'green';
function rampHex(p, lmin, lmax) {
  const r = RAMPS[rampName];
  let t = lmax > lmin ? (p.L - lmin) / (lmax - lmin) : 0.5;
  return r[Math.max(0, Math.min(3, Math.floor(t * 4)))];
}
function pieceColor(p) {
  const Ls = result.pieces.map(q => q.L);
  return rampHex(p, Math.min(...Ls), Math.max(...Ls));
}

function sheetSVG(upto = null) {
  const { width: w, height: h, pieces } = result;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">`;
  s += `<rect width="${w}" height="${h}" fill="#eaf2f5"/>`;
  pieces.forEach((p, i) => {
    if (upto === null) {
      s += `<path fill-rule="evenodd" d="${p.d}" fill="${pieceColor(p)}"/>`;
    } else if (i < upto) {
      s += `<path fill-rule="evenodd" d="${p.d}" fill="${pieceColor(p)}"/>`;
    } else if (i === upto) {
      s += `<path fill-rule="evenodd" d="${p.d}" fill="${pieceColor(p)}" stroke="#e0432e" stroke-width="4"/>`;
      s += `<text x="${p.cx}" y="${p.cy}" font-size="42" font-weight="700" text-anchor="middle" fill="#e0432e" stroke="#fff" stroke-width="6" paint-order="stroke">${p.n}</text>`;
    } else {
      s += `<path fill-rule="evenodd" d="${p.d}" fill="#ffffff" opacity="0.35"/>`;
    }
  });
  return s + "</svg>";
}

function renderGuide() {
  const p = result.pieces[guideIdx];
  $("guide-sheet").innerHTML = sheetSVG(guideIdx);
  $("guide-label").textContent = `Piece ${p.n} of ${result.count} - cut this color, stick where the red outline is.`;
  $("prev").disabled = guideIdx === 0;
  $("next").textContent = guideIdx === result.count - 1 ? "Last piece ✓" : "Next piece →";
}

// ---- parametric flat-shaded portrait (built from landmarks + segmentation, not photo shading) ----
const LV = { bg:245, face:165, nose:100, lips:100, neck:100, dark:35, hair2:100, clothLight:215, clothDark:20 };
const EYE_LI = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const EYE_RI = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const LIPS_I = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
const BROW_LI = [70,63,105,66,107,55,65,52,53,46];
const BROW_RI = [336,296,334,293,300,276,283,282,295,285];

function convexHull(pts) {
  const p = [...pts].sort((a,b) => a[0]-b[0] || a[1]-b[1]);
  const cross = (o,a,b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const lower = [];
  for (const q of p) { while (lower.length>=2 && cross(lower[lower.length-2], lower[lower.length-1], q) <= 0) lower.pop(); lower.push(q); }
  const upper = [];
  for (let i=p.length-1;i>=0;i--) { const q=p[i]; while (upper.length>=2 && cross(upper[upper.length-2], upper[upper.length-1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0,-1).concat(upper.slice(0,-1));
}

function polyMask(w, h, pts, idxs, scale) {
  let hp = convexHull(idxs.map(i => pts[i]));
  const cx = hp.reduce((s,p)=>s+p[0],0)/hp.length, cy = hp.reduce((s,p)=>s+p[1],0)/hp.length;
  hp = hp.map(p => [(p[0]-cx)*scale+cx, (p[1]-cy)*scale+cy]);
  const m = new Uint8Array(w*h);
  const ys = hp.map(p=>p[1]);
  const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(h-1, Math.ceil(Math.max(...ys)));
  for (let y=y0; y<=y1; y++) {
    const xs = [];
    for (let i=0;i<hp.length;i++) {
      const a=hp[i], b=hp[(i+1)%hp.length];
      if ((a[1]<=y+0.5) !== (b[1]<=y+0.5))
        xs.push(a[0] + (y+0.5-a[1])*(b[0]-a[0])/(b[1]-a[1]));
    }
    xs.sort((p,q)=>p-q);
    for (let k=0;k+1<xs.length;k+=2)
      for (let x=Math.max(0,Math.round(xs[k])); x<=Math.min(w-1,Math.round(xs[k+1])); x++) m[y*w+x]=1;
  }
  return m;
}

function blurMask(m, w, h, r) {  // two-pass separable box blur, threshold majority
  let cur = Float32Array.from(m);
  for (let pass=0; pass<2; pass++) {
    const tmp = new Float32Array(w*h);
    for (let y=0;y<h;y++) {
      let acc=0;
      for (let x=-r;x<=r;x++) acc += cur[y*w+Math.min(w-1,Math.max(0,x))];
      for (let x=0;x<w;x++) {
        tmp[y*w+x] = acc/(2*r+1);
        const xa=Math.min(w-1,x+r+1), xs=Math.max(0,x-r);
        acc += cur[y*w+xa]-cur[y*w+xs];
      }
    }
    const out = new Float32Array(w*h);
    for (let x=0;x<w;x++) {
      let acc=0;
      for (let y=-r;y<=r;y++) acc += tmp[Math.min(h-1,Math.max(0,y))*w+x];
      for (let y=0;y<h;y++) {
        out[y*w+x] = acc/(2*r+1);
        const ya=Math.min(h-1,y+r+1), ys=Math.max(0,y-r);
        acc += tmp[ya*w+x]-tmp[ys*w+x];
      }
    }
    cur = out;
  }
  const m2 = new Uint8Array(w*h);
  for (let i=0;i<w*h;i++) m2[i] = cur[i] >= 0.5 ? 1 : 0;
  return m2;
}

function fillHoles(m, w, h) {
  const seen = new Uint8Array(w*h), st = [];
  for (let x=0;x<w;x++){ st.push(x, (h-1)*w+x); }
  for (let y=0;y<h;y++){ st.push(y*w, y*w+w-1); }
  while (st.length) {
    const i = st.pop();
    if (seen[i] || m[i]) continue;
    seen[i]=1;
    const x=i%w, y=(i/w)|0;
    if (x>0) st.push(i-1); if (x<w-1) st.push(i+1);
    if (y>0) st.push(i-w); if (y<h-1) st.push(i+w);
  }
  const out = new Uint8Array(w*h);
  for (let i=0;i<w*h;i++) out[i] = (m[i] || !seen[i]) ? 1 : 0;
  return out;
}

function removeIslands(m, w, h, minArea) {
  const lab = new Int32Array(w*h).fill(-1);
  let nc = 0; const areas = [];
  for (let i=0;i<w*h;i++) {
    if (!m[i] || lab[i]>=0) continue;
    const st=[i]; lab[i]=nc; let a=0;
    while (st.length) {
      const j=st.pop(); a++;
      const x=j%w, y=(j/w)|0;
      for (const k of [x>0?j-1:-1, x<w-1?j+1:-1, y>0?j-w:-1, y<h-1?j+w:-1])
        if (k>=0 && m[k] && lab[k]<0) { lab[k]=nc; st.push(k); }
    }
    areas.push(a); nc++;
  }
  for (let i=0;i<w*h;i++) if (lab[i]>=0 && areas[lab[i]]<minArea) m[i]=0;
}

const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
function shapeMask(w,h,poly) {
  const m = new Uint8Array(w*h);
  let y0=Math.max(0,Math.floor(Math.min(...poly.map(p=>p[1]))));
  let y1=Math.min(h-1,Math.ceil(Math.max(...poly.map(p=>p[1]))));
  for(let y=y0;y<=y1;y++) {
    let xs=[];
    for(let i=0;i<poly.length;i++) {
      const a=poly[i],b=poly[(i+1)%poly.length];
      if ((a[1]<=y+.5)!==(b[1]<=y+.5)) xs.push(a[0]+(y+.5-a[1])*(b[0]-a[0])/(b[1]-a[1]));
    }
    xs.sort((a,b)=>a-b);
    for(let i=0;i+1<xs.length;i+=2)
      for(let x=Math.max(0,Math.round(xs[i]));x<=Math.min(w-1,Math.round(xs[i+1]));x++) m[y*w+x]=1;
  }
  return m;
}
function median(a,fallback) { if(!a.length)return fallback;a.sort((x,y)=>x-y);return a[a.length>>1]; }
function largestHairAboveBrow(hair,w,h,browTop) {
  const seen=new Uint8Array(w*h), q=new Int32Array(w*h);
  for(let i=0;i<hair.length;i++) {
    if(!hair[i]||seen[i])continue;
    let front=0,back=1,minY=(i/w)|0;q[0]=i;seen[i]=1;
    while(front<back) {
      const j=q[front++],x=j%w,y=(j/w)|0;minY=Math.min(minY,y);
      for(const k of [x>0?j-1:-1,x<w-1?j+1:-1,y>0?j-w:-1,y<h-1?j+w:-1])
        if(k>=0&&hair[k]&&!seen[k]) { seen[k]=1;q[back++]=k; }
    }
    if(minY>browTop-8)for(let k=0;k<back;k++)hair[q[k]]=0;
  }
}
function binaryClose(m,w,h,iters=4) {
  let a=m;
  for(let t=0;t<iters;t++) {
    const b=new Uint8Array(m.length);
    for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++) {
      const i=y*w+x;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if(a[i+dy*w+dx])b[i]=1;
    }
    a=b;
  }
  for(let t=0;t<iters;t++) {
    const b=new Uint8Array(m.length);
    for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++) {
      const i=y*w+x;
      b[i]=1;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if(!a[i+dy*w+dx])b[i]=0;
    }
    a=b;
  }
  return a;
}
function buildSynthetic(photo,w,h,cat,catW,catH,lm,hands) {
  const pts=lm.map(p=>[p[0]*w,p[1]*h]),n=w*h,d=photo.data;
  const cls=new Uint8Array(n),lum=new Float32Array(n),chr=new Float32Array(n);
  const fl=[],fc=[];
  for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
    const i=y*w+x,r=d[i*4],g=d[i*4+1],b=d[i*4+2];
    cls[i]=cat[Math.min(catH-1,(y*catH/h)|0)*catW+Math.min(catW-1,(x*catW/w)|0)];
    lum[i]=.299*r+.587*g+.114*b;chr[i]=Math.max(r,g,b)-Math.min(r,g,b);
    if(cls[i]===3&&i%3===0){fl.push(lum[i]);fc.push(chr[i]);}
  }
  const medL=median(fl,140),medC=median(fc,20);
  let face=new Uint8Array(n),hair=new Uint8Array(n),cloth=new Uint8Array(n);
  for(let i=0;i<n;i++){face[i]=+(cls[i]===3);hair[i]=+(cls[i]===1);cloth[i]=+(cls[i]===4);}
  face=fillHoles(blurMask(face,w,h,6),w,h);
  const oval=convexHull(FACE_OVAL.map(i=>pts[i]));
  const ocx=oval.reduce((a,p)=>a+p[0],0)/oval.length,ocy=oval.reduce((a,p)=>a+p[1],0)/oval.length;
  const om=shapeMask(w,h,oval.map(p=>[(p[0]-ocx)*1.35+ocx,(p[1]-ocy)*1.35+ocy]));
  for(let i=0;i<n;i++)face[i]&=om[i];
  hair=blurMask(hair,w,h,6);
  for(let i=0;i<n;i++)hair[i]&=face[i]^1;
  // Two shallow cuts stay attached to the existing hair/face boundary.
  const faceWidthNotch=Math.abs(pts[454][0]-pts[234][0]);
  for(const idx of [67,297]) {
    const x0=Math.floor(pts[idx][0]), anchor=Math.floor(pts[idx][1]);
    if(x0<8||x0>=w-8)continue;
    let boundary=null;
    for(let y=Math.max(5,anchor-Math.floor(.18*faceWidthNotch));y<Math.min(h-6,anchor+Math.floor(.18*faceWidthNotch));y++)
      if(hair[y*w+x0]&&face[(y+5)*w+x0])boundary=y;
    if(boundary===null)continue;
    const depth=Math.max(7,Math.floor(.031*faceWidthNotch)),half=Math.max(5,Math.floor(.023*faceWidthNotch));
    for(let dy=0;dy<depth;dy++){
      const y=boundary-dy;if(y<0)break;
      const hw=Math.max(1,Math.floor(half*(1-dy/depth)));
      for(let x=Math.max(0,x0-hw);x<Math.min(w,x0+hw+1);x++)
        if(hair[y*w+x]){hair[y*w+x]=0;face[y*w+x]=1;}
    }
  }
  const browTop=Math.min(...BROW_LI.concat(BROW_RI).map(i=>pts[i][1]));
  largestHairAboveBrow(hair,w,h,browTop);
  const hm=shapeMask(w,h,oval.map(p=>[(p[0]-ocx)*1.9+ocx,(p[1]-ocy)*2.2+ocy-.25*ocy]));
  for(let i=0;i<n;i++)hair[i]&=hm[i];
  cloth=blurMask(cloth,w,h,4);
  for(let i=0;i<n;i++)cloth[i]&=(face[i]|hair[i])^1;
  const hairL=[],clothL=[];
  for(let i=0;i<n;i+=3){if(hair[i])hairL.push(lum[i]);if(cloth[i])clothL.push(lum[i]);}
  const hairMed=hairL.length?median(hairL,null):null;
  const FACE=medL<95?95:165,SHADE=FACE-55;
  const HAIR=hairMed===null?35:hairMed<.62*medL?35:hairMed<1.08*medL?120:215;
  const BROW=HAIR<=120?35:120;
  const CLOTH=clothL.length>200?(clothL.reduce((a,b)=>a+b,0)/clothL.length<110?20:215):null;
  const neutral=medC<8,noseBaseY=pts[2][1];
  let beard=new Uint8Array(n);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
    const i=y*w+x;
    if(face[i]&&y>noseBaseY&&lum[i]<.85*medL&&chr[i]<=Math.max(4,.65*medC))beard[i]=1;
  }
  beard=blurMask(beard,w,h,7);removeIslands(beard,w,h,1400);
  if(neutral) {
    const jawLimit=pts[152][1]+1.25*(pts[152][1]-noseBaseY);
    for(let y=Math.max(0,Math.ceil(jawLimit));y<h;y++)beard.fill(0,y*w,(y+1)*w);
    beard=fillHoles(binaryClose(beard,w,h),w,h);
  } else if(medL<150) {
    let bright=new Uint8Array(n);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const i=y*w+x;
      if(face[i]&&y>noseBaseY&&lum[i]>1.45*medL&&chr[i]<.4*medC)bright[i]=1;
    }
    bright=blurMask(bright,w,h,7);removeIslands(bright,w,h,2500);
    let bc=0;for(let i=0;i<n;i++)bc+=bright[i];
    if(bc>2500)for(let i=0;i<n;i++)beard[i]|=bright[i];
  }
  let bs=0,chin=0,center=0;
  const chinY=pts[152][1],fw=Math.abs(pts[454][0]-pts[234][0]);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(beard[y*w+x]) {
    bs++;if(y>chinY-25)chin++;if(Math.abs(x-pts[152][0])<.25*fw)center++;
  }
  // The strict gate only decides what gets PAINTED into the levels map.
  // meta.beard goes out on a loose signal so the server - which has the real
  // photo and a separation gate - makes the beard call, not this one check.
  const hasBeard=bs>1400&&chin/bs>.2&&center/bs>.2,BEARD=FACE===95?40:55;
  const beardSignal=bs>400?BEARD:null;
  // Pick the shadow side from the measured cheek tones. The server traces its
  // plane along the corresponding facial mesh, not an arbitrary luma blob.
  const cheekMed=(id)=>{
    const x=Math.round(pts[id][0]),y=Math.round(pts[id][1]),v=[];
    for(let yy=Math.max(0,y-8);yy<=Math.min(h-1,y+8);yy++)
      for(let xx=Math.max(0,x-8);xx<=Math.min(w-1,x+8);xx++)v.push(lum[yy*w+xx]);
    return median(v,medL);
  };
  const shadeSide=cheekMed(205)<cheekMed(425)?'left':'right';
  const out=new Uint8Array(n).fill(245);
  for(let i=0;i<n;i++){
    if(CLOTH!==null&&cloth[i])out[i]=CLOTH;
    if(cls[i]===2&&!face[i])out[i]=SHADE;
    if(face[i])out[i]=FACE;
    if(hasBeard&&beard[i])out[i]=BEARD;
    if(hair[i])out[i]=HAIR;
  }
  // One raised hand: landmark hull gates the segmenter's actual skin edge.
  // A hard wrist limit keeps the hand piece off the neck and shoulder.
  let chosen=null;
  for(const hand of hands){
    const hp=hand.map(v=>[v.x*w,v.y*h]),p=hp[9];
    if(Math.hypot(p[0]-pts[1][0],p[1]-pts[1][1])>.8*fw)continue;
    if(!chosen||Math.abs(p[0]-pts[1][0])<Math.abs(chosen[9][0]-pts[1][0]))chosen=hp;
  }
  let handArea=0;
  if(chosen){
    let hull=convexHull(chosen);
    const hc=hull.reduce((a,p)=>[a[0]+p[0]/hull.length,a[1]+p[1]/hull.length],[0,0]);
    hull=hull.map(p=>[(p[0]-hc[0])*1.19+hc[0],(p[1]-hc[1])*1.19+hc[1]]);
    const roi=shapeMask(w,h,hull);
    let raw=new Uint8Array(n);
    const wristLimit=Math.max(0,Math.min(h,Math.floor(chosen[0][1]+.03*fw)));
    for(let y=0;y<wristLimit;y++)for(let x=0;x<w;x++){
      const i=y*w+x;if(cls[i]===2&&roi[i])raw[i]=1;
    }
    raw=blurMask(raw,w,h,3);
    // Only the largest connected portion is a usable hand cutout.
    const seen=new Uint8Array(n),groups=[];
    for(let i=0;i<n;i++){
      if(!raw[i]||seen[i])continue;
      const q=[i];seen[i]=1;
      for(let k=0;k<q.length;k++){
        const j=q[k],x=j%w,y=(j/w)|0;
        for(const v of [x>0?j-1:-1,x<w-1?j+1:-1,y>0?j-w:-1,y<h-1?j+w:-1])
          if(v>=0&&raw[v]&&!seen[v]){seen[v]=1;q.push(v);}
      }
      groups.push(q);
    }
    groups.sort((a,b)=>b.length-a.length);
    if(groups.length&&groups[0].length>Math.max(1800,.025*fw*fw)){
      handArea=groups[0].length;
      for(const i of groups[0])out[i]=180;
    }
  }
  return {levels:out,meta:{face:FACE,shade:SHADE,hair:HAIR,beard:beardSignal,cloth:CLOTH,brow:BROW,lips:SHADE,shade_side:shadeSide,hand:handArea?180:null}};
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

async function processFile(file) {
  show("step-work");
  const msg = (t) => { $("work-msg").textContent = t; };
  try {
    msg("Reading the photo…");
    const bmp = await decodeImage(file);
    const scale = Math.min(1, 900 / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const cx = cv.getContext("2d", { willReadFrequently: true });
    cx.drawImage(bmp, 0, 0, w, h);

    await loadModels(msg);
    msg("Finding you in the photo…");
    const lmRes = landmarker.detect(cv);
    if (!lmRes.faceLandmarks || !lmRes.faceLandmarks.length)
      throw new Error("No face found. Try a brighter, face-forward selfie.");
    const landmarks = lmRes.faceLandmarks[0].map(p => [p.x, p.y]);
    const blendshapes = {};
    if (lmRes.faceBlendshapes && lmRes.faceBlendshapes.length)
      for (const c of lmRes.faceBlendshapes[0].categories) blendshapes[c.categoryName] = c.score;

    const segRes = segmenter.segment(cv);
    const cat = segRes.categoryMask.getAsUint8Array();

    msg("Styling the portrait…");
    const photoPix = cx.getImageData(0, 0, w, h);
    const { levels, meta } = buildSynthetic(photoPix, w, h, cat, segRes.categoryMask.width, segRes.categoryMask.height, landmarks, handmarker.detect(cv).landmarks);
    const sc = document.createElement("canvas"); sc.width = w; sc.height = h;
    const sctx = sc.getContext("2d");
    const sid = sctx.createImageData(w, h);
    const mcv = document.createElement("canvas"); mcv.width = w; mcv.height = h;
    const mctx = mcv.getContext("2d");
    const mid = mctx.createImageData(w, h);
    for (let i = 0; i < w*h; i++) {
      const v = levels[i];
      sid.data[i*4] = v; sid.data[i*4+1] = v; sid.data[i*4+2] = v; sid.data[i*4+3] = 255;
      const mv = v < 245 ? 255 : 0;
      mid.data[i*4] = mv; mid.data[i*4+1] = mv; mid.data[i*4+2] = mv; mid.data[i*4+3] = 255;
    }
    sctx.putImageData(sid, 0, 0);
    mctx.putImageData(mid, 0, 0);
    const maskData = mcv.toDataURL("image/png").split(",")[1];
    const photoData = sc.toDataURL("image/png").split(",")[1];

    msg("Cutting the paper shapes…");
    const pixelsData = cv.toDataURL("image/jpeg", 0.85).split(",")[1];
    const body = JSON.stringify({ photo: photoData, mask: maskData, landmarks, meta, pixels: pixelsData, blendshapes });
    const cut = () => fetch("/api/process", {
      method: "POST", headers: { "Content-Type": "application/json" }, body
    });
    let r = await cut(), data = null;
    try { data = await r.json(); } catch (e) { data = null; }
    if (!r.ok || !data) {  // cold-start hiccup: warm the function and retry once
      msg("Warming up the cutter…");
      r = await cut();
      data = await r.json();
    }
    if (!r.ok || data.error) throw new Error(data.error || "Processing failed.");

    result = data; guideIdx = 0;
    $("sheet").innerHTML = sheetSVG();
    $("summary").textContent = `${data.count} pieces to cut. Each number on the sheet is one piece.`;
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
  const blob = new Blob([sheetSVG()], { type: "image/svg+xml" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "papercut-sheet.svg";
  a.click();
});

// paper color swatches
for (const name of Object.keys(RAMPS)) {
  const b = document.createElement("button");
  b.className = "swatch"; b.title = name; b.dataset.ramp = name;
  b.style.background = `linear-gradient(135deg, ${RAMPS[name][1]}, ${RAMPS[name][2]})`;
  b.addEventListener("click", () => {
    rampName = name;
    document.querySelectorAll(".swatch").forEach(x => x.classList.toggle("sel", x === b));
    $("sheet").innerHTML = sheetSVG();
  });
  $("swatches").appendChild(b);
}
document.querySelector(".swatch").classList.add("sel");
