// outfit mirror front end: photo upload or live camera with a hand-gesture trigger,
// a top-to-bottom scan animation over the body landmarks, and the explainable results.

const $ = (id) => document.getElementById(id);
const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0";
const GESTURE_MODEL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32]];
const HOLD_MS = 700;   // how long the open palm has to be held before the countdown starts
const COUNTDOWN = 3;   // seconds to lower the hand and pose

const state = { mode: "photo", blob: null, mirrored: false, result: null, busy: false, stream: null, recognizer: null, anim: null };

// ---------- setup ----------

fetch("/api/aesthetics").then((r) => r.json()).then(({ aesthetics }) => {
  for (const name of aesthetics) $("target").add(new Option(name, name));
});

document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("again").addEventListener("click", () => { reset(); if (state.mode === "live") startLive(); });
$("target").addEventListener("change", () => { if (state.blob && !state.busy) analyze(state.blob); });

$("file").addEventListener("change", (e) => e.target.files[0] && usePhoto(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && usePhoto(e.dataTransfer.files[0]));

window.addEventListener("resize", () => state.result && drawSkeleton(1));
window.addEventListener("keydown", (e) => {   // space bar: a fallback trigger in live mode
  if (e.code === "Space" && state.mode === "live" && !$("live").hidden && !state.counting) { e.preventDefault(); countdown(); }
});

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  reset();
  if (mode === "live") startLive(); else stopLive();
}

function reset() {
  cancelAnimationFrame(state.anim);
  state.blob = state.result = null;
  $("stage").hidden = true;
  $("overlay").classList.remove("show");
  $("drop").hidden = state.mode !== "photo";
  $("live").hidden = state.mode !== "live";
  $("error").hidden = true;
  $("file").value = "";
}

// ---------- photo mode ----------

function usePhoto(file) {
  if (!file.type.startsWith("image/")) return showError("That isn't an image. Try a JPG or PNG.");
  showStage(URL.createObjectURL(file), false);
  analyze(file);
}

function showStage(url, mirrored) {
  state.mirrored = mirrored;
  $("drop").hidden = $("live").hidden = true;
  $("stage").hidden = false;
  $("stage").classList.toggle("mirrored", mirrored);
  $("overlay").classList.remove("show");
  $("photo").src = url;
}

// ---------- live mode ----------

async function startLive() {
  setStatus("Starting camera…");
  try {
    state.stream = state.stream || await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 1280, height: 720 } });
  } catch {
    return setStatus("Camera blocked. Allow camera access, or switch to Photo.");
  }
  const video = $("video");
  video.srcObject = state.stream;
  await video.play();
  setStatus("Loading the gesture model…");
  try {
    if (!state.recognizer) {
      const { FilesetResolver, GestureRecognizer } = await import(`${MP}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
      state.recognizer = await GestureRecognizer.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: GESTURE_MODEL, delegate: "GPU" }, runningMode: "VIDEO", numHands: 1,
      });
    }
  } catch {
    setStatus("Step back so your whole outfit is in frame, then press space");
    return;
  }
  setStatus("Step back so your whole outfit is in frame, then hold up an open palm ✋");
  watchForPalm();
}

function stopLive() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
}

function watchForPalm() {
  const video = $("video");
  let since = null;
  const tick = () => {
    if (state.mode !== "live" || $("live").hidden || state.counting) return;
    if (video.readyState >= 2) {
      const r = state.recognizer.recognizeForVideo(video, performance.now());
      const g = r.gestures?.[0]?.[0];
      if (g && g.categoryName === "Open_Palm" && g.score > 0.6) {
        since ??= performance.now();
        setStatus("Hold it…", true);
        if (performance.now() - since > HOLD_MS) return countdown();
      } else if (since) {
        since = null;
        setStatus("Step back so your whole outfit is in frame, then hold up an open palm ✋");
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function countdown() {
  state.counting = true;
  setStatus("Hand down, strike a pose", true);
  const el = $("countdown");
  el.hidden = false;
  for (let n = COUNTDOWN; n > 0; n--) {
    el.textContent = n;
    await new Promise((r) => setTimeout(r, 1000));
  }
  el.hidden = true;
  state.counting = false;
  const blob = await captureFrame();
  showStage(URL.createObjectURL(blob), true);
  analyze(blob);
}

function captureFrame() {
  const video = $("video");
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);   // unmirrored, so left and right stay true for the analysis
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
}

function setStatus(text, hot = false) {
  $("liveStatus").textContent = text;
  $("liveStatus").classList.toggle("hot", hot);
}

// ---------- analysis ----------

async function analyze(blob) {
  state.blob = blob;
  state.busy = true;
  $("error").hidden = true;
  document.querySelectorAll(".tile").forEach((t) => t.classList.add("loading"));
  scanLoop();

  const form = new FormData();
  form.append("photo", blob, "outfit.jpg");
  form.append("target", $("target").value);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Something went wrong.");
    state.result = data;
    render(data);
    reveal();
  } catch (e) {
    cancelAnimationFrame(state.anim);
    clearCanvas();
    showError(e.message === "Failed to fetch" ? "Can't reach the mirror server. Is it running?" : e.message);
  } finally {
    state.busy = false;
    document.querySelectorAll(".tile").forEach((t) => t.classList.remove("loading"));
  }
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

function render(d) {
  const f = d.feedback;
  const by = Object.fromEntries(f.sections.map((s) => [s.key, s]));

  const a = by.aesthetic;
  $("aestheticTile").classList.remove("idle");
  $("aestheticName").textContent = a ? a.label : "—";
  const why = d.vibe?.because?.length ? ` It's the ${d.vibe.because.join(", ")}.` : "";
  $("aestheticReason").textContent = f.headline + why;
  $("runners").innerHTML = (d.vibe?.runners_up || []).map((r) => `<span>${r.aesthetic}</span>`).join("");

  const c = by.compare;
  $("compareBody").hidden = !c;
  if (c) {
    $("matchFill").style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => ($("matchFill").style.width = c.score + "%")));
    $("matchLabel").textContent = c.label;
    $("compareReason").textContent = c.reason;
  }

  scoreTile("color", by.color);
  $("swatches").innerHTML = d.swatches.map((s) =>
    `<div style="background:${s.hex};flex-grow:${s.share}" data-name="${s.name} · ${Math.round(s.share * 100)}%"></div>`).join("");
  scoreTile("proportion", by.proportion);

  if (f.notes.length) $("proportionReason").textContent += " " + f.notes.join(" ");
  const tips = f.tips;
  $("tipsTile").hidden = !tips.length;
  $("tips").innerHTML = tips.map((t) => `<li>${t}</li>`).join("");
}

function scoreTile(key, s) {
  $(key + "Score").innerHTML = s.score == null ? "—" : `${s.score}<small>/100</small>`;
  $(key + "Label").textContent = s.label;
  $(key + "Reason").textContent = s.reason;
}

// ---------- scan animation ----------

function canvasSize() {
  const c = $("scan");
  const dpr = window.devicePixelRatio || 1;
  c.width = c.clientWidth * dpr;
  c.height = c.clientHeight * dpr;
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: c.clientWidth, h: c.clientHeight };
}

function clearCanvas() {
  const { ctx, w, h } = canvasSize();
  ctx.clearRect(0, 0, w, h);
}

function scanLine(ctx, w, y) {
  const g = ctx.createLinearGradient(0, y - 60, 0, y);
  g.addColorStop(0, "rgba(200,255,61,0)");
  g.addColorStop(1, "rgba(200,255,61,.35)");
  ctx.fillStyle = g;
  ctx.fillRect(0, y - 60, w, 60);
  ctx.fillStyle = "#c8ff3d";
  ctx.shadowColor = "#c8ff3d";
  ctx.shadowBlur = 16;
  ctx.fillRect(0, y - 1.5, w, 3);
  ctx.shadowBlur = 0;
}

// while waiting on the server: a scan line sweeping up and down
function scanLoop() {
  cancelAnimationFrame(state.anim);
  const start = performance.now();
  const frame = (now) => {
    const { ctx, w, h } = canvasSize();
    ctx.clearRect(0, 0, w, h);
    const t = ((now - start) / 1600) % 2;
    scanLine(ctx, w, (t < 1 ? t : 2 - t) * h);
    state.anim = requestAnimationFrame(frame);
  };
  state.anim = requestAnimationFrame(frame);
}

// once results land: one final pass top to bottom, dropping landmark dots as it goes, then the garment mask
function reveal() {
  cancelAnimationFrame(state.anim);
  const d = state.result;
  const [x0, y0, x1, y1] = d.box;
  const [W, H] = d.size;
  Object.assign($("overlay").style, { left: `${(x0 / W) * 100}%`, top: `${(y0 / H) * 100}%`, width: `${((x1 - x0) / W) * 100}%`, height: `${((y1 - y0) / H) * 100}%` });
  $("overlay").src = d.overlay;
  const start = performance.now();
  const frame = (now) => {
    const p = Math.min(1, (now - start) / 1400);
    drawSkeleton(p);
    if (p > 0.55) $("overlay").classList.add("show");
    if (p < 1) state.anim = requestAnimationFrame(frame);
  };
  state.anim = requestAnimationFrame(frame);
}

function drawSkeleton(progress) {
  const { ctx, w, h } = canvasSize();
  ctx.clearRect(0, 0, w, h);
  const lm = state.result?.landmarks;
  if (!lm) return;
  const cut = progress * h;
  const pt = (i) => [lm[i].x * w, lm[i].y * h];
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(200,255,61,.7)";
  for (const [a, b] of BONES) {
    const [ax, ay] = pt(a), [bx, by] = pt(b);
    if (Math.max(ay, by) > cut || lm[a].visibility < 0.3 || lm[b].visibility < 0.3) continue;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  lm.forEach((l, i) => {
    const [x, y] = pt(i);
    if (y > cut || l.visibility < 0.3 || i < 11) return;   // skip the face points, the outfit is the point
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#c8ff3d";
    ctx.shadowColor = "#c8ff3d";
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
  if (progress < 1) scanLine(ctx, w, cut);
}
