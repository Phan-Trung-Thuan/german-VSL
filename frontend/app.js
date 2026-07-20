/* ===================================================================
   DE → Sign Pipeline  –  Frontend Logic
   Pipeline: Record → ASR → Gloss → GIF
   =================================================================== */

// ── DOM references ──────────────────────────────────────────────────
const startBtn        = document.getElementById("startBtn");
const stopBtn         = document.getElementById("stopBtn");
const runBtn          = document.getElementById("runBtn");
const clearBtn        = document.getElementById("clearBtn");
const glossBtn        = document.getElementById("glossBtn");
const downloadBtn     = document.getElementById("downloadBtn");

const fileInput       = document.getElementById("fileInput");
const dropZone        = document.getElementById("dropZone");
const audioPlayer     = document.getElementById("audioPlayer");
const audioWrap       = document.getElementById("audioWrap");

const transcriptBox   = document.getElementById("transcriptBox");
const asrMeta         = document.getElementById("asrMeta");
const asrSpinner      = document.getElementById("asrSpinner");

const glossTokensEl   = document.getElementById("glossTokens");
const glossRawEl      = document.getElementById("glossRaw");
const glossSpinner    = document.getElementById("glossSpinner");

const gifWrap         = document.getElementById("gifWrap");
const gifMeta         = document.getElementById("gifMeta");
const gifSpinner      = document.getElementById("gifSpinner");

const logBar          = document.getElementById("logBar");
const logMsg          = document.getElementById("logMsg");
const timingBar       = document.getElementById("timingBar");
const timingText      = document.getElementById("timingText");
const timerBadge      = document.getElementById("timerBadge");

const apiStatus       = document.getElementById("apiStatus");
const asrBadge        = document.getElementById("asrBadge");
const lexiconBadge    = document.getElementById("lexiconBadge");

const glosserSelect   = document.getElementById("glosserSelect");
const signedLangSelect = document.getElementById("signedLangSelect");

// ── State ───────────────────────────────────────────────────────────
let audioContext    = null;
let mediaStream     = null;
let sourceNode      = null;
let processorNode   = null;
let recordedBuffers = [];
let recordedLength  = 0;
let sampleRate      = 44100;
let recordingBlob   = null;
let timerInterval   = null;
let startedAt       = null;
let gifDataUrl      = null;

// ── Pipeline step helpers ────────────────────────────────────────────
const stepEls = [1, 2, 3, 4].map(i => document.getElementById(`step${i}`));

function setStep(n) {
  stepEls.forEach((el, i) => {
    el.classList.remove("active", "done");
    if (i + 1 < n)  el.classList.add("done");
    if (i + 1 === n) el.classList.add("active");
  });
}

function markStepDone(n) {
  if (stepEls[n - 1]) stepEls[n - 1].classList.add("done");
}

// ── Error / log helpers ──────────────────────────────────────────────
function showError(msg) {
  logMsg.textContent = msg;
  logBar.classList.remove("hidden");
  setTimeout(() => logBar.classList.add("hidden"), 8000);
}
function clearError() { logBar.classList.add("hidden"); }

function showTiming(parts) {
  timingText.innerHTML = parts.join("&nbsp;&nbsp;|&nbsp;&nbsp;");
  timingBar.classList.remove("hidden");
}

// ── Timer ────────────────────────────────────────────────────────────
function formatTime(s) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}
function startTimer() {
  startedAt = Date.now();
  timerBadge.textContent = "00:00";
  timerBadge.classList.add("recording");
  timerInterval = setInterval(() => {
    timerBadge.textContent = formatTime((Date.now() - startedAt) / 1000);
  }, 250);
}
function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  timerBadge.classList.remove("recording");
}

// ── Health check ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    apiStatus.textContent = `Backend: OK`;
    apiStatus.className = "badge badge-ok";

    const shortModel = (data.asr_model || "").split("/").pop();
    asrBadge.textContent = `ASR: ${shortModel}${data.asr_model_loaded ? " ✓" : ""}`;
    asrBadge.className = data.asr_model_loaded ? "badge badge-ok" : "badge badge-muted";

    lexiconBadge.textContent = `Lexicon: ${data.lexicon_ok ? "OK" : "Missing"}`;
    lexiconBadge.className = data.lexicon_ok ? "badge badge-ok" : "badge badge-error";
  } catch {
    apiStatus.textContent = "Backend: offline";
    apiStatus.className = "badge badge-error";
  }
}

// ── Recording ────────────────────────────────────────────────────────
async function startRecording() {
  clearError();
  timingBar.classList.add("hidden");
  recordedBuffers = [];
  recordedLength  = 0;
  recordingBlob   = null;
  audioWrap.classList.add("hidden");

  try {
    mediaStream  = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sampleRate   = audioContext.sampleRate;

    sourceNode    = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);

    processorNode.onaudioprocess = e => {
      const data = e.inputBuffer.getChannelData(0);
      recordedBuffers.push(new Float32Array(data));
      recordedLength += data.length;
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    startBtn.disabled = true;
    stopBtn.disabled  = false;
    runBtn.disabled   = true;
    startTimer();
    setStep(1);
  } catch (err) {
    showError(`Microphone access denied: ${err.message}`);
  }
}

function mergeBuffers(buffers, length) {
  const out = new Float32Array(length);
  let off = 0;
  for (const buf of buffers) { out.set(buf, off); off += buf.length; }
  return out;
}

function floatTo16Bit(view, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function writeStr(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function encodeWAV(samples, rate) {
  const buf  = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  writeStr(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(view, 8, "WAVE");
  writeStr(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  floatTo16Bit(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function stopRecording() {
  stopTimer();
  if (processorNode) { processorNode.disconnect(); processorNode.onaudioprocess = null; }
  if (sourceNode)    sourceNode.disconnect();
  if (mediaStream)   mediaStream.getTracks().forEach(t => t.stop());
  if (audioContext)  audioContext.close();

  const samples = mergeBuffers(recordedBuffers, recordedLength);
  recordingBlob = encodeWAV(samples, sampleRate);
  audioPlayer.src = URL.createObjectURL(recordingBlob);
  audioWrap.classList.remove("hidden");

  startBtn.disabled = false;
  stopBtn.disabled  = true;
  runBtn.disabled   = false;

  const dur = (recordedLength / sampleRate).toFixed(1);
  timerBadge.textContent = formatTime(recordedLength / sampleRate);
}

// ── File upload / drop ───────────────────────────────────────────────
function loadAudioFile(file) {
  if (!file) return;
  recordingBlob = file;
  audioPlayer.src = URL.createObjectURL(file);
  audioWrap.classList.remove("hidden");
  runBtn.disabled = false;
  clearError();
}

fileInput.addEventListener("change", () => loadAudioFile(fileInput.files[0]));
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  loadAudioFile(e.dataTransfer.files[0]);
});

// ── Render gloss tokens ──────────────────────────────────────────────
function renderGloss(tokens, rawStr) {
  glossTokensEl.innerHTML = "";
  if (!tokens || tokens.length === 0) {
    glossTokensEl.innerHTML = '<span class="gloss-empty">No glosses found for this text.</span>';
    return;
  }
  tokens.forEach((t, i) => {
    const span = document.createElement("span");
    span.className = "gloss-token";
    span.style.animationDelay = `${i * 40}ms`;
    span.textContent = t;
    glossTokensEl.appendChild(span);
  });
  glossRawEl.textContent = rawStr ? `Raw: ${rawStr}` : "";
}

// ── Render GIF ───────────────────────────────────────────────────────
function renderGIF(base64, sizeBytes) {
  gifDataUrl = `data:image/gif;base64,${base64}`;

  gifWrap.innerHTML = "";
  const img = document.createElement("img");
  img.id  = "gifImage";
  img.src = gifDataUrl;
  img.alt = "Sign language animation";
  gifWrap.appendChild(img);

  gifMeta.textContent = `GIF size: ${(sizeBytes / 1024).toFixed(1)} KB`;

  downloadBtn.href = gifDataUrl;
  downloadBtn.removeAttribute("hidden");
}

// ── Full pipeline (single API call) ─────────────────────────────────
async function runFullPipeline() {
  if (!recordingBlob) { showError("No audio. Record or upload a file first."); return; }

  clearError();
  timingBar.classList.add("hidden");
  transcriptBox.value = "";
  asrMeta.textContent = "";
  glossTokensEl.innerHTML = '<span class="gloss-empty">Processing…</span>';
  glossRawEl.textContent  = "";
  gifWrap.innerHTML = '<div class="gif-placeholder"><span class="gif-placeholder-icon">⏳</span><p>Generating…</p></div>';
  gifMeta.textContent = "";
  downloadBtn.setAttribute("hidden", "");

  runBtn.disabled    = true;
  asrSpinner.classList.remove("hidden");
  glossSpinner.classList.remove("hidden");
  gifSpinner.classList.remove("hidden");
  setStep(2);

  const formData = new FormData();
  const audioFile = recordingBlob instanceof File
    ? recordingBlob
    : new File([recordingBlob], "recording.wav", { type: "audio/wav" });
  formData.append("file", audioFile);
  formData.append("glosser", glosserSelect.value);
  formData.append("signed_language", signedLangSelect.value);
  formData.append("gif_width", "400");

  try {
    const res  = await fetch("/api/pipeline", { method: "POST", body: formData });
    const data = await res.json();

    asrSpinner.classList.add("hidden");
    glossSpinner.classList.add("hidden");
    gifSpinner.classList.add("hidden");
    runBtn.disabled = false;

    if (!res.ok) {
      showError(`Pipeline error: ${data.detail || res.status}`);
      setStep(1);
      return;
    }

    // Step 2: ASR
    transcriptBox.value = data.transcript || "";
    asrMeta.textContent =
      `Model: ${(data.asr_model || "").split("/").pop()} | ` +
      `Device: ${data.device} | ` +
      `Audio: ${data.audio_duration_seconds}s | ` +
      `ASR: ${data.asr_inference_seconds}s`;
    markStepDone(2);
    glossBtn.disabled = false;
    setStep(3);

    // Step 3: Gloss
    renderGloss(data.gloss_tokens, data.gloss);
    markStepDone(3);
    setStep(4);

    // Step 4: GIF
    renderGIF(data.gif_base64, data.gif_size_bytes);
    markStepDone(4);

    showTiming([
      `🎙️ Audio: ${data.audio_duration_seconds}s`,
      `🤖 ASR: ${data.asr_inference_seconds}s`,
      `🧩 Gloss+GIF: ${data.gloss_gif_seconds}s`,
      `🌐 Signed: ${data.signed_language}`,
    ]);

    checkHealth();

  } catch (err) {
    asrSpinner.classList.add("hidden");
    glossSpinner.classList.add("hidden");
    gifSpinner.classList.add("hidden");
    runBtn.disabled = false;
    showError(`Network error: ${err.message}`);
    setStep(1);
  }
}

// ── Gloss-only from edited transcript ───────────────────────────────
async function runGlossOnly() {
  const text = transcriptBox.value.trim();
  if (!text) { showError("Transcript is empty."); return; }

  clearError();
  glossBtn.disabled = true;
  glossSpinner.classList.remove("hidden");
  glossTokensEl.innerHTML = '<span class="gloss-empty">Computing gloss…</span>';

  try {
    const res  = await fetch("/api/gif", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        glosser: glosserSelect.value,
        spoken_language: "de",
        signed_language: signedLangSelect.value,
        gif_width: 400,
      }),
    });
    const data = await res.json();

    glossSpinner.classList.add("hidden");
    glossBtn.disabled = false;

    if (!res.ok) { showError(`Gloss error: ${data.detail}`); return; }

    renderGloss(data.gloss_tokens, data.gloss);
    renderGIF(data.gif_base64, data.gif_size_bytes);
  } catch (err) {
    glossSpinner.classList.add("hidden");
    glossBtn.disabled = false;
    showError(`Network error: ${err.message}`);
  }
}

// ── Clear all ────────────────────────────────────────────────────────
function clearAll() {
  recordingBlob   = null;
  recordedBuffers = [];
  recordedLength  = 0;
  gifDataUrl      = null;

  audioPlayer.removeAttribute("src");
  audioWrap.classList.add("hidden");
  fileInput.value = "";

  transcriptBox.value = "";
  asrMeta.textContent = "";

  glossTokensEl.innerHTML = '<span class="gloss-empty">Gloss tokens will appear here…</span>';
  glossRawEl.textContent  = "";

  gifWrap.innerHTML =
    '<div class="gif-placeholder"><span class="gif-placeholder-icon">🤟</span>' +
    '<p>Run the pipeline to generate the sign language animation</p></div>';
  gifMeta.textContent = "";
  downloadBtn.setAttribute("hidden", "");

  runBtn.disabled    = true;
  glossBtn.disabled  = true;
  timerBadge.textContent = "00:00";
  timingBar.classList.add("hidden");
  clearError();

  stepEls.forEach(el => el.classList.remove("active", "done"));
  setStep(1);
}

// ── Event listeners ──────────────────────────────────────────────────
startBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
runBtn.addEventListener("click", runFullPipeline);
clearBtn.addEventListener("click", clearAll);
glossBtn.addEventListener("click", runGlossOnly);

// ── Init ─────────────────────────────────────────────────────────────
setStep(1);
checkHealth();
setInterval(checkHealth, 30_000);
