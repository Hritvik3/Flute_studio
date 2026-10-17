/* Flute Studio frontend */

"use strict";

// ============================================================
// helpers
// ============================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = body.detail || JSON.stringify(body);
    } catch (_) { /* keep statusText */ }
    throw new Error(msg);
  }
  return res.json();
}

const fmtDur = (s) => {
  if (s == null) return "";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
};

const fmtSize = (b) => {
  if (!b) return "";
  if (b > 1024 * 1024) return `${(b / 1048576).toFixed(1)} MB`;
  return `${Math.round(b / 1024)} KB`;
};

function toast(message, isError = false) {
  const el = document.createElement("div");
  el.textContent = message;
  el.style.cssText = `
    position:fixed; bottom:20px; right:20px; z-index:99;
    background:${isError ? "#ff6b6b" : "#4cc9a6"};
    color:#06281f; font-weight:600; padding:10px 16px;
    border-radius:8px; font-size:13px; box-shadow:0 4px 14px rgba(0,0,0,.5)`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ============================================================
// state
// ============================================================

const state = {
  library: [],
  models: [],
  styles: {},
  defaults: {},
  device: "?",
  mixTracks: [],
  stemsets: [],
};

// ============================================================
// tabs
// ============================================================

$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    $$(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

function goToTab(name) {
  const btn = $$(".tab").find((b) => b.dataset.tab === name);
  if (btn) btn.click();
}

// ============================================================
// library
// ============================================================

async function refreshLibrary() {
  state.library = await api("/api/library");
  state.stemsets = state.library.filter((f) => f.kind === "stemset");
  $("#libraryCount").textContent = `(${state.library.length})`;
  renderLibraryList();
  populateSelects();
}

function kindLabel(kind) {
  return { upload: "upload", stem: "stem", stemset: "stems", mix: "mix", master: "master" }[kind] || kind;
}

function renderLibraryList() {
  const list = $("#libraryList");
  if (!state.library.length) {
    list.innerHTML = `<p class="empty">No files yet — upload something above.</p>`;
    return;
  }
  list.innerHTML = "";
  state.library.forEach((f) => {
    const row = document.createElement("div");
    row.className = "file-item";
    row.innerHTML = `
      <span class="badge ${f.kind}">${kindLabel(f.kind)}</span>
      <span class="name" title="${f.name}">${f.name}</span>
      <span class="meta">${fmtSize(f.size_bytes)}</span>
      <audio controls preload="none" src="${f.url}"></audio>
      <button class="small" data-act="dl">⬇</button>
      <button class="small danger" data-act="del">✕</button>`;

    row.querySelector('[data-act="dl"]').addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = f.url;
      a.download = f.name;
      a.click();
    });
    row.querySelector('[data-act="del"]').addEventListener("click", async () => {
      if (!confirm(`Delete "${f.name}"?`)) return;
      await api(`/api/library/${f.id}`, { method: "DELETE" });
      toast("Deleted");
      refreshLibrary();
    });
    list.appendChild(row);
  });
}

function populateSelects() {
  // Separate source: uploads + mixes + masters (anything non-stem)
  const sources = state.library.filter((f) => f.kind !== "stem" && f.kind !== "stemset");
  fillSelect($("#sepSource"), sources, "Separate a track…");
  fillSelect($("#masterSource"), sources, "Master a track…");
  fillSelect($("#masterRef"), sources, "No reference", true);

  // Mixer source: everything playable
  const mixables = state.library.filter((f) => f.kind !== "stemset");
  fillSelect($("#mixSource"), mixables, "Choose a track or stem…");
}

function fillSelect(sel, items, placeholder, allowNone = false) {
  const current = sel.value;
  sel.innerHTML = "";
  if (placeholder) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = placeholder;
    sel.appendChild(o);
  }
  items.forEach((f) => {
    const o = document.createElement("option");
    o.value = f.id;
    const tag = f.kind !== "upload" ? ` [${kindLabel(f.kind)}]` : "";
    o.textContent = `${f.name}${tag}`;
    sel.appendChild(o);
  });
  if (allowNone) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = placeholder;
    sel.appendChild(o);
  }
  if (current && items.some((i) => i.id === current)) sel.value = current;
}

// ============================================================
// upload
// ============================================================

const dropzone = $("#dropzone");
const fileInput = $("#fileInput");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => {
  uploadFiles(fileInput.files);
  fileInput.value = "";
});

async function uploadFiles(fileList) {
  const status = $("#uploadStatus");
  for (const file of fileList) {
    status.textContent = `Uploading ${file.name}…`;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || res.statusText);
      }
      const rec = await res.json();
      const a = rec.analysis || {};
      toast(
        `Uploaded ${rec.name}` +
          (a.duration ? ` (${fmtDur(a.duration)}, ${a.integrated_lufs} LUFS)` : "")
      );
    } catch (err) {
      toast(`${file.name}: ${err.message}`, true);
    }
  }
  status.textContent = "";
  refreshLibrary();
}

// ============================================================
// separation
// ============================================================

function setupSeparation() {
  const modelSel = $("#sepModel");
  modelSel.innerHTML = "";
  state.models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.label;
    modelSel.appendChild(o);
  });

  $("#sepRun").addEventListener("click", async () => {
    const fileId = $("#sepSource").value;
    if (!fileId) return toast("Choose a source track first", true);
    try {
      const job = await api("/api/separate", {
        method: "POST",
        body: JSON.stringify({ file_id: fileId, model: modelSel.value }),
      });
      $("#sepResults").classList.add("hidden");
      trackJob(job.id, {
        wrap: "#sepProgressWrap",
        bar: "#sepBar",
        stage: "#sepStage",
        button: "#sepRun",
        onDone: (j) => {
          renderStemResults(j.result.stemset);
          refreshLibrary();
        },
      });
    } catch (err) {
      toast(err.message, true);
    }
  });

  $("#sepToMix").addEventListener("click", () => {
    const set = state.lastStemset;
    if (!set) return;
    set.stems.forEach((s) => addMixTrack(s.id, s.name));
    goToTab("mix");
    toast("Stems added to mixer");
  });
}

function renderStemResults(stemset) {
  state.lastStemset = stemset;
  $("#sepResults").classList.remove("hidden");
  $("#sepResultMeta").textContent =
    `${stemset.name} — ${stemset.model}, ${fmtDur(stemset.duration)}, ` +
    `${stemset.sample_rate} Hz · ${stemset.stems.length} stems`;
  const grid = $("#sepStems");
  grid.innerHTML = "";
  stemset.stems.forEach((s) => {
    const card = document.createElement("div");
    card.className = "stem-card";
    card.innerHTML = `
      <div class="name">${s.name}</div>
      <audio controls preload="none" src="${s.url}"></audio>`;
    grid.appendChild(card);
  });
}

// ============================================================
// mixer
// ============================================================

function addMixTrack(fileId, label) {
  const rec = state.library.find((f) => f.id === fileId);
  state.mixTracks.push({
    file_id: fileId,
    name: label || (rec ? rec.name : fileId),
    gain_db: 0,
    pan: 0,
    mute: false,
    solo: false,
  });
  renderMixTracks();
}

function renderMixTracks() {
  const wrap = $("#mixTracks");
  wrap.innerHTML = "";
  if (!state.mixTracks.length) {
    wrap.innerHTML = `<p class="empty">No tracks — add some above.</p>`;
    return;
  }
  state.mixTracks.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "track";
    row.innerHTML = `
      <span class="idx">${i + 1}</span>
      <span class="tname" title="${t.name}">${t.name}</span>
      <label class="tctl">
        <span>Gain <b>${t.gain_db.toFixed(1)} dB</b></span>
        <input type="range" min="-40" max="12" step="0.5" value="${t.gain_db}">
      </label>
      <label class="tctl">
        <span>Pan <b>${t.pan === 0 ? "C" : t.pan < 0 ? `L${Math.round(-t.pan * 100)}` : `R${Math.round(t.pan * 100)}`}</b></span>
        <input type="range" min="-1" max="1" step="0.01" value="${t.pan}">
      </label>
      <button title="Mute" class="${t.mute ? "on-m" : ""}">M</button>
      <button title="Solo" class="${t.solo ? "on-s" : ""}">S</button>
      <button title="Remove" class="rm">✕</button>`;

    const [gainRange] = row.querySelectorAll('input[type="range"]');
    gainRange.addEventListener("input", () => {
      t.gain_db = parseFloat(gainRange.value);
      renderMixTracks();
    });
    const panRange = row.querySelectorAll('input[type="range"]')[1];
    panRange.addEventListener("input", () => {
      t.pan = parseFloat(panRange.value);
      renderMixTracks();
    });
    const [muteBtn, soloBtn, rmBtn] = row.querySelectorAll("button");
    muteBtn.addEventListener("click", () => { t.mute = !t.mute; renderMixTracks(); });
    soloBtn.addEventListener("click", () => { t.solo = !t.solo; renderMixTracks(); });
    rmBtn.addEventListener("click", () => {
      state.mixTracks.splice(i, 1);
      renderMixTracks();
    });
    wrap.appendChild(row);
  });
}

function setupMixer() {
  $("#mixAdd").addEventListener("click", () => {
    const id = $("#mixSource").value;
    if (!id) return toast("Choose a track first", true);
    addMixTrack(id);
  });

  bindSlider("#mixGain", "#mixGainVal", (v) => `${parseFloat(v).toFixed(1)} dB`);
  bindSlider("#mixCeil", "#mixCeilVal", (v) => `${parseFloat(v).toFixed(1)} dB`);
  const lufsOn = $("#mixLufsOn");
  const lufsVal = $("#mixLufsVal");
  $("#mixLufs").addEventListener("input", () => {
    lufsVal.textContent = lufsOn.checked ? $("#mixLufs").value : "off";
  });
  lufsOn.addEventListener("change", () => {
    lufsVal.textContent = lufsOn.checked ? $("#mixLufs").value : "off";
  });

  $("#mixRun").addEventListener("click", async () => {
    if (!state.mixTracks.length) return toast("Add tracks to the mixer first", true);
    try {
      const body = {
        tracks: state.mixTracks.map((t) => ({
          file_id: t.file_id,
          gain_db: t.gain_db,
          pan: t.pan,
          mute: t.mute,
          solo: t.solo,
        })),
        master_gain_db: parseFloat($("#mixGain").value),
        target_lufs: lufsOn.checked ? parseFloat($("#mixLufs").value) : null,
        ceiling_db: parseFloat($("#mixCeil").value),
        limiter: true,
        output_name: $("#mixName").value || null,
      };
      const job = await api("/api/mix", {
        method: "POST",
        body: JSON.stringify(body),
      });
      trackJob(job.id, {
        wrap: "#mixProgressWrap",
        bar: "#mixBar",
        stage: "#mixStage",
        button: "#mixRun",
        onDone: (j) => {
          renderMixResult(j.result.mix, j.result.report);
          refreshLibrary();
        },
      });
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function renderMixResult(mix, report) {
  const el = $("#mixResult");
  el.classList.remove("hidden");
  const a = report.after || {};
  $("#mixResultBody").innerHTML = `
    <p class="hint">${mix.name}.wav · ${report.tracks_used} tracks ·
       ${fmtDur(report.duration)} @ ${report.sample_rate} Hz</p>
    <audio class="result-audio" controls src="${mix.url}"></audio>
    <p><a class="ghost" style="display:inline-block;text-decoration:none;padding:8px 16px;border:1px solid var(--border);border-radius:8px;font-size:13px" href="${mix.url}" download>⬇ Download mix</a></p>
    ${reportTable([
      ["Integrated loudness", a.integrated_lufs, "LUFS"],
      ["Peak", a.peak_db, "dB"],
      ["RMS", a.rms_db, "dB"],
      ["Crest factor", a.crest_factor_db, "dB"],
      ["Dynamic range", a.dynamics_range, "dB"],
      ["Spectral centroid", a.centroid_hz, "Hz"],
      ["Stereo width", a.stereo_width, ""],
    ])}`;
}

// ============================================================
// mastering
// ============================================================

function setupMastering() {
  const styleSel = $("#masterStyle");
  styleSel.innerHTML = "";
  Object.keys(state.styles).forEach((name) => {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name.replace(/_/g, " ");
    styleSel.appendChild(o);
  });
  styleSel.value = "studio";

  bindSlider("#masterMatch", "#masterMatchVal", (v) => `${v}%`);
  bindSlider("#masterWidth", "#masterWidthVal", (v) => parseFloat(v).toFixed(2));
  bindSlider("#masterLufs", "#masterLufsVal", (v) => parseFloat(v).toFixed(1));
  bindSlider("#masterCeil", "#masterCeilVal", (v) => `${parseFloat(v).toFixed(1)} dB`);

  const revSlider = $("#masterRev"), revVal = $("#masterRevVal"), revAuto = $("#masterRevAuto");
  const delSlider = $("#masterDel"), delVal = $("#masterDelVal"), delAuto = $("#masterDelAuto");

  function applyStyleDefaults() {
    const s = state.styles[styleSel.value];
    if (!s) return;
    if (revAuto.checked) {
      revSlider.value = s.reverb;
      revVal.textContent = `${s.reverb}%`;
    }
    if (delAuto.checked) {
      delSlider.value = s.delay;
      delVal.textContent = `${s.delay}%`;
    }
  }
  styleSel.addEventListener("change", applyStyleDefaults);

  revAuto.addEventListener("change", () => {
    revVal.textContent = revAuto.checked ? "style" : `${revSlider.value}%`;
    if (revAuto.checked) applyStyleDefaults();
  });
  delAuto.addEventListener("change", () => {
    delVal.textContent = delAuto.checked ? "style" : `${delSlider.value}%`;
    if (delAuto.checked) applyStyleDefaults();
  });
  revSlider.addEventListener("input", () => {
    if (revAuto.checked) { revAuto.checked = false; }
    revVal.textContent = `${revSlider.value}%`;
  });
  delSlider.addEventListener("input", () => {
    if (delAuto.checked) { delAuto.checked = false; }
    delVal.textContent = `${delSlider.value}%`;
  });
  applyStyleDefaults();

  $("#masterRun").addEventListener("click", async () => {
    const fileId = $("#masterSource").value;
    if (!fileId) return toast("Choose a source track first", true);

    const toggles = {};
    $$("#masterToggles input").forEach((cb) => {
      toggles[cb.dataset.fx] = cb.checked;
    });

    const body = {
      file_id: fileId,
      style: styleSel.value,
      reference_id: $("#masterRef").value || null,
      match_percent: parseFloat($("#masterMatch").value),
      target_lufs: parseFloat($("#masterLufs").value),
      stereo_width: parseFloat($("#masterWidth").value),
      reverb_percent: revAuto.checked ? null : parseFloat(revSlider.value),
      delay_percent: delAuto.checked ? null : parseFloat(delSlider.value),
      ceiling_db: parseFloat($("#masterCeil").value),
      output_name: $("#masterName").value || null,
      ...toggles,
    };

    try {
      const job = await api("/api/master", {
        method: "POST",
        body: JSON.stringify(body),
      });
      trackJob(job.id, {
        wrap: "#masterProgressWrap",
        bar: "#masterBar",
        stage: "#masterStage",
        button: "#masterRun",
        onDone: (j) => {
          renderMasterResult(j.result.master, j.result.report);
          refreshLibrary();
        },
      });
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function renderMasterResult(master, report) {
  const el = $("#masterResult");
  el.classList.remove("hidden");
  const before = report.before || {};
  const after = report.after || {};
  const fx = Object.entries(report.effects || {})
    .filter(([, on]) => on)
    .map(([k]) => k)
    .join(" · ");

  $("#masterResultBody").innerHTML = `
    <p class="hint">${master.name}.wav · style <b style="color:var(--accent)">${report.style}</b>
      · effects: ${fx || "none"}
      · reverb ${report.reverb_percent}% · delay ${report.delay_percent}%
      · width ${report.stereo_width.toFixed(2)}</p>
    <audio class="result-audio" controls src="${master.url}"></audio>
    <p><a class="ghost" style="display:inline-block;text-decoration:none;padding:8px 16px;border:1px solid var(--border);border-radius:8px;font-size:13px" href="${master.url}" download>⬇ Download master</a></p>
    <table class="report-table">
      <tr><th>Measure</th><th>Before</th><th>After</th><th>Δ</th></tr>
      ${deltaRow("Integrated loudness (LUFS)", before.integrated_lufs, after.integrated_lufs)}
      ${deltaRow("Peak (dB)", before.peak_db, after.peak_db)}
      ${deltaRow("RMS (dB)", before.rms_db, after.rms_db)}
      ${deltaRow("Crest factor (dB)", before.crest_factor_db, after.crest_factor_db)}
      ${deltaRow("Dynamic range (dB)", before.dynamics_range, after.dynamics_range)}
      ${deltaRow("Spectral centroid (Hz)", before.centroid_hz, after.centroid_hz)}
      ${deltaRow("Stereo width", before.stereo_width, after.stereo_width)}
    </table>
    ${report.band_corrections ? bandTable(report.band_corrections) : ""}`;
}

function deltaRow(label, b, a) {
  if (b == null || a == null) return "";
  const d = a - b;
  const cls = d > 0.01 ? "delta-up" : d < -0.01 ? "delta-down" : "";
  return `<tr><td>${label}</td><td>${fmt(b)}</td><td>${fmt(a)}</td>
    <td class="${cls}">${d >= 0 ? "+" : ""}${fmt(d)}</td></tr>`;
}

function bandTable(bands) {
  const rows = Object.entries(bands)
    .map(
      ([band, v]) =>
        `<tr><td>${band.replace(/_/g, " ")}</td><td colspan="3">${v >= 0 ? "+" : ""}${v.toFixed(2)} dB</td></tr>`
    )
    .join("");
  return `<p class="hint" style="margin-top:12px">Reference band corrections applied:</p>
    <table class="report-table"><tr><th>Band</th><th>Correction</th></tr>${rows}</table>`;
}

function reportTable(rows) {
  return `<table class="report-table">
    <tr><th>Measure</th><th>Value</th><th></th></tr>
    ${rows.map(([l, v, u]) => `<tr><td>${l}</td><td>${fmt(v)}</td><td>${u}</td></tr>`).join("")}
  </table>`;
}

const fmt = (v) => (v == null ? "—" : typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)) : v);

// ============================================================
// job tracking
// ============================================================

async function trackJob(jobId, opts) {
  const wrap = $(opts.wrap), bar = $(opts.bar), stage = $(opts.stage);
  const button = opts.button ? $(opts.button) : null;
  wrap.classList.remove("hidden");
  if (button) button.disabled = true;

  const timer = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (_) {
      return;
    }
    bar.style.width = `${Math.round(job.progress * 100)}%`;
    stage.textContent = `${job.stage} — ${Math.round(job.progress * 100)}%`;

    if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
      clearInterval(timer);
      if (button) button.disabled = false;
      if (job.status === "done") {
        toast("Job complete");
        if (opts.onDone) opts.onDone(job);
        setTimeout(() => wrap.classList.add("hidden"), 1200);
      } else if (job.status === "error") {
        toast(job.error || "Job failed", true);
        stage.textContent = job.error || "Failed";
      } else {
        stage.textContent = "Cancelled";
      }
      refreshJobs();
    }
  }, 700);
}

async function refreshJobs() {
  const jobs = await api("/api/jobs");
  const wrap = $("#jobsList");
  if (!jobs.length) {
    wrap.innerHTML = `<p class="empty">No jobs yet.</p>`;
    return;
  }
  wrap.innerHTML = "";
  jobs.forEach((j) => {
    const el = document.createElement("div");
    el.className = "job";
    el.innerHTML = `
      <div class="job-head">
        <span class="badge">${j.kind}</span>
        <span class="title">${j.title}</span>
        <span class="status ${j.status}">${j.status}</span>
        ${j.status === "running" || j.status === "queued"
          ? `<button class="small danger" data-act="cancel">Cancel</button>` : ""}
      </div>
      <div class="progress" style="margin-top:8px"><div class="bar" style="width:${Math.round(j.progress * 100)}%"></div></div>
      ${j.error ? `<p class="hint" style="color:var(--danger)">${j.error}</p>` : ""}
      <details><summary class="hint" style="cursor:pointer">log</summary>
        <div class="log">${(j.log || []).join("\n")}</div></details>`;

    const cancelBtn = el.querySelector('[data-act="cancel"]');
    if (cancelBtn) {
      cancelBtn.addEventListener("click", () =>
        api(`/api/jobs/${j.id}/cancel`, { method: "POST" }).then(refreshJobs)
      );
    }
    wrap.appendChild(el);
  });
}

setInterval(() => {
  if ($("#tab-jobs").classList.contains("active")) refreshJobs();
}, 2500);

// ============================================================
// misc
// ============================================================

function bindSlider(sliderSel, labelSel, fmtFn) {
  const slider = $(sliderSel), label = $(labelSel);
  slider.addEventListener("input", () => {
    label.textContent = fmtFn(slider.value);
  });
}

// ============================================================
// boot
// ============================================================

async function boot() {
  const health = await api("/api/health");
  state.models = health.models;
  state.styles = health.styles;
  state.defaults = health.defaults;
  state.device = health.device;
  $("#devicePill").textContent =
    health.device === "cuda" ? "⚡ GPU (CUDA)" : "🖥 CPU";

  setupSeparation();
  setupMixer();
  setupMastering();
  await refreshLibrary();
  refreshJobs();
}

boot().catch((err) => toast(`Startup failed: ${err.message}`, true));
