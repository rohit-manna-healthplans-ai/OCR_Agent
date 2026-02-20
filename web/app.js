function $(id){ return document.getElementById(id); }

function pretty(obj){
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

function setStatus(text, kind){
  const el = $("status");
  if (!el) return;
  el.textContent = text || "Idle";
  el.style.background = "rgba(255,255,255,0.06)";
  if (kind === "ok") el.style.background = "rgba(16,185,129,0.14)";
  if (kind === "warn") el.style.background = "rgba(245,158,11,0.14)";
  if (kind === "bad") el.style.background = "rgba(239,68,68,0.14)";
}

function downloadJson(filename, obj){
  const blob = new Blob([pretty(obj)], {type:"application/json;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

let batchResults = []; // each: { filename, formatted_text, raw_json, llm_json }
let lastSelectedFiles = []; // File[] from input at run-time
let filePreviewUrls = new Map(); // filename -> objectURL (images only)

function isImageFile(file){
  const t = (file && file.type) ? file.type.toLowerCase() : "";
  if (t.startsWith("image/")) return true;
  const name = (file && file.name) ? file.name.toLowerCase() : "";
  return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".webp");
}

function revokePreviewUrls(){
  for (const url of filePreviewUrls.values()){
    try { URL.revokeObjectURL(url); } catch {}
  }
  filePreviewUrls.clear();
}

function buildPreviewUrls(files){
  revokePreviewUrls();
  for (const f of files){
    if (isImageFile(f)){
      filePreviewUrls.set(f.name, URL.createObjectURL(f));
    }
  }
}

function populatePicker(){
  const picker = $("resultPicker");
  if (!picker) return;
  picker.innerHTML = "";
  if (!batchResults.length){
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no results)";
    picker.appendChild(opt);
    picker.disabled = true;
    return;
  }
  picker.disabled = false;
  batchResults.forEach((r, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = r.filename || `file_${idx+1}`;
    picker.appendChild(opt);
  });
  picker.value = "0";
}

function renderImagesAll(){
  const grid = $("imagesGrid");
  if (!grid) return;
  grid.innerHTML = "";

  const files = lastSelectedFiles || [];
  if (!files.length){
    const div = document.createElement("div");
    div.className = "muted";
    div.textContent = "No files selected.";
    grid.appendChild(div);
    return;
  }

  for (const f of files){
    const card = document.createElement("div");
    card.className = "thumb";

    const title = document.createElement("div");
    title.className = "thumbTitle";
    title.textContent = f.name;
    card.appendChild(title);

    if (isImageFile(f)){
      const img = document.createElement("img");
      img.className = "thumbImg";
      img.src = filePreviewUrls.get(f.name) || "";
      img.alt = f.name;
      card.appendChild(img);
    } else {
      const p = document.createElement("div");
      p.className = "muted";
      p.textContent = "Preview not available for this file type (e.g., PDF).";
      card.appendChild(p);
    }

    grid.appendChild(card);
  }
}

function renderSelected(){
  const picker = $("resultPicker");
  const idx = picker && picker.value ? parseInt(picker.value, 10) : 0;
  const r = batchResults[idx] || null;

  $("extractedText").textContent = r ? (r.formatted_text || "") : "";
  $("rawJson").textContent = r ? pretty(r.raw_json || {}) : "";
  $("llmJson").textContent = r ? pretty(r.llm_json || {}) : "";

  const rawBtn = $("downloadRawBtn");
  if (rawBtn){
    rawBtn.onclick = () => {
      if (!r) return;
      downloadJson((r.filename || "result") + ".raw.json", r.raw_json || {});
    };
  }

  // Images tab shows ALL selected files always (batch-friendly)
  renderImagesAll();
}

function activateTab(tabId){
  const panes = document.querySelectorAll('.tabpane');
  panes.forEach(p => p.classList.add('hidden'));

  const btns = document.querySelectorAll('.tabbtn');
  btns.forEach(b => b.classList.remove('active'));

  const pane = document.getElementById(tabId);
  if (pane) pane.classList.remove('hidden');

  const btn = document.querySelector(`.tabbtn[data-tab="${tabId}"]`);
  if (btn) btn.classList.add('active');
}

async function run(){
  const files = $("file").files;
  if (!files || files.length === 0){
    setStatus("Select files first", "warn");
    return;
  }
  if (files.length > 10){
    setStatus("Batch limit is 10 files. Please select max 10.", "warn");
    return;
  }

  lastSelectedFiles = Array.from(files);
  buildPreviewUrls(lastSelectedFiles);

  const engine = $("engine").value || "auto";
  const isBatch = files.length > 1;
  const endpoint = isBatch ? "/ocr-batch" : "/ocr";

  const fd = new FormData();
  if (isBatch){
    for (const f of files){
      fd.append("files", f);
    }
  } else {
    fd.append("file", files[0]);
  }

  setStatus(isBatch ? `Running batch OCR (${files.length} files)...` : "Running OCR...", "warn");

  const qs = new URLSearchParams({ engine });

  try {
    const res = await fetch(`${endpoint}?${qs.toString()}`, { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok){
      setStatus("Error: " + (data && data.detail ? data.detail : res.statusText), "bad");
      return;
    }

    if (isBatch){
      batchResults = Array.isArray(data.results) ? data.results : [];
    } else {
      batchResults = [data];
    }

    populatePicker();
    renderSelected();
    setStatus(`Done. ${batchResults.length} result(s).`, "ok");

    // After run, default to Extracted Text tab
    activateTab('tabText');
  } catch (e){
    setStatus("Request failed: " + (e && e.message ? e.message : String(e)), "bad");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const runBtn = $("runBtn");
  if (runBtn) runBtn.addEventListener("click", run);

  const picker = $("resultPicker");
  if (picker) picker.addEventListener("change", renderSelected);

  document.querySelectorAll('.tabbtn').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.getAttribute('data-tab')));
  });

  batchResults = [];
  populatePicker();
  renderSelected();
  setStatus("Idle", "ok");
  activateTab('tabText');
});
