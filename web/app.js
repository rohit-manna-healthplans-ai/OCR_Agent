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

function getExtractedText(result){
  // Project standard: raw extracted OCR text is result.text
  return (result && typeof result.text === "string") ? result.text : "";
}

function getStructured(result){
  // Project standard: structured JSON is result.structured
  return (result && result.structured) ? result.structured : null;
}

let batchResults = []; // each item: { filename, ...run_ocr_result }

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

function renderSelected(){
  const picker = $("resultPicker");
  const idx = picker && picker.value ? parseInt(picker.value, 10) : 0;
  const r = batchResults[idx] || null;

  const extracted = r ? getExtractedText(r) : "";
  const structured = r ? getStructured(r) : null;

  $("extractedText").textContent = extracted || "";
  $("structuredJson").textContent = structured ? pretty(structured) : "";
  $("rawJson").textContent = r ? pretty(r) : "";

  // wire download buttons for selected
  const rawBtn = $("downloadCurrentRawBtn");
  const structBtn = $("downloadCurrentStructuredBtn");
  if (rawBtn){
    rawBtn.onclick = () => {
      if (!r) return;
      downloadJson((r.filename || "result") + ".raw.json", r);
    };
  }
  if (structBtn){
    structBtn.onclick = () => {
      if (!r) return;
      downloadJson((r.filename || "result") + ".structured.json", structured || {});
    };
  }
}

async function run(){
  const files = $("file").files;
  if (!files || files.length === 0){
    setStatus("Select a file first", "warn");
    return;
  }
  if (files.length > 10){
    setStatus("Batch limit is 10 files. Please select max 10.", "warn");
    return;
  }

  const engine = $("engine").value || "auto";
  const enableOllama = $("enableOllama").checked ? "true" : "false";

  // Decide endpoint based on number of files
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

  const qs = new URLSearchParams({
    engine,
    enable_ollama: enableOllama
  });

  try {
    const res = await fetch(`${endpoint}?${qs.toString()}`, { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok){
      setStatus("Error: " + (data && data.detail ? data.detail : res.statusText), "bad");
      return;
    }

    if (isBatch){
      batchResults = Array.isArray(data.results) ? data.results : [];
      populatePicker();
      renderSelected();
      setStatus(`Done. ${batchResults.length} results.`, "ok");
    } else {
      batchResults = [{ filename: files[0].name, ...data }];
      populatePicker();
      renderSelected();
      setStatus("Done.", "ok");
    }
  } catch (e){
    setStatus("Request failed: " + (e && e.message ? e.message : String(e)), "bad");
  }
}

// Init handlers
document.addEventListener("DOMContentLoaded", () => {
  const runBtn = $("runBtn");
  if (runBtn) runBtn.addEventListener("click", run);

  const picker = $("resultPicker");
  if (picker) picker.addEventListener("change", renderSelected);

  // initial
  batchResults = [];
  populatePicker();
  renderSelected();
  setStatus("Idle", "ok");
});
