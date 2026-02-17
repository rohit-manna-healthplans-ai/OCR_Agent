/* Frontend logic (no bundler). Fixes duplicate variable declarations and keeps buttons clickable. */

function $(id) { return document.getElementById(id); }

function setStatus(msg) {
  const el = $("status");
  if (el) el.textContent = msg;
}

function prettyJson(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

function buildQuery(params) {
  const esc = encodeURIComponent;
  return Object.keys(params)
    .map(k => `${esc(k)}=${esc(params[k])}`)
    .join("&");
}

async function runSingleOCR() {
  const files = $("files").files;
  if (!files || files.length === 0) {
    alert("Please select at least 1 file.");
    return;
  }

  const engine = $("engine").value;
  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;
  const returnLayout = $("return_layout").value;
  const debug = $("debug").value;

  const qs = buildQuery({
    engine,
    preset,
    dpi,
    max_pages: maxPages,
    return_debug: debug,
    return_layout: returnLayout
  });

  const url = `/ocr?${qs}`;

  const fd = new FormData();
  fd.append("file", files[0]);

  $("raw").textContent = "";
  $("fields").textContent = "";
  $("text").textContent = "";
  $("analysis").textContent = "";
  setStatus("Running OCR...");

  const res = await fetch(url, { method: "POST", body: fd });
  const data = await res.json();

  if (!res.ok) {
    setStatus("Error");
    $("raw").textContent = prettyJson(data);
    return;
  }

  $("raw").textContent = prettyJson(data);
  $("fields").textContent = prettyJson(data.fields || {});
  $("text").textContent = (data.text || "");
  $("analysis").textContent = prettyJson(data.analysis || {});
  setStatus("Done ✅");
}

async function runBatchOCR() {
  const files = $("files").files;
  if (!files || files.length === 0) {
    alert("Please select at least 1 file.");
    return;
  }
  if (files.length > 10) {
    alert("Batch limit is 10 files. Please select max 10 files.");
    return;
  }

  const engine = $("engine").value;
  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;
  const returnLayout = $("return_layout").value;
  const debug = $("debug").value;

  const qs = buildQuery({
    engine,
    preset,
    dpi,
    max_pages: maxPages,
    return_debug: debug,
    return_layout: returnLayout
  });

  const url = `/ocr-batch?${qs}`;

  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  $("raw").textContent = "";
  $("fields").textContent = "";
  $("text").textContent = "";
  $("analysis").textContent = "";
  setStatus(`Running batch OCR (${files.length} files)...`);

  const res = await fetch(url, { method: "POST", body: fd });
  const data = await res.json();

  if (!res.ok) {
    setStatus("Error");
    $("raw").textContent = prettyJson(data);
    return;
  }

  $("raw").textContent = prettyJson(data);

  // show first file highlights
  const first = (data.results && data.results[0]) ? data.results[0] : null;
  if (first) {
    $("fields").textContent = prettyJson(first.fields || {});
    $("text").textContent = (first.text || "");
    $("analysis").textContent = prettyJson(first.analysis || {});
  }
  setStatus("Batch Done ✅");
}

async function downloadDocx() {
  const files = $("files").files;
  if (!files || files.length === 0) {
    alert("Please select at least 1 file.");
    return;
  }

  const engine = $("engine").value;
  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;

  const qs = buildQuery({ engine, preset, dpi, max_pages: maxPages });
  const url = `/ocr-docx?${qs}`;

  const fd = new FormData();
  fd.append("file", files[0]);

  setStatus("Generating DOCX...");
  const res = await fetch(url, { method: "POST", body: fd });

  if (!res.ok) {
    const err = await res.text();
    alert("DOCX error: " + err);
    setStatus("Error");
    return;
  }

  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ocr_result.docx";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setStatus("DOCX downloaded ✅");
}

window.addEventListener("DOMContentLoaded", () => {
  $("run").addEventListener("click", runSingleOCR);
  $("batch").addEventListener("click", runBatchOCR);
  $("docx").addEventListener("click", downloadDocx);
});
