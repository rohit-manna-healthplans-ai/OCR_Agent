
/* CLEAN WORKING VERSION - Compatible with current index.html */

function $(id) { return document.getElementById(id); }

function setStatus(msg) {
  const el = $("status");
  if (el) el.textContent = msg;
}

function prettyJson(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

function clearOutputs() {
  if ($("fields")) $("fields").textContent = "";
  if ($("text")) $("text").textContent = "";
  if ($("analysis")) $("analysis").textContent = "";
  if ($("batch_results")) $("batch_results").innerHTML = "";
}

/* ================= SINGLE OCR ================= */

async function runSingleOCR() {
  const fileInput = $("files");
  if (!fileInput || fileInput.files.length === 0) {
    alert("Please select at least 1 file.");
    return;
  }

  const engine = $("engine") ? $("engine").value : "auto";
  const url = `/ocr?engine=${encodeURIComponent(engine)}`;

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  clearOutputs();
  setStatus("Running OCR...");

  try {
    const res = await fetch(url, { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok) {
      setStatus("Error");
      console.error(data);
      return;
    }

    $("fields").textContent = prettyJson(data.fields || {});
    $("text").textContent = data.text || "";
    $("analysis").textContent = prettyJson(data.analysis || {});
    setStatus("Done ✅");

  } catch (err) {
    console.error(err);
    setStatus("Error ❌");
  }
}

/* ================= BATCH OCR ================= */

function renderBatchResults(data) {
  const container = $("batch_results");
  if (!container) return;

  container.innerHTML = "";

  if (!data.results || data.results.length === 0) {
    container.innerHTML = "<p>No results found.</p>";
    return;
  }

  data.results.forEach((doc, index) => {
    const card = document.createElement("div");
    card.className = "batch-card";

    const title = document.createElement("h3");
    title.textContent = doc.filename || `Document ${index + 1}`;

    const meta = document.createElement("div");
    meta.innerHTML = `
      <strong>Status:</strong> ${doc.status || "completed"} |
      <strong>Text Length:</strong> ${(doc.text || "").length}
    `;

    card.appendChild(title);
    card.appendChild(meta);

    card.addEventListener("click", () => {
      $("fields").textContent = prettyJson(doc.fields || {});
      $("text").textContent = doc.text || "";
      $("analysis").textContent = prettyJson(doc.analysis || {});
    });

    container.appendChild(card);
  });

  // auto show first
  const first = data.results[0];
  $("fields").textContent = prettyJson(first.fields || {});
  $("text").textContent = first.text || "";
  $("analysis").textContent = prettyJson(first.analysis || {});
}

async function runBatchOCR() {
  const fileInput = $("files");
  if (!fileInput || fileInput.files.length === 0) {
    alert("Please select at least 1 file.");
    return;
  }

  const engine = $("engine") ? $("engine").value : "auto";
  const url = `/ocr-batch?engine=${encodeURIComponent(engine)}`;

  const fd = new FormData();
  for (const f of fileInput.files) {
    fd.append("files", f);
  }

  clearOutputs();
  setStatus(`Running batch OCR (${fileInput.files.length} files)...`);

  try {
    const res = await fetch(url, { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok) {
      setStatus("Error");
      console.error(data);
      return;
    }

    renderBatchResults(data);
    setStatus("Batch Done ✅");

  } catch (err) {
    console.error(err);
    setStatus("Error ❌");
  }
}

/* ================= DOCX ================= */

async function downloadDocx() {
  const fileInput = $("files");
  if (!fileInput || fileInput.files.length === 0) {
    alert("Please select at least 1 file.");
    return;
  }

  const engine = $("engine") ? $("engine").value : "auto";
  const url = `/ocr-docx?engine=${encodeURIComponent(engine)}`;

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  setStatus("Generating DOCX...");

  try {
    const res = await fetch(url, { method: "POST", body: fd });

    if (!res.ok) {
      const err = await res.text();
      alert("DOCX error: " + err);
      setStatus("Error ❌");
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

  } catch (err) {
    console.error(err);
    setStatus("Error ❌");
  }
}

/* ================= INIT ================= */

window.addEventListener("DOMContentLoaded", () => {
  if ($("run")) $("run").addEventListener("click", runSingleOCR);
  if ($("batch")) $("batch").addEventListener("click", runBatchOCR);
  if ($("docx")) $("docx").addEventListener("click", downloadDocx);
});
