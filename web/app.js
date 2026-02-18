/* WORKING UI + TABLE + JSON VIEW + JSON DOWNLOAD (Batch + Single) */

function $(id) { return document.getElementById(id); }

function setStatus(msg) {
  const el = $("status");
  if (el) el.textContent = msg;
}

function prettyJson(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

let __currentDoc = null;
let __currentBatch = null;

function renderJsonPanel(obj) {
  const el = $("json");
  if (!el) return;
  el.textContent = prettyJson(obj || {});
}

function downloadJson(filename, obj) {
  const blob = new Blob([prettyJson(obj)], { type: "application/json;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function downloadCurrentJson() {
  if (__currentDoc) {
    const base = (__currentDoc.filename ? __currentDoc.filename : "ocr_result");
    const safe = base.replace(/[^a-z0-9._-]/gi, "_");
    downloadJson(safe + ".json", __currentDoc);
    return;
  }
  if (__currentBatch) {
    downloadJson("batch_results.json", __currentBatch);
    return;
  }
  alert("No JSON available yet. Run OCR first.");
}

function clearOutputs() {
  if ($("fields")) $("fields").textContent = "";
  if ($("text")) $("text").textContent = "";
  if ($("analysis")) $("analysis").textContent = "";
  if ($("json")) $("json").textContent = "";
  if ($("tables")) $("tables").innerHTML = '<div class="muted">No tables extracted yet.</div>';
  if ($("batch_results")) $("batch_results").innerHTML = "";
  __currentDoc = null;
  __currentBatch = null;
}

function renderTables(doc) {
  const wrap = $("tables");
  if (!wrap) return;

  const pages = (doc && doc.pages) ? doc.pages : [];
  const tableBlocks = [];

  pages.forEach((p, pageIndex) => {
    const tables = p.tables || [];
    tables.forEach((t, ti) => {
      tableBlocks.push({ pageIndex, t, ti });
    });
  });

  if (tableBlocks.length === 0) {
    wrap.innerHTML = '<div class="muted">No tables detected in this document.</div>';
    return;
  }

  const htmlParts = [];

  for (const blk of tableBlocks) {
    const t = blk.t || {};
    const cells = t.cells || [];
    const nRows = t.n_rows || cells.length || 0;
    const nCols = t.n_cols || (cells[0] ? cells[0].length : 0);
    const bbox = t.bbox ? t.bbox.join(",") : "n/a";

    htmlParts.push(`
      <div class="table-block">
        <div class="table-head">
          <div>Page ${blk.pageIndex + 1} • Table ${blk.ti + 1} (${nRows}×${nCols})</div>
          <div class="table-note">bbox: ${escapeHtml(bbox)}</div>
        </div>
        <div style="overflow:auto;">
          <table class="ocr-table">
            <tbody>
              ${cells.map(row => `
                <tr>
                  ${(row || []).map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `);
  }

  wrap.innerHTML = htmlParts.join("");
}

function renderDoc(doc) {
  if (!doc) return;
  __currentDoc = doc;
  if ($("fields")) $("fields").textContent = prettyJson(doc.fields || {});
  if ($("text")) $("text").textContent = doc.text || "";
  if ($("analysis")) $("analysis").textContent = prettyJson(doc.analysis || {});
  renderJsonPanel(doc);
  renderTables(doc);
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
      alert("OCR failed. Check server logs.");
      return;
    }

    renderDoc(data);
    setStatus("Done ✅");

  } catch (err) {
    console.error(err);
    setStatus("Error ❌");
  }
}

/* ================= BATCH OCR ================= */

function renderBatchResults(data) {
  __currentBatch = data;

  const container = $("batch_results");
  if (!container) return;

  container.innerHTML = "";

  if (!data.results || data.results.length === 0) {
    container.innerHTML = "<div class='muted'>No batch results found.</div>";
    return;
  }

  data.results.forEach((doc, index) => {
    const card = document.createElement("div");
    card.className = "batch-card";

    const title = document.createElement("h3");
    title.textContent = doc.filename || `Document ${index + 1}`;

    const meta = document.createElement("div");
    meta.className = "batch-meta";

    const status = (doc.status || "completed").toLowerCase();
    const badgeClass = status === "completed" ? "ok" : (status === "processing" ? "warn" : "bad");

    const pagesCount = (doc.pages && doc.pages.length) ? doc.pages.length : "n/a";
    const tablesCount = (doc.pages || []).reduce((a, p) => a + ((p.tables || []).length), 0);

    meta.innerHTML = `
      <span class="badge ${badgeClass}">${escapeHtml(status.toUpperCase())}</span>
      <span>Pages: ${escapeHtml(pagesCount)}</span>
      <span>Tables: ${escapeHtml(tablesCount)}</span>
    `;

    card.appendChild(title);
    card.appendChild(meta);

    card.addEventListener("click", () => {
      Array.from(container.querySelectorAll(".batch-card")).forEach(x => x.classList.remove("active"));
      card.classList.add("active");
      renderDoc(doc);
    });

    container.appendChild(card);
    if (index === 0) card.classList.add("active");
  });

  renderDoc(data.results[0]);
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
      alert("Batch OCR failed. Check server logs.");
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
  if ($("json_download")) $("json_download").addEventListener("click", downloadCurrentJson);
});
