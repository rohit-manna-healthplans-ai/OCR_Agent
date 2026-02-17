const $ = (id) => document.getElementById(id);

function setStatus(msg) {
  $("status").textContent = msg || "";
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderFields(fields) {
  const root = $("fields");
  root.innerHTML = "";

  const keys = fields ? Object.keys(fields) : [];
  if (!keys.length) {
    root.innerHTML = `<div class="kv empty">No fields detected.</div>`;
    return;
  }

  const order = ["Policy No", "SL No/Certificate No", "Company/TPA ID No", "Name", "Address", "City", "Pincode"];
  const sorted = [...keys].sort((a, b) => {
    const ia = order.indexOf(a), ib = order.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });

  for (const k of sorted) {
    const v = fields[k];
    const row = document.createElement("div");
    row.className = "kv";
    row.innerHTML = `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div>`;
    root.appendChild(row);
  }
}

function renderTables(pages) {
  const root = $("tables");
  root.innerHTML = "";

  let found = 0;
  for (const p of (pages || [])) {
    for (const tb of (p.tables || [])) {
      found++;
      const div = document.createElement("div");
      div.className = "tablecard";
      const cells = tb.cells || [];
      let html = `<div class="tmeta">page ${p.page_index} | rows=${tb.n_rows} cols=${tb.n_cols}</div>`;
      html += `<div class="tgrid">`;
      html += `<table><tbody>`;
      for (const row of cells.slice(0, 25)) {
        html += "<tr>";
        for (const cell of row.slice(0, 12)) {
          html += `<td>${escapeHtml(cell || "")}</td>`;
        }
        html += "</tr>";
      }
      html += `</tbody></table></div>`;
      div.innerHTML = html;
      root.appendChild(div);
    }
  }

  if (!found) {
    root.innerHTML = `<div class="kv empty">No tables detected (best-effort).</div>`;
  }
}

async function runSingleOCR() {
  const files = $("files").files;
  if (!files || !files.length) { setStatus("Please select file(s)."); return; }
  const f = files[0];

  const engine = $("engine").value;
  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;
  const debug = $("debug").value;

  const form = new FormData();
  form.append("file", f);

  const url = `/ocr?engine=${encodeURIComponent(engine)}&preset=${encodeURIComponent(preset)}&dpi=${encodeURIComponent(dpi)}&max_pages=${encodeURIComponent(maxPages)}&return_debug=${encodeURIComponent(debug)}`;
  setStatus("Running OCR...");
  const res = await fetch(url, { method: "POST", body: form });
  const data = await res.json();

  $("json").textContent = pretty(data);

  if (!res.ok) {
    $("text").textContent = "";
    renderFields({});
    renderTables([]);
    setStatus(`Error: ${data.error || res.statusText}`);
    return;
  }

  $("text").textContent = data.text || "";
  renderFields(data.fields || {});
  renderTables(data.pages || []);
  setStatus(`Done: ${data.file_name || f.name} | engine: ${data.meta?.engine_used || data.meta?.engine || engine}`);
}

async function runBatchOCR() {
  const files = $("files").files;
  if (!files || !files.length) { setStatus("Please select file(s)."); return; }

  const engine = $("engine").value;
  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;
  const debug = $("debug").value;

  const form = new FormData();
  for (const f of files) form.append("files", f);

  const url = `/ocr-batch?engine=${encodeURIComponent(engine)}&preset=${encodeURIComponent(preset)}&dpi=${encodeURIComponent(dpi)}&max_pages=${encodeURIComponent(maxPages)}&return_debug=${encodeURIComponent(debug)}`;
  setStatus("Running batch OCR...");
  const res = await fetch(url, { method: "POST", body: form });
  const data = await res.json();
  $("json").textContent = pretty(data);

  if (!res.ok) {
    setStatus(`Batch error: ${data.error || res.statusText}`);
    return;
  }

  setStatus(`Batch done. ok=${data.ok}/${data.count}. (See JSON)`);
}

async function downloadDocx() {
  const files = $("files").files;
  if (!files || !files.length) { setStatus("Please select file(s)."); return; }
  const f = files[0];

  const engine = $("engine").value;
  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;

  const form = new FormData();
  form.append("file", f);

  const url = `/ocr-docx?engine=${encodeURIComponent(engine)}&preset=${encodeURIComponent(preset)}&dpi=${encodeURIComponent(dpi)}&max_pages=${encodeURIComponent(maxPages)}`;
  setStatus("Building DOCX...");

  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    setStatus(`DOCX error: ${data.error || res.statusText}`);
    return;
  }

  const blob = await res.blob();
  const a = document.createElement("a");
  const fname = (f.name || "ocr").replace(/\.[^/.]+$/, "") + ".docx";
  a.href = URL.createObjectURL(blob);
  a.download = fname;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setStatus("DOCX downloaded.");
}

$("runSingle").addEventListener("click", () => runSingleOCR().catch(e => setStatus("Failed: " + e.message)));
$("runBatch").addEventListener("click", () => runBatchOCR().catch(e => setStatus("Failed: " + e.message)));
$("downloadDocx").addEventListener("click", () => downloadDocx().catch(e => setStatus("Failed: " + e.message)));
