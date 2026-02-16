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
    root.innerHTML = `<div class="kv empty">No fields detected (try preset clean_doc/photo/low_light).</div>`;
    return;
  }

  // stable order
  const order = ["Policy No", "SL No/Certificate No", "Company/TPA ID No", "Name", "Address", "City"];
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

function renderLines(data) {
  const root = $("lines");
  root.innerHTML = "";

  if (!data?.pages?.length) return;

  let shown = 0;
  for (const p of data.pages) {
    for (const ln of (p.lines || [])) {
      if (shown >= 200) break;
      const div = document.createElement("div");
      div.className = "line";
      div.innerHTML = `
        <div>${escapeHtml(ln.text)}</div>
        <div class="meta">conf: ${Number(ln.conf).toFixed(3)} | bbox: [${ln.bbox.join(", ")}] | page: ${p.page_index}</div>
      `;
      root.appendChild(div);
      shown++;
    }
    if (shown >= 200) break;
  }

  if (shown === 0) {
    root.innerHTML = `<div class="line"><div>No line boxes (digital PDF text-first or no detections).</div></div>`;
  }
}

$("run").addEventListener("click", async () => {
  const f = $("file").files?.[0];
  if (!f) {
    setStatus("Please select a file.");
    return;
  }

  const preset = $("preset").value;
  const dpi = $("dpi").value;
  const maxPages = $("max_pages").value;
  const debug = $("debug").value;

  const form = new FormData();
  form.append("file", f);

  const url = `/ocr?preset=${encodeURIComponent(preset)}&dpi=${encodeURIComponent(dpi)}&max_pages=${encodeURIComponent(maxPages)}&return_debug=${encodeURIComponent(debug)}`;

  setStatus("Running OCR...");

  try {
    const res = await fetch(url, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      $("text").textContent = "";
      $("json").textContent = pretty(data);
      renderFields({});
      renderLines({ pages: [] });
      setStatus(`Error: ${data.error || res.statusText}`);
      return;
    }

    $("text").textContent = data.text || "";
    $("json").textContent = pretty(data);
    renderFields(data.fields || {});
    renderLines(data);
    setStatus(`Done. Engine=${data?.meta?.engine} | digital_pdf=${data?.meta?.digital_pdf ?? false}`);
  } catch (e) {
    setStatus("Failed: " + e.message);
  }
});
