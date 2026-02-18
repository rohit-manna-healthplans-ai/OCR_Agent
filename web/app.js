/* OCR Viewer UI v3 (Page View + Human JSON) */
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

function escapeHtml(s){
  return String(s ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

let currentDoc = null;
let currentBatch = null;
let currentHuman = null;

function setHeader(doc){
  $("doc_title").textContent = (doc && doc.filename) ? doc.filename : "OCR Result";
  const pages = (doc && doc.pages) ? doc.pages.length : 0;
  const engine = doc?.meta?.engine_used || doc?.meta?.engine || "n/a";
  const dpi = doc?.meta?.dpi ?? "n/a";
  const tables = (doc?.pages || []).reduce((a,p)=>a+((p.tables||[]).length),0);
  const pairs = (doc?.fields?.pairs || []).length;
  $("doc_meta").textContent = `Pages: ${pages} • Tables: ${tables} • Pairs: ${pairs} • Engine: ${engine} • DPI: ${dpi}`;
}

function clearUI(){
  currentDoc = null;
  currentBatch = null;
  currentHuman = null;
  $("batch_results").innerHTML = "";
  $("fields").textContent = "";
  $("text").textContent = "";
  $("json").textContent = "";
  $("analysis").textContent = "";
  $("human").textContent = "";
  $("tables").innerHTML = "";
  $("page_render").innerHTML = "";
  $("page_md").textContent = "";
  $("page_select").innerHTML = "";
  $("doc_title").textContent = "No document loaded";
  $("doc_meta").textContent = "";
  setStatus("Idle");
  setActiveTab("page");
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

function downloadText(filename, text, mime){
  const blob = new Blob([text], {type: mime || "text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function toCSV(rows){
  const esc = (v)=>`"${String(v??"").replaceAll('"','""')}"`;
  return (rows||[]).map(r => (r||[]).map(esc).join(",")).join("\n");
}

function renderTables(doc){
  const wrap = $("tables");
  const blocks = [];
  (doc?.pages || []).forEach((p, pi)=>{
    (p.tables || []).forEach((t, ti)=> blocks.push({pi, ti, t}));
  });

  if (!blocks.length){
    wrap.innerHTML = `<div style="color:rgba(232,238,252,0.7)">No tables detected.</div>`;
    return;
  }

  wrap.innerHTML = blocks.map(b=>{
    const t = b.t || {};
    const rows = t.rows || t.cells || [];
    const r = t.row_count || t.n_rows || rows.length || 0;
    const c = t.col_count || t.n_cols || (rows[0]?.length || 0);
    const bbox = Array.isArray(t.bbox) ? t.bbox.join(",") : (t.bbox || "n/a");
    const id = `p${b.pi+1}_t${b.ti+1}`;
    return `
      <div class="table">
        <div class="table__head">
          <div>Page ${b.pi+1} • Table ${b.ti+1} (${r}×${c})</div>
          <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap">
            <button class="mini ghost" type="button" data-csv="${escapeHtml(id)}">CSV</button>
            <div class="table__note">bbox: ${escapeHtml(bbox)}</div>
          </div>
        </div>
        <div style="overflow:auto">
          <table class="grid" data-table="${escapeHtml(id)}"><tbody>
            ${rows.map(row=>`<tr>${row.map(cell=>`<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}
          </tbody></table>
        </div>
      </div>
    `;
  }).join("");

  wrap.querySelectorAll("[data-csv]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const id = btn.getAttribute("data-csv");
      const table = wrap.querySelector(`[data-table="${CSS.escape(id)}"]`);
      if (!table) return;
      const rows = [];
      table.querySelectorAll("tr").forEach(tr=>{
        const row = [];
        tr.querySelectorAll("td").forEach(td=>row.push(td.textContent || ""));
        rows.push(row);
      });
      downloadText(`${id}.csv`, toCSV(rows), "text/csv;charset=utf-8");
    });
  });
}

function mdToHtml(md){
  const lines = String(md || "").split("\n");
  const parts = [];
  for (const raw of lines){
    const line = raw.trimEnd();
    if (!line) continue;

    if (line.startsWith("## ")){
      parts.push(`<h2>${escapeHtml(line.slice(3))}</h2>`);
      continue;
    }
    if (line.startsWith("# ")){
      parts.push(`<h1>${escapeHtml(line.slice(2))}</h1>`);
      continue;
    }
    // bold: **text**
    const html = escapeHtml(line).replaceAll(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    parts.push(`<p>${html}</p>`);
  }
  return parts.join("");
}

function buildHumanJson(doc){
  const pages = doc?.pages || [];
  const fields = doc?.fields?.pairs || [];
  const tables = pages.map((p,i)=>({
    page: i+1,
    tables: (p.tables || []).map(t=>({
      row_count: t.row_count || t.n_rows || 0,
      col_count: t.col_count || t.n_cols || 0,
      bbox: t.bbox || null,
      rows: t.rows || t.cells || []
    }))
  }));

  return {
    meta: doc.meta || {},
    summary: {
      total_pages: pages.length,
      total_tables: pages.reduce((a,p)=>a+((p.tables||[]).length),0),
      total_fields: fields.length
    },
    fields: fields,
    tables_by_page: tables
  };
}

function renderPageView(doc, pageIndex){
  const pages = doc?.pages || [];
  const p = pages[pageIndex] || null;
  const md = p?.text_markdown || p?.text || "";
  $("page_md").textContent = md;
  $("page_render").innerHTML = mdToHtml(md);
}

function populatePageSelect(doc){
  const sel = $("page_select");
  const pages = doc?.pages || [];
  sel.innerHTML = "";
  for (let i=0; i<pages.length; i++){
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `Page ${i+1}`;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", ()=>{
    const idx = parseInt(sel.value || "0", 10);
    renderPageView(doc, isNaN(idx) ? 0 : idx);
  });
}

function renderDoc(doc){
  currentDoc = doc;
  setHeader(doc);

  currentHuman = buildHumanJson(doc);
  $("human").textContent = pretty(currentHuman);

  $("fields").textContent = pretty(doc.fields || {});
  $("text").textContent = doc.text || "";
  $("json").textContent = pretty(doc);
  $("analysis").textContent = pretty(doc.analysis || {});
  renderTables(doc);

  populatePageSelect(doc);
  renderPageView(doc, 0);

  applySearch($("search").value || "");
}

function setActiveTab(name){
  document.querySelectorAll(".tab").forEach(t=>{
    t.classList.toggle("is-active", t.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach(p=>{
    p.hidden = (p.dataset.panel !== name);
  });
}

function applySearch(q){
  const query = (q||"").trim().toLowerCase();
  if (!currentDoc) return;

  const activePanel = document.querySelector(".panel:not([hidden])")?.dataset.panel;

  const filterPre = (el, raw)=>{
    if (!el) return;
    if (!query){ el.textContent = raw; return; }
    const lines = String(raw||"").split("\n");
    const matches = lines.filter(l => l.toLowerCase().includes(query));
    el.textContent = matches.length ? matches.join("\n") : "(no matches)";
  };

  if (activePanel === "fields") filterPre($("fields"), pretty(currentDoc.fields || {}));
  if (activePanel === "text") filterPre($("text"), currentDoc.text || "");
  if (activePanel === "json") filterPre($("json"), pretty(currentDoc));
  if (activePanel === "analysis") filterPre($("analysis"), pretty(currentDoc.analysis || {}));
  if (activePanel === "human") filterPre($("human"), pretty(currentHuman || {}));
  if (activePanel === "page") {
    // lightweight: filter markdown view
    const pages = currentDoc.pages || [];
    const idx = parseInt($("page_select").value || "0", 10) || 0;
    const md = pages[idx]?.text_markdown || pages[idx]?.text || "";
    if (!query){
      $("page_md").textContent = md;
      $("page_render").innerHTML = mdToHtml(md);
    } else {
      const lines = String(md||"").split("\n");
      const matches = lines.filter(l => l.toLowerCase().includes(query));
      const out = matches.length ? matches.join("\n") : "(no matches)";
      $("page_md").textContent = out;
      $("page_render").innerHTML = mdToHtml(out);
    }
  }
}

async function copyFrom(id){
  const el = $(id);
  if (!el) return;
  try{
    await navigator.clipboard.writeText(el.textContent || "");
    setStatus("Copied ✅", "ok");
    setTimeout(()=>setStatus("Idle"), 800);
  }catch{
    alert("Copy failed (browser permissions).");
  }
}

async function runSingle(){
  const files = $("files").files;
  if (!files || !files.length) return alert("Select a file first.");
  const engine = $("engine").value || "auto";
  const fd = new FormData();
  fd.append("file", files[0]);

  setStatus("Running OCR…", "warn");
  try{
    const res = await fetch(`/ocr?engine=${encodeURIComponent(engine)}`, {method:"POST", body: fd});
    const data = await res.json();
    if (!res.ok){ console.error(data); setStatus("Error", "bad"); return alert("OCR failed. Check logs."); }
    renderDoc(data);
    setStatus("Done ✅", "ok");
  }catch(e){
    console.error(e); setStatus("Error", "bad");
  }
}

function renderBatchList(batch){
  const wrap = $("batch_results");
  wrap.innerHTML = "";
  const results = batch?.results || [];
  if (!results.length){
    wrap.innerHTML = `<div style="color:rgba(232,238,252,0.7)">No batch results.</div>`;
    return;
  }

  results.forEach((doc, i)=>{
    const item = document.createElement("div");
    item.className = "batch__item" + (i===0 ? " is-active" : "");

    const status = (doc.status || "completed").toLowerCase();
    const badge = document.createElement("span");
    badge.className = "badge " + (status==="completed" ? "ok" : status==="processing" ? "warn" : "bad");
    badge.textContent = status.toUpperCase();

    const title = document.createElement("div");
    title.className = "batch__title";
    title.textContent = doc.filename || `Document ${i+1}`;

    const meta = document.createElement("div");
    meta.className = "batch__meta";

    const pages = (doc.pages || []).length;
    const tables = (doc.pages || []).reduce((a,p)=>a+((p.tables||[]).length),0);
    const pairs = (doc.fields?.pairs || []).length;

    meta.appendChild(badge);
    meta.appendChild(document.createTextNode(` Pages: ${pages} `));
    meta.appendChild(document.createTextNode(` Tables: ${tables} `));
    meta.appendChild(document.createTextNode(` Pairs: ${pairs} `));

    item.appendChild(title);
    item.appendChild(meta);

    item.addEventListener("click", ()=>{
      wrap.querySelectorAll(".batch__item").forEach(x=>x.classList.remove("is-active"));
      item.classList.add("is-active");
      renderDoc(doc);
    });

    wrap.appendChild(item);
  });

  renderDoc(results[0]);
}

async function runBatch(){
  const files = $("files").files;
  if (!files || !files.length) return alert("Select files for batch.");
  const engine = $("engine").value || "auto";

  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  setStatus(`Batch OCR (${files.length})…`, "warn");
  try{
    const res = await fetch(`/ocr-batch?engine=${encodeURIComponent(engine)}`, {method:"POST", body: fd});
    const data = await res.json();
    if (!res.ok){ console.error(data); setStatus("Error", "bad"); return alert("Batch failed. Check logs."); }
    currentBatch = data;
    renderBatchList(data);
    setStatus("Batch Done ✅", "ok");
  }catch(e){
    console.error(e); setStatus("Error", "bad");
  }
}

async function downloadDocx(){
  const files = $("files").files;
  if (!files || !files.length) return alert("Select a file first.");
  const engine = $("engine").value || "auto";

  const fd = new FormData();
  fd.append("file", files[0]);

  setStatus("Generating DOCX…", "warn");
  try{
    const res = await fetch(`/ocr-docx?engine=${encodeURIComponent(engine)}`, {method:"POST", body: fd});
    if (!res.ok){ setStatus("Error", "bad"); return alert("DOCX failed."); }
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "ocr_result.docx";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setStatus("DOCX downloaded ✅", "ok");
  }catch(e){
    console.error(e); setStatus("Error", "bad");
  }
}

window.addEventListener("DOMContentLoaded", ()=>{
  $("run").addEventListener("click", runSingle);
  $("batch").addEventListener("click", runBatch);
  $("docx").addEventListener("click", downloadDocx);
  $("clear").addEventListener("click", clearUI);

  $("json_download").addEventListener("click", ()=>{
    if (currentDoc) return downloadJson((currentDoc.filename||"ocr_result").replace(/[^a-z0-9._-]/gi,"_") + ".json", currentDoc);
    if (currentBatch) return downloadJson("batch_results.json", currentBatch);
    alert("No JSON yet. Run OCR first.");
  });

  document.querySelectorAll(".tab").forEach(t=>{
    t.addEventListener("click", ()=>{
      setActiveTab(t.dataset.tab);
      applySearch($("search").value || "");
    });
  });

  $("search").addEventListener("input", (e)=> applySearch(e.target.value));

  document.querySelectorAll("[data-copy]").forEach(btn=>{
    btn.addEventListener("click", ()=> copyFrom(btn.getAttribute("data-copy")));
  });

  setActiveTab("page");
  setStatus("Idle");
});
