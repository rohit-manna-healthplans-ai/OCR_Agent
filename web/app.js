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

let lastRaw = null;
let lastStructured = null;

async function run(){
  const file = $("file").files[0];
  if (!file){
    setStatus("Select a file first", "warn");
    return;
  }

  const engine = $("engine").value || "auto";
  const enableOllama = $("enableOllama").checked ? "true" : "false";

  const fd = new FormData();
  fd.append("file", file);

  setStatus("Running OCR...", "warn");

  try{
    const url = new URL("http://127.0.0.1:8000/ocr");
    url.searchParams.set("engine", engine);
    url.searchParams.set("enable_ollama", enableOllama);
    url.searchParams.set("return_debug", "true");
    url.searchParams.set("return_layout", "true");

    const resp = await fetch(url.toString(), { method: "POST", body: fd });
    if (!resp.ok){
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${txt}`);
    }
    const data = await resp.json();

    lastRaw = data;
    lastStructured = data?.structured ?? null;

    $("extractedText").textContent = (data?.text ?? "").toString();
    $("structuredJson").textContent = pretty(lastStructured ?? {});
    $("rawJson").textContent = pretty(lastRaw ?? {});

    setStatus("Done", "ok");
  }catch(e){
    setStatus("Error: " + (e?.message || e), "bad");
  }
}

function clearAll(){
  lastRaw = null;
  lastStructured = null;
  $("extractedText").textContent = "";
  $("structuredJson").textContent = "";
  $("rawJson").textContent = "";
  setStatus("Idle");
}

$("run").addEventListener("click", run);
$("clear").addEventListener("click", clearAll);

$("downloadRaw").addEventListener("click", ()=>{
  if (!lastRaw) return setStatus("No result to download", "warn");
  downloadJson("ocr_raw.json", lastRaw);
});

$("downloadStructured").addEventListener("click", ()=>{
  if (!lastStructured) return setStatus("No structured JSON to download", "warn");
  downloadJson("ocr_structured.json", lastStructured);
});
