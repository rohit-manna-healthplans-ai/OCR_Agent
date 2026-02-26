(function () {
  "use strict";

  var results = [];
  var currentIndex = 0;

  function el(id) {
    return document.getElementById(id);
  }

  function setStatus(msg, type) {
    var s = document.getElementById("status");
    if (!s) return;
    s.textContent = msg || "Idle";
    s.style.background = "rgba(255,255,255,0.06)";
    if (type === "ok") s.style.background = "rgba(16,185,129,0.14)";
    if (type === "warn") s.style.background = "rgba(245,158,11,0.14)";
    if (type === "bad") s.style.background = "rgba(239,68,68,0.14)";
  }

  function showTab(name) {
    var textPane = document.getElementById("paneText");
    var rawPane = document.getElementById("paneRaw");
    var btns = document.querySelectorAll(".tabbtn");
    if (textPane) textPane.classList.toggle("hidden", name !== "text");
    if (rawPane) rawPane.classList.toggle("hidden", name !== "raw");
    if (btns[0]) btns[0].classList.toggle("active", name === "text");
    if (btns[1]) btns[1].classList.toggle("active", name === "raw");
  }

  function renderResult() {
    var r = results[currentIndex] || null;
    var textEl = document.getElementById("extractedText");
    var rawEl = document.getElementById("rawJson");
    if (textEl) textEl.textContent = r ? (r.formatted_text || "") : "";
    if (rawEl) rawEl.textContent = r ? JSON.stringify(r.raw_json || {}, null, 2) : "";
  }

  function updateResultSelect() {
    var row = document.getElementById("resultRow");
    var sel = document.getElementById("resultSelect");
    if (!row || !sel) return;
    if (results.length <= 1) {
      row.classList.add("hidden");
      return;
    }
    row.classList.remove("hidden");
    sel.innerHTML = "";
    results.forEach(function (r, i) {
      var opt = document.createElement("option");
      opt.value = i;
      opt.textContent = r.filename || "File " + (i + 1);
      sel.appendChild(opt);
    });
    sel.value = "0";
    currentIndex = 0;
  }

  var timerInterval = null;

  function formatElapsed(ms) {
    return (ms / 1000).toFixed(1) + "s";
  }

  function setTimerBox(value, state) {
    var box = document.getElementById("timerBox");
    var val = document.getElementById("timerValue");
    if (val) val.textContent = value;
    if (box) {
      box.classList.remove("running", "error");
      if (state === "running") box.classList.add("running");
      if (state === "error") box.classList.add("error");
    }
  }

  // Expose on window first so onclick="runOcr()" works before DOMContentLoaded
  window.runOcr = function () {
    var input = document.getElementById("fileInput");
    if (!input || !input.files || input.files.length === 0) {
      setStatus("Select files first", "warn");
      return;
    }
    if (input.files.length > 10) {
      setStatus("Max 10 files", "warn");
      return;
    }
    var isBatch = input.files.length > 1;
    var endpoint = isBatch ? "/ocr-batch" : "/ocr";
    var fd = new FormData();
    var i;
    if (isBatch) {
      for (i = 0; i < input.files.length; i++) {
        fd.append("files", input.files[i]);
      }
    } else {
      fd.append("file", input.files[0]);
    }

    var startTime = Date.now();
    setTimerBox("0.0s", "running");
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(function () {
      var s = document.getElementById("status");
      if (s) s.textContent = "Running OCR... " + formatElapsed(Date.now() - startTime);
      setTimerBox(formatElapsed(Date.now() - startTime), "running");
    }, 500);
    setStatus("Running OCR...", "warn");
    showTab("text");

    fetch(endpoint, { method: "POST", body: fd })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, status: res.status, data: data };
        }).catch(function () {
          return { ok: false, status: res.status, data: {} };
        });
      })
      .then(function (out) {
        if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
        var elapsed = Date.now() - startTime;
        var elapsedStr = formatElapsed(elapsed);
        if (!out.ok) {
          setTimerBox(elapsedStr, "error");
          setStatus("Error: " + (out.data.detail || out.status), "bad");
          return;
        }
        setTimerBox(elapsedStr, "done");
        results = isBatch && Array.isArray(out.data.results) ? out.data.results : [out.data];
        currentIndex = 0;
        updateResultSelect();
        renderResult();
        setStatus("Done. " + results.length + " result(s).", "ok");
      })
      .catch(function (err) {
        if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
        var elapsed = Date.now() - startTime;
        setTimerBox(formatElapsed(elapsed), "error");
        setStatus("Request failed: " + (err && err.message ? err.message : String(err)), "bad");
      });
  };

  window.showTab = showTab;
  window.selectResult = function () {
    var sel = document.getElementById("resultSelect");
    if (sel) currentIndex = parseInt(sel.value, 10) || 0;
    renderResult();
  };

  document.addEventListener("DOMContentLoaded", function () {
    try {
      setTimerBox("—", "idle");
      renderResult();
      setStatus("Idle", "ok");
    } catch (e) {
      console.error(e);
    }
  });
})();
