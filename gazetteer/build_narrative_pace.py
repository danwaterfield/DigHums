#!/usr/bin/env python3
"""Build the self-contained narrative pace HTML.

Reads narrative_pace_data.json and writes a self-contained narrative_pace.html
with four tab views: Century, Arcs, Grid, Ecology.

Usage:
    python3 gazetteer/build_narrative_pace.py
    open gazetteer/narrative_pace.html
"""
import json
from pathlib import Path

JSON_PATH = Path(__file__).parent / "narrative_pace_data.json"
OUT_PATH  = Path(__file__).parent / "narrative_pace.html"


def build(json_path=JSON_PATH, out_path=OUT_PATH):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data_js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__NOVELS_DATA__", data_js, 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"Narrative pace -> {out_path}")
    print(f"  {len(data['novels'])} novels")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Narrative Pace — 18th-Century Fiction</title>
<style>
  :root {{
    --bg: #1a1a2e;
    --surface: #16213e;
    --card: #0f3460;
    --accent: #e94560;
    --text: #eaeaea;
    --muted: #888;
    --border: #2a3a5e;
    --dialogue:    #3498db;
    --singulative: #2ecc71;
    --iterative:   #e67e22;
    --description: #9b59b6;
    --fid:         #e74c3c;
    --commentary:  #95a5a6;
    --domestic:    #2ecc71;
    --gothic:      #9b59b6;
    --picaresque:  #e74c3c;
    --epistolary:  #3498db;
    --amatory:     #e67e22;
    --satirical:   #95a5a6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; font-size: 14px; }}
  header {{ padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; }}
  header h1 {{ font-size: 18px; font-weight: 600; }}
  header p {{ color: var(--muted); font-size: 13px; }}

  /* Tabs */
  #tab-bar {{ display: flex; gap: 4px; padding: 12px 24px 0; border-bottom: 1px solid var(--border); background: var(--surface); }}
  .tab-btn {{ padding: 8px 18px; border: none; border-radius: 6px 6px 0 0; background: transparent; color: var(--muted); cursor: pointer; font-size: 13px; transition: background 0.15s, color 0.15s; }}
  .tab-btn:hover {{ background: var(--card); color: var(--text); }}
  .tab-btn.active {{ background: var(--bg); color: var(--text); border: 1px solid var(--border); border-bottom: 1px solid var(--bg); }}

  /* Panels */
  .tab-panel {{ display: none; padding: 24px; }}
  .tab-panel.active {{ display: block; }}

  /* Controls bar */
  .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 18px; }}
  .controls label {{ color: var(--muted); font-size: 12px; }}
  .controls select {{ background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; font-size: 13px; }}
  .controls input[type=range] {{ accent-color: var(--accent); width: 120px; }}
  .controls input[type=checkbox] {{ accent-color: var(--accent); }}

  /* Pills */
  .pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }}
  .pill {{ padding: 4px 12px; border-radius: 20px; border: 1px solid var(--border); background: var(--card); color: var(--text); cursor: pointer; font-size: 12px; transition: opacity 0.15s; }}
  .pill.active {{ opacity: 1; border-color: var(--accent); }}
  .pill.inactive {{ opacity: 0.4; }}

  /* Canvas wrapper */
  .chart-wrap {{ position: relative; width: 100%; }}
  canvas {{ display: block; }}

  /* Novel grid */
  #grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; margin-top: 16px; }}
  .novel-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; cursor: pointer; transition: border-color 0.15s; }}
  .novel-card:hover {{ border-color: var(--accent); }}
  .card-title {{ font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
  .card-meta {{ font-size: 11px; color: var(--muted); margin-bottom: 10px; }}
  .card-bar {{ display: flex; height: 6px; border-radius: 3px; overflow: hidden; width: 100%; }}

  /* Legend */
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }}

  /* Ecology layout */
  #ecology-panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 768px) {{ #ecology-panels {{ grid-template-columns: 1fr; }} }}
  .panel-title {{ font-size: 14px; font-weight: 600; margin-bottom: 12px; }}
  .callout {{ background: var(--card); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 14px 18px; margin-top: 24px; font-size: 13px; line-height: 1.6; color: var(--muted); }}
  .callout strong {{ color: var(--text); }}

  /* Tooltip */
  #tooltip {{ position: fixed; background: rgba(15,52,96,0.95); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; font-size: 12px; pointer-events: none; z-index: 1000; display: none; max-width: 220px; line-height: 1.5; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Narrative Pace</h1>
    <p>Dialogue, singulative, iterative, description, FID and commentary across the long eighteenth century</p>
  </div>
</header>

<div id="tab-bar">
  <button class="tab-btn active" data-tab="century">Century</button>
  <button class="tab-btn" data-tab="arcs">Arcs</button>
  <button class="tab-btn" data-tab="grid">Grid</button>
  <button class="tab-btn" data-tab="ecology">Ecology</button>
</div>

<!-- ═══════════════════ TAB: Century ═══════════════════ -->
<div id="tab-century" class="tab-panel active">
  <div class="legend" id="century-legend"></div>
  <div class="chart-wrap" style="height:340px">
    <canvas id="century-scatter"></canvas>
  </div>
  <h3 style="margin:20px 0 10px;font-size:13px;color:var(--muted)">Category breakdown per novel</h3>
  <div class="chart-wrap" style="height:320px;overflow-y:auto">
    <canvas id="century-bars"></canvas>
  </div>
</div>

<!-- ═══════════════════ TAB: Arcs ═══════════════════ -->
<div id="tab-arcs" class="tab-panel">
  <div class="controls">
    <label>Novel
      <select id="arcs-select"></select>
    </label>
    <label>Smoothing
      <input type="range" id="arcs-smooth" min="1" max="10" value="3">
      <span id="arcs-smooth-val">3</span>
    </label>
    <label><input type="checkbox" id="arcs-boundaries" checked> Volume boundaries</label>
    <label><input type="checkbox" id="arcs-dominant"> Dominant only</label>
  </div>
  <div class="legend" id="arcs-legend"></div>
  <div class="chart-wrap" style="height:380px">
    <canvas id="arcs-chart"></canvas>
  </div>
</div>

<!-- ═══════════════════ TAB: Grid ═══════════════════ -->
<div id="tab-grid" class="tab-panel">
  <div class="pills" id="grid-pills"></div>
  <div id="grid-container"></div>
</div>

<!-- ═══════════════════ TAB: Ecology ═══════════════════ -->
<div id="tab-ecology" class="tab-panel">
  <div id="ecology-panels">
    <div>
      <div class="panel-title">Filler proportion vs FID (by novel)</div>
      <div class="chart-wrap" style="height:320px">
        <canvas id="eco-scatter"></canvas>
      </div>
    </div>
    <div>
      <div class="panel-title">Environmental evidence by decade</div>
      <div class="chart-wrap" style="height:320px">
        <canvas id="eco-env"></canvas>
      </div>
    </div>
  </div>
  <div class="callout">
    <strong>The Ghosh thesis:</strong> Amitav Ghosh argues that literary realism suppresses the representation of extreme or volatile environments in favour of the quotidian. These charts test an analogous hypothesis for eighteenth-century British fiction: as urban pollution and sensory intensity increase across the century (right panel), does prose shift toward <em>iterative</em> and <em>descriptive</em> filler — normalising the urban environment — and away from the immediacy of singulative narrative? Higher filler% with rising FID (free indirect discourse) would suggest authorial ambivalence: immersion in sensory detail without direct report.
  </div>
</div>

<div id="tooltip"></div>

<script>
(function () {{
  "use strict";

  // ── Data ──────────────────────────────────────────────────────────────────
  const RAW = __NOVELS_DATA__;
  const NOVELS = RAW.novels;
  const ENV = RAW.environmental || [];

  // ── Colours ───────────────────────────────────────────────────────────────
  const CAT_COLOURS = {{
    dialogue:    "#3498db",
    singulative: "#2ecc71",
    iterative:   "#e67e22",
    description: "#9b59b6",
    fid:         "#e74c3c",
    commentary:  "#95a5a6",
  }};
  const CATS = Object.keys(CAT_COLOURS);

  const GENRE_COLOURS = {{
    domestic:   "#2ecc71",
    gothic:     "#9b59b6",
    picaresque: "#e74c3c",
    epistolary: "#3498db",
    amatory:    "#e67e22",
    satirical:  "#95a5a6",
  }};

  function genreColour(g) {{ return GENRE_COLOURS[g] || "#aaa"; }}

  // ── Tooltip ───────────────────────────────────────────────────────────────
  const tooltip = document.getElementById("tooltip");
  function showTip(x, y, lines) {{
    tooltip.style.display = "block";
    tooltip.style.left = (x + 14) + "px";
    tooltip.style.top  = (y + 14) + "px";
    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
    lines.forEach(function(line) {{
      const d = document.createElement("div");
      d.textContent = line;
      tooltip.appendChild(d);
    }});
  }}
  function hideTip() {{ tooltip.style.display = "none"; }}

  // ── Tabs ──────────────────────────────────────────────────────────────────
  document.querySelectorAll(".tab-btn").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      document.querySelectorAll(".tab-btn").forEach(function(b) {{ b.classList.remove("active"); }});
      document.querySelectorAll(".tab-panel").forEach(function(p) {{ p.classList.remove("active"); }});
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "arcs") drawArcs();
      if (btn.dataset.tab === "century") drawCentury();
      if (btn.dataset.tab === "ecology") drawEcology();
    }});
  }});

  // ── Utility ───────────────────────────────────────────────────────────────
  function dpr() {{ return window.devicePixelRatio || 1; }}

  function setupCanvas(canvas, w, h) {{
    const r = dpr();
    canvas.width  = w * r;
    canvas.height = h * r;
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(r, r);
    return ctx;
  }}

  function smooth(arr, k) {{
    if (k <= 1) return arr.slice();
    const out = [];
    for (let i = 0; i < arr.length; i++) {{
      let sum = 0, count = 0;
      for (let j = Math.max(0, i - k); j <= Math.min(arr.length - 1, i + k); j++) {{
        sum += arr[j]; count++;
      }}
      out.push(sum / count);
    }}
    return out;
  }}

  function linReg(pts) {{
    const n = pts.length;
    if (n < 2) return null;
    let sx = 0, sy = 0, sxy = 0, sxx = 0;
    pts.forEach(function(p) {{ sx += p.x; sy += p.y; sxy += p.x * p.y; sxx += p.x * p.x; }});
    const m = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const b = (sy - m * sx) / n;
    return {{ m: m, b: b }};
  }}

  // ── Build legends ─────────────────────────────────────────────────────────
  function buildCatLegend(container) {{
    while (container.firstChild) container.removeChild(container.firstChild);
    CATS.forEach(function(cat) {{
      const item = document.createElement("div");
      item.className = "legend-item";
      const sw = document.createElement("div");
      sw.className = "legend-swatch";
      sw.style.background = CAT_COLOURS[cat];
      const lbl = document.createElement("span");
      lbl.textContent = cat;
      item.appendChild(sw);
      item.appendChild(lbl);
      container.appendChild(item);
    }});
  }}

  buildCatLegend(document.getElementById("century-legend"));
  buildCatLegend(document.getElementById("arcs-legend"));

  // ── TAB: Century ──────────────────────────────────────────────────────────
  function drawCentury() {{
    const scatter = document.getElementById("century-scatter");
    const container = scatter.parentElement;
    const W = container.clientWidth || 800;
    const H = 320;
    const ctx = setupCanvas(scatter, W, H);

    const PAD = {{ t: 20, r: 20, b: 40, l: 50 }};
    const CW = W - PAD.l - PAD.r;
    const CH = H - PAD.t - PAD.b;

    // Compute filler% per novel
    const pts = NOVELS.map(function(n) {{
      const s = n.summary;
      const filler = (s.iterative || 0) + (s.description || 0) + (s.fid || 0);
      return {{ x: n.year, y: filler, novel: n }};
    }});

    const years = pts.map(function(p) {{ return p.x; }});
    const minX = Math.min.apply(null, years) - 5;
    const maxX = Math.max.apply(null, years) + 5;
    const maxY = 1.0;

    function px(x) {{ return PAD.l + (x - minX) / (maxX - minX) * CW; }}
    function py(y) {{ return PAD.t + (1 - y / maxY) * CH; }}

    // Axes
    ctx.strokeStyle = "#2a3a5e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + CH);
    ctx.lineTo(PAD.l + CW, PAD.t + CH);
    ctx.stroke();

    // Y gridlines + labels
    ctx.fillStyle = "#888";
    ctx.font = "11px system-ui";
    ctx.textAlign = "right";
    for (let v = 0; v <= 1.0; v += 0.2) {{
      const y = py(v);
      ctx.strokeStyle = "#1e2e4e";
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + CW, y); ctx.stroke();
      ctx.fillText(Math.round(v * 100) + "%", PAD.l - 6, y + 4);
    }}

    // X labels (decades)
    ctx.textAlign = "center";
    ctx.fillStyle = "#888";
    for (let yr = Math.ceil(minX / 10) * 10; yr <= maxX; yr += 10) {{
      ctx.fillText(yr, px(yr), PAD.t + CH + 18);
    }}

    // Regression line
    const reg = linReg(pts);
    if (reg) {{
      const x0 = minX, x1 = maxX;
      ctx.strokeStyle = "rgba(233,69,96,0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(px(x0), py(reg.m * x0 + reg.b));
      ctx.lineTo(px(x1), py(reg.m * x1 + reg.b));
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Dots
    pts.forEach(function(p) {{
      const cx2 = px(p.x), cy2 = py(p.y);
      ctx.beginPath();
      ctx.arc(cx2, cy2, 6, 0, Math.PI * 2);
      ctx.fillStyle = genreColour(p.novel.genre);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }});

    // Mouse interactions
    scatter._pts = pts.map(function(p) {{ return {{ cx: px(p.x), cy: py(p.y), novel: p.novel, filler: p.y }}; }});
    scatter._bounds = {{ PAD: PAD, W: W, H: H }};

    scatter.onmousemove = function(e) {{
      const rect = scatter.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let found = null;
      scatter._pts.forEach(function(p) {{
        if (Math.hypot(mx - p.cx, my - p.cy) < 10) found = p;
      }});
      if (found) {{
        showTip(e.clientX, e.clientY, [
          found.novel.title,
          found.novel.author + " · " + found.novel.year,
          "Filler: " + Math.round(found.filler * 100) + "%"
        ]);
      }} else hideTip();
    }};

    scatter.onclick = function(e) {{
      const rect = scatter.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      scatter._pts.forEach(function(p) {{
        if (Math.hypot(mx - p.cx, my - p.cy) < 10) {{
          // Navigate to Arcs tab
          document.querySelector("[data-tab='arcs']").click();
          const sel = document.getElementById("arcs-select");
          for (let i = 0; i < sel.options.length; i++) {{
            if (sel.options[i].value === p.novel.id) {{
              sel.selectedIndex = i;
              break;
            }}
          }}
          drawArcs();
        }}
      }});
    }};

    scatter.onmouseleave = hideTip;

    // Stacked bars below
    drawCenturyBars();
  }}

  function drawCenturyBars() {{
    const canvas = document.getElementById("century-bars");
    const container = canvas.parentElement;
    const barH = 22;
    const PAD = {{ t: 10, r: 20, b: 10, l: 170 }};
    const W = container.clientWidth || 800;
    const H = NOVELS.length * barH + PAD.t + PAD.b;
    canvas.parentElement.style.height = H + "px";
    const ctx = setupCanvas(canvas, W, H);

    const CW = W - PAD.l - PAD.r;

    NOVELS.forEach(function(n, i) {{
      const y = PAD.t + i * barH;
      // Label
      ctx.fillStyle = "#aaa";
      ctx.font = "11px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(n.title.length > 22 ? n.title.slice(0, 20) + "…" : n.title, PAD.l - 6, y + barH / 2);

      // Bar segments
      let x = PAD.l;
      CATS.forEach(function(cat) {{
        const val = n.summary[cat] || 0;
        const w = val * CW;
        ctx.fillStyle = CAT_COLOURS[cat];
        ctx.fillRect(x, y + 3, w, barH - 6);
        x += w;
      }});
    }});
  }}

  // ── TAB: Arcs ─────────────────────────────────────────────────────────────
  function buildArcsSelect() {{
    const sel = document.getElementById("arcs-select");
    while (sel.firstChild) sel.removeChild(sel.firstChild);
    NOVELS.forEach(function(n) {{
      const opt = document.createElement("option");
      opt.value = n.id;
      opt.textContent = n.title + " (" + n.year + ")";
      sel.appendChild(opt);
    }});
  }}
  buildArcsSelect();

  document.getElementById("arcs-select").addEventListener("change", drawArcs);
  document.getElementById("arcs-smooth").addEventListener("input", function() {{
    document.getElementById("arcs-smooth-val").textContent = this.value;
    drawArcs();
  }});
  document.getElementById("arcs-boundaries").addEventListener("change", drawArcs);
  document.getElementById("arcs-dominant").addEventListener("change", drawArcs);

  function drawArcs() {{
    const sel = document.getElementById("arcs-select");
    const novelId = sel.value;
    const novel = NOVELS.find(function(n) {{ return n.id === novelId; }}) || NOVELS[0];
    if (!novel) return;

    const k = parseInt(document.getElementById("arcs-smooth").value, 10);
    const showBounds = document.getElementById("arcs-boundaries").checked;
    const dominantOnly = document.getElementById("arcs-dominant").checked;

    const canvas = document.getElementById("arcs-chart");
    const container = canvas.parentElement;
    const W = container.clientWidth || 800;
    const H = 360;
    const ctx = setupCanvas(canvas, W, H);

    const PAD = {{ t: 20, r: 20, b: 40, l: 50 }};
    const CW = W - PAD.l - PAD.r;
    const CH = H - PAD.t - PAD.b;

    const positions = novel.arc.positions;
    const n = positions.length;

    // Smooth each category
    const smoothed = {{}};
    CATS.forEach(function(cat) {{
      smoothed[cat] = smooth(novel.arc[cat] || new Array(n).fill(0), k);
    }});

    function px(i) {{ return PAD.l + (positions[i] / 1.0) * CW; }}
    function py(y) {{ return PAD.t + (1 - y) * CH; }}

    // Background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, W, H);

    // Axes
    ctx.strokeStyle = "#2a3a5e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + CH);
    ctx.lineTo(PAD.l + CW, PAD.t + CH);
    ctx.stroke();

    // Y gridlines
    ctx.fillStyle = "#888";
    ctx.font = "11px system-ui";
    ctx.textAlign = "right";
    for (let v = 0; v <= 1.0; v += 0.2) {{
      const y = py(v);
      ctx.strokeStyle = "#1e2e4e";
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + CW, y); ctx.stroke();
      ctx.fillText(Math.round(v * 100) + "%", PAD.l - 6, y + 4);
    }}

    // X labels
    ctx.textAlign = "center";
    ctx.fillStyle = "#888";
    for (let v = 0; v <= 1.0; v += 0.1) {{
      ctx.fillText(Math.round(v * 100) + "%", PAD.l + v * CW, PAD.t + CH + 18);
    }}

    if (dominantOnly) {{
      // Draw dominant category at each position
      for (let i = 0; i < n; i++) {{
        let best = CATS[0], bestVal = smoothed[CATS[0]][i];
        CATS.forEach(function(cat) {{
          if (smoothed[cat][i] > bestVal) {{ best = cat; bestVal = smoothed[cat][i]; }}
        }});
        ctx.fillStyle = CAT_COLOURS[best] + "cc";
        ctx.fillRect(px(i), PAD.t, (i + 1 < n ? px(i + 1) : PAD.l + CW) - px(i), CH);
      }}
    }} else {{
      // Stacked area — draw from bottom of stack upward
      for (let ci = CATS.length - 1; ci >= 0; ci--) {{
        const cat = CATS[ci];
        // compute cumulative stack
        const stackTop = new Array(n).fill(0);
        const stackBot = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {{
          let bot = 0;
          for (let cj = CATS.length - 1; cj > ci; cj--) {{
            bot += smoothed[CATS[cj]][i];
          }}
          stackBot[i] = bot;
          stackTop[i] = bot + smoothed[cat][i];
        }}

        ctx.beginPath();
        ctx.moveTo(px(0), py(stackBot[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(px(i), py(stackBot[i]));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(px(i), py(stackTop[i]));
        ctx.closePath();
        ctx.fillStyle = CAT_COLOURS[cat] + "cc";
        ctx.fill();
      }}
    }}

    // Volume boundaries
    if (showBounds && novel.volume_boundaries && novel.volume_boundaries.length) {{
      novel.volume_boundaries.forEach(function(b) {{
        const bx = PAD.l + b * CW;
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(bx, PAD.t);
        ctx.lineTo(bx, PAD.t + CH);
        ctx.stroke();
        ctx.setLineDash([]);
      }});
    }}

    // Title annotation
    ctx.fillStyle = "#ccc";
    ctx.font = "13px system-ui";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(novel.title + " · " + novel.author + " · " + novel.year, PAD.l + 4, PAD.t + 4);
  }}

  // ── TAB: Grid ─────────────────────────────────────────────────────────────
  function buildGrid() {{
    const pillsContainer = document.getElementById("grid-pills");

    // Collect unique genres + authors
    const genres = Array.from(new Set(NOVELS.map(function(n) {{ return n.genre; }})));
    const authors = Array.from(new Set(NOVELS.map(function(n) {{ return n.author; }})));

    const activeFilters = {{ genres: new Set(genres), authors: new Set(authors) }};

    function renderPills() {{
      while (pillsContainer.firstChild) pillsContainer.removeChild(pillsContainer.firstChild);

      // Genre pills
      genres.forEach(function(g) {{
        const p = document.createElement("button");
        p.className = "pill " + (activeFilters.genres.has(g) ? "active" : "inactive");
        p.textContent = g;
        p.style.borderColor = genreColour(g);
        p.addEventListener("click", function() {{
          if (activeFilters.genres.has(g)) activeFilters.genres.delete(g);
          else activeFilters.genres.add(g);
          renderPills();
          renderCards();
        }});
        pillsContainer.appendChild(p);
      }});

      // Separator
      const sep = document.createElement("span");
      sep.textContent = "·";
      sep.style.color = "#555";
      pillsContainer.appendChild(sep);

      // Author pills
      authors.forEach(function(a) {{
        const p = document.createElement("button");
        p.className = "pill " + (activeFilters.authors.has(a) ? "active" : "inactive");
        p.textContent = a;
        p.addEventListener("click", function() {{
          if (activeFilters.authors.has(a)) activeFilters.authors.delete(a);
          else activeFilters.authors.add(a);
          renderPills();
          renderCards();
        }});
        pillsContainer.appendChild(p);
      }});
    }}

    function renderCards() {{
      const container = document.getElementById("grid-container");
      while (container.firstChild) container.removeChild(container.firstChild);

      const filtered = NOVELS.filter(function(n) {{
        return activeFilters.genres.has(n.genre) && activeFilters.authors.has(n.author);
      }});

      filtered.forEach(function(n) {{
        const card = document.createElement("div");
        card.className = "novel-card";
        card.style.borderTop = "3px solid " + genreColour(n.genre);

        const title = document.createElement("div");
        title.className = "card-title";
        title.textContent = n.title;
        card.appendChild(title);

        const meta = document.createElement("div");
        meta.className = "card-meta";
        meta.textContent = n.author + " · " + n.year + " · " + n.genre;
        card.appendChild(meta);

        // Sparkline bar
        const bar = document.createElement("div");
        bar.className = "card-bar";
        CATS.forEach(function(cat) {{
          const seg = document.createElement("div");
          const val = n.summary[cat] || 0;
          seg.style.width = (val * 100) + "%";
          seg.style.background = CAT_COLOURS[cat];
          seg.title = cat + ": " + Math.round(val * 100) + "%";
          bar.appendChild(seg);
        }});
        card.appendChild(bar);

        card.addEventListener("click", function() {{
          document.querySelector("[data-tab='arcs']").click();
          const sel = document.getElementById("arcs-select");
          for (let i = 0; i < sel.options.length; i++) {{
            if (sel.options[i].value === n.id) {{ sel.selectedIndex = i; break; }}
          }}
          drawArcs();
        }});

        container.appendChild(card);
      }});
    }}

    renderPills();
    renderCards();
  }}

  buildGrid();

  // ── TAB: Ecology ──────────────────────────────────────────────────────────
  function drawEcology() {{
    drawEcoScatter();
    drawEcoEnv();
  }}

  function drawEcoScatter() {{
    const canvas = document.getElementById("eco-scatter");
    const container = canvas.parentElement;
    const W = container.clientWidth || 380;
    const H = 300;
    const ctx = setupCanvas(canvas, W, H);

    const PAD = {{ t: 20, r: 20, b: 50, l: 55 }};
    const CW = W - PAD.l - PAD.r;
    const CH = H - PAD.t - PAD.b;

    // X = filler%, Y = fid%
    const pts = NOVELS.map(function(n) {{
      const s = n.summary;
      return {{
        x: (s.iterative || 0) + (s.description || 0),
        y: s.fid || 0,
        novel: n,
      }};
    }});

    const maxX = Math.max.apply(null, pts.map(function(p) {{ return p.x; }})) * 1.15;
    const maxY = Math.max.apply(null, pts.map(function(p) {{ return p.y; }})) * 1.15;

    function px(v) {{ return PAD.l + (v / maxX) * CW; }}
    function py(v) {{ return PAD.t + (1 - v / maxY) * CH; }}

    // Axes
    ctx.strokeStyle = "#2a3a5e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + CH);
    ctx.lineTo(PAD.l + CW, PAD.t + CH);
    ctx.stroke();

    // Gridlines
    ctx.fillStyle = "#888";
    ctx.font = "10px system-ui";
    ctx.textAlign = "right";
    for (let v = 0; v <= maxY; v += 0.05) {{
      if (v > maxY) break;
      const y = py(v);
      ctx.strokeStyle = "#1e2e4e";
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + CW, y); ctx.stroke();
      ctx.fillText(Math.round(v * 100) + "%", PAD.l - 4, y + 3);
    }}
    ctx.textAlign = "center";
    for (let v = 0; v <= maxX; v += 0.1) {{
      if (v > maxX) break;
      ctx.fillText(Math.round(v * 100) + "%", px(v), PAD.t + CH + 16);
    }}

    // Axis labels
    ctx.fillStyle = "#aaa";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("Filler % (iterative + description)", PAD.l + CW / 2, H - 6);
    ctx.save();
    ctx.translate(12, PAD.t + CH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("FID %", 0, 0);
    ctx.restore();

    // Regression
    const reg = linReg(pts);
    if (reg) {{
      const x0 = 0, x1 = maxX;
      ctx.strokeStyle = "rgba(233,69,96,0.5)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(px(x0), py(reg.m * x0 + reg.b));
      ctx.lineTo(px(x1), py(reg.m * x1 + reg.b));
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Dots
    pts.forEach(function(p) {{
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), 6, 0, Math.PI * 2);
      ctx.fillStyle = genreColour(p.novel.genre);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }});

    canvas._pts = pts.map(function(p) {{ return {{ cx: px(p.x), cy: py(p.y), novel: p.novel }}; }});
    canvas.onmousemove = function(e) {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let found = null;
      canvas._pts.forEach(function(p) {{
        if (Math.hypot(mx - p.cx, my - p.cy) < 10) found = p;
      }});
      if (found) showTip(e.clientX, e.clientY, [found.novel.title, found.novel.author + " · " + found.novel.year]);
      else hideTip();
    }};
    canvas.onmouseleave = hideTip;
  }}

  function drawEcoEnv() {{
    if (!ENV.length) return;
    const canvas = document.getElementById("eco-env");
    const container = canvas.parentElement;
    const W = container.clientWidth || 380;
    const H = 300;
    const ctx = setupCanvas(canvas, W, H);

    const PAD = {{ t: 20, r: 20, b: 50, l: 45 }};
    const CW = W - PAD.l - PAD.r;
    const CH = H - PAD.t - PAD.b;

    const decades = ENV.map(function(e) {{ return e.decade; }});
    const maxVal = Math.max.apply(null, ENV.map(function(e) {{ return e.total; }})) * 1.1;

    const n = decades.length;
    const barW = CW / n * 0.8;
    const gap = CW / n;

    function bx(i) {{ return PAD.l + i * gap + gap * 0.1; }}
    function by(v) {{ return PAD.t + (1 - v / maxVal) * CH; }}
    function bh(v) {{ return (v / maxVal) * CH; }}

    // Axes
    ctx.strokeStyle = "#2a3a5e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + CH);
    ctx.lineTo(PAD.l + CW, PAD.t + CH);
    ctx.stroke();

    // Y gridlines
    ctx.fillStyle = "#888";
    ctx.font = "10px system-ui";
    ctx.textAlign = "right";
    for (let v = 0; v <= maxVal; v += 20) {{
      const y = by(v);
      if (y < PAD.t) break;
      ctx.strokeStyle = "#1e2e4e";
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + CW, y); ctx.stroke();
      ctx.fillText(v, PAD.l - 4, y + 3);
    }}

    // Bars (stacked smoke + smell)
    ENV.forEach(function(e, i) {{
      const smellH = bh(e.smell);
      const smokeH = bh(e.smoke);

      ctx.fillStyle = "#e67e2299";
      ctx.fillRect(bx(i), by(e.smell), barW, smellH);

      ctx.fillStyle = "#95a5a699";
      ctx.fillRect(bx(i), by(e.smell) - smokeH, barW, smokeH);
    }});

    // X labels (decades, every other)
    ctx.fillStyle = "#888";
    ctx.font = "10px system-ui";
    ctx.textAlign = "center";
    decades.forEach(function(d, i) {{
      if (i % 2 === 0) {{
        ctx.fillText(d, bx(i) + barW / 2, PAD.t + CH + 16);
      }}
    }});

    // Mini legend
    ctx.fillStyle = "#e67e22";
    ctx.fillRect(PAD.l, PAD.t + CH + 28, 10, 8);
    ctx.fillStyle = "#aaa";
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";
    ctx.fillText("smell", PAD.l + 13, PAD.t + CH + 36);

    ctx.fillStyle = "#95a5a6";
    ctx.fillRect(PAD.l + 60, PAD.t + CH + 28, 10, 8);
    ctx.fillStyle = "#aaa";
    ctx.fillText("smoke", PAD.l + 73, PAD.t + CH + 36);
  }}

  // ── Init ──────────────────────────────────────────────────────────────────
  drawCentury();

}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
