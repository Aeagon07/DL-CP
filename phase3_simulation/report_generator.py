"""
Phase 3 — HTML Report Generator with Live Pipeline Animation
=============================================================
Generates a self-contained HTML report including:
  - Animated pipeline simulation (SVG + JavaScript)
  - Chart.js breach-rate distribution chart
  - P10/P50/P90 box-plot style risk table
  - Scenario comparison heatmap
  - Auto-generated narrative recommendations

Key classes:
  ReportConfig    — output path and cosmetic settings
  ReportGenerator — builds the HTML string and writes to disk
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from phase3_simulation.simpy_engine import SimulationResult, TimelineFrame
from phase3_simulation.risk_aggregator import (
    RiskAggregator,
    RiskDistribution,
    ScenarioComparison,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ReportConfig:
    output_path: str = "phase3_simulation_report.html"
    title: str = "Appian Operations Center — Monte Carlo Risk Report"
    n_runs: int = 1000
    horizon_hours: float = 8.0
    include_animation: bool = True
    open_browser: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
class ReportGenerator:
    """
    Builds a self-contained HTML report from Monte Carlo results.

    Usage:
        gen = ReportGenerator(ReportConfig())
        gen.generate_html_report(all_results, distributions, comparisons)
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._agg = RiskAggregator()

    # ------------------------------------------------------------------
    def generate_html_report(
        self,
        all_results: Dict[str, List[SimulationResult]],
        distributions: Optional[Dict[str, List[RiskDistribution]]] = None,
        comparisons: Optional[List[ScenarioComparison]] = None,
    ) -> str:
        """
        Main entry point. Returns path to the written HTML file.
        """
        if distributions is None:
            distributions = self._agg.compute_all_distributions(all_results)
        if comparisons is None:
            comparisons = self._agg.compare_all_to_baseline(all_results)

        tail = self._agg.tail_risk_summary(all_results)
        summary_table = self._agg.build_summary_table(all_results)
        viz_timeline = self._extract_viz_timeline(all_results)

        html = self._build_html(all_results, distributions, comparisons, tail, summary_table, viz_timeline)

        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info(f"Report written → {out.resolve()}")

        if self.config.open_browser:
            import webbrowser
            webbrowser.open(out.resolve().as_uri())

        return str(out.resolve())

    # ------------------------------------------------------------------
    def _extract_viz_timeline(
        self,
        all_results: Dict[str, List[SimulationResult]],
    ) -> Dict[str, List[dict]]:
        """Extract timeline frames from the baseline viz run."""
        baseline = all_results.get("baseline", list(all_results.values())[0])
        viz_id = self.config.n_runs // 2
        out: Dict[str, List[dict]] = {}
        for r in baseline:
            if r.run_id == viz_id and r.timeline:
                out[r.queue_name] = [
                    {
                        "t": f.t_minutes,
                        "wip": f.wip,
                        "in_service": f.in_service,
                        "queue_length": f.queue_length,
                        "completed": f.completed,
                        "breached": f.breached,
                    }
                    for f in r.timeline
                ]
        return out

    # ------------------------------------------------------------------
    def _build_html(
        self,
        all_results,
        distributions,
        comparisons,
        tail,
        summary_table,
        viz_timeline,
    ) -> str:
        scenarios = list(all_results.keys())
        queues = sorted({r.queue_name for results in all_results.values() for r in results})

        breach_data = self._build_breach_chart_data(all_results, queues)
        heatmap_data = self._build_heatmap_data(all_results, scenarios, queues)
        recs_html = self._render_recommendations(tail)
        table_html = self._render_summary_table(summary_table)
        animation_js = self._build_animation_js(viz_timeline, queues)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_runs = self.config.n_runs

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{self.config.title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --accent: #4361ee; --accent2: #06d6a0; --danger: #f72585;
    --warn: #f4a261; --text: #c9d1d9; --muted: #8b949e;
    --font: 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font); }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 1.8rem; color: #fff; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 32px; }}
  h2 {{ font-size: 1.2rem; color: #fff; margin-bottom: 16px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }}
  .stat-val {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .stat-lbl {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
  .badge-better {{ background: #06d6a020; color: var(--accent2); border: 1px solid var(--accent2); }}
  .badge-worse  {{ background: #f7258520; color: var(--danger); border: 1px solid var(--danger); }}
  .badge-neutral{{ background: #8b949e20; color: var(--muted); border: 1px solid var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #21262d; color: var(--muted); padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); position: sticky; top: 0; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .tbl-wrap {{ max-height: 420px; overflow-y: auto; border-radius: 8px; border: 1px solid var(--border); }}
  /* Pipeline Animation */
  #pipeline-section {{ margin-bottom: 24px; }}
  #pipeline-canvas {{ width: 100%; border-radius: 12px; background: #0d1117; border: 1px solid var(--border); }}
  .queue-lane {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .lane-label {{ width: 160px; font-size: 0.78rem; color: var(--muted); text-align: right; flex-shrink: 0; }}
  .lane-bar-wrap {{ flex: 1; background: #21262d; border-radius: 6px; height: 28px; overflow: hidden; position: relative; }}
  .lane-bar {{ height: 100%; border-radius: 6px; transition: width 0.4s ease; display: flex; align-items: center; padding-left: 8px; font-size: 0.75rem; font-weight: 600; color: #fff; white-space: nowrap; }}
  .lane-stats {{ width: 120px; font-size: 0.72rem; color: var(--muted); text-align: left; flex-shrink: 0; }}
  #anim-controls {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }}
  .btn {{ background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 8px 18px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: opacity 0.2s; }}
  .btn:hover {{ opacity: 0.85; }}
  .btn-outline {{ background: transparent; border: 1px solid var(--border); color: var(--text); }}
  .time-display {{ font-size: 0.85rem; color: var(--muted); margin-left: auto; }}
  .rec {{ padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; font-size: 0.88rem; line-height: 1.5; }}
  .rec-warn  {{ background: #f4a26115; border-left: 3px solid var(--warn); }}
  .rec-danger{{ background: #f7258515; border-left: 3px solid var(--danger); }}
  .rec-ok    {{ background: #06d6a015; border-left: 3px solid var(--accent2); }}
  .heat-cell {{ display: inline-block; width: 100%; text-align: center; padding: 6px 2px; border-radius: 4px; font-size: 0.78rem; font-weight: 600; }}
  canvas {{ max-height: 340px; }}
  @media(max-width:768px) {{ .grid-2,.grid-3{{ grid-template-columns:1fr; }} .lane-label{{width:100px;}} }}
</style>
</head>
<body>
<div class="container">

  <h1>🎲 Monte Carlo Simulation Report</h1>
  <p class="subtitle">{self.config.title} &nbsp;·&nbsp; Generated {ts} &nbsp;·&nbsp; {n_runs:,} runs per scenario</p>

  <!-- KPI Cards -->
  <div class="grid-3">
    <div class="card">
      <div class="stat-val">{tail['worst_case_breach_rate']*100:.1f}%</div>
      <div class="stat-lbl">Worst-Case P90 Breach Rate</div>
    </div>
    <div class="card">
      <div class="stat-val">{tail['expected_breach_rate']*100:.1f}%</div>
      <div class="stat-lbl">Expected Avg Breach Rate</div>
    </div>
    <div class="card">
      <div class="stat-val">{len(tail['queues_at_risk'])}</div>
      <div class="stat-lbl">Queues At Risk (&gt;15% P90)</div>
    </div>
  </div>

  <!-- Live Pipeline Animation -->
  <div id="pipeline-section" class="card">
    <h2>⚡ Live Queue Simulation — Baseline Run</h2>
    <div id="anim-controls">
      <button class="btn" id="btn-play">▶ Play</button>
      <button class="btn btn-outline" id="btn-reset">↺ Reset</button>
      <span class="time-display" id="time-display">T = 0 min</span>
    </div>
    <div id="pipeline-lanes"></div>
  </div>

  <!-- Charts -->
  <div class="grid-2">
    <div class="card">
      <h2>📊 Breach Rate Distribution by Queue</h2>
      <canvas id="breachChart"></canvas>
    </div>
    <div class="card">
      <h2>🗺 Scenario Risk Heatmap (P50 Breach %)</h2>
      <div id="heatmap-container"></div>
    </div>
  </div>

  <!-- Recommendations -->
  <div class="card" style="margin-bottom:24px">
    <h2>🧠 Auto-Generated Risk Recommendations</h2>
    {recs_html}
  </div>

  <!-- Summary Table -->
  <div class="card">
    <h2>📋 Full Risk Summary Table (P10 / P50 / P90 Breach Rate)</h2>
    <div class="tbl-wrap">{table_html}</div>
  </div>

</div>

<script>
// ── Animation Data ──────────────────────────────────────────────────
const QUEUE_COLORS = {{
  "Document Review":     "#4361ee",
  "Compliance Check":    "#7209b7",
  "Payment Processing":  "#f72585",
  "Customer Onboarding": "#06d6a0",
  "Risk Assessment":     "#fb8500",
  "Audit Preparation":   "#e63946",
}};

const vizData = {json.dumps(viz_timeline)};
const queueNames = {json.dumps(queues)};

{animation_js}

// ── Breach Chart ────────────────────────────────────────────────────
const breachData = {json.dumps(breach_data)};
const bCtx = document.getElementById('breachChart').getContext('2d');
new Chart(bCtx, {{
  type: 'bar',
  data: {{
    labels: breachData.queues,
    datasets: breachData.datasets,
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 11 }} }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{(ctx.raw*100).toFixed(1)}}%`
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{
        ticks: {{ color: '#8b949e', callback: v => (v*100).toFixed(0)+'%' }},
        grid: {{ color: '#21262d' }},
        title: {{ display: true, text: 'Breach Rate', color: '#8b949e' }}
      }}
    }}
  }}
}});

// ── Heatmap ─────────────────────────────────────────────────────────
(function() {{
  const hd = {json.dumps(heatmap_data)};
  const cont = document.getElementById('heatmap-container');
  let html = '<table style="width:100%;border-collapse:collapse;font-size:0.78rem;">';
  html += '<tr><th style="padding:6px;color:#8b949e;text-align:left;">Queue</th>';
  hd.scenarios.forEach(s => {{
    html += `<th style="padding:6px;color:#8b949e;text-align:center;font-size:0.72rem;">${{s.replace(/_/g,' ')}}</th>`;
  }});
  html += '</tr>';
  hd.queues.forEach((q,qi) => {{
    html += `<tr><td style="padding:6px 8px;color:#c9d1d9;white-space:nowrap;">${{q}}</td>`;
    hd.scenarios.forEach((s,si) => {{
      const val = hd.values[qi][si];
      const pct = (val*100).toFixed(1);
      const r = Math.min(255, Math.round(val * 800));
      const g = Math.max(0, Math.round(150 - val * 600));
      const bg = `rgba(${{r}},${{g}},60,0.6)`;
      html += `<td style="padding:4px;"><div class="heat-cell" style="background:${{bg}}">${{pct}}%</div></td>`;
    }});
    html += '</tr>';
  }});
  html += '</table>';
  cont.innerHTML = html;
}})();
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    def _build_breach_chart_data(self, all_results, queues):
        COLORS = [
            "#4361ee", "#f72585", "#06d6a0", "#f4a261",
            "#7209b7", "#e63946", "#fb8500", "#4cc9f0",
            "#e76f51", "#2ec4b6",
        ]
        datasets = []
        for i, (scenario, results) in enumerate(all_results.items()):
            by_queue: Dict[str, list] = {}
            for r in results:
                by_queue.setdefault(r.queue_name, []).append(r.breach_rate)
            import numpy as np
            data = [float(np.median(by_queue.get(q, [0]))) for q in queues]
            datasets.append({
                "label": scenario.replace("_", " "),
                "data": data,
                "backgroundColor": COLORS[i % len(COLORS)] + "99",
                "borderColor": COLORS[i % len(COLORS)],
                "borderWidth": 1,
            })
        return {"queues": queues, "datasets": datasets}

    # ------------------------------------------------------------------
    def _build_heatmap_data(self, all_results, scenarios, queues):
        import numpy as np
        values = []
        for q in queues:
            row = []
            for s in scenarios:
                results = all_results.get(s, [])
                brs = [r.breach_rate for r in results if r.queue_name == q]
                row.append(round(float(np.median(brs)) if brs else 0.0, 4))
            values.append(row)
        return {"scenarios": scenarios, "queues": queues, "values": values}

    # ------------------------------------------------------------------
    def _render_recommendations(self, tail) -> str:
        recs = tail.get("recommendations", [])
        if not recs:
            return '<p style="color:var(--muted)">No significant risks detected.</p>'
        html = ""
        for r in recs:
            if "⚠️" in r:
                cls = "rec-warn"
            elif "🔴" in r:
                cls = "rec-danger"
            else:
                cls = "rec-ok"
            html += f'<div class="rec {cls}">{r}</div>'
        return html

    # ------------------------------------------------------------------
    def _render_summary_table(self, rows) -> str:
        html = """<table>
<thead><tr>
  <th>Scenario</th><th>Queue</th>
  <th>Breach P10</th><th>Breach P50</th><th>Breach P90</th>
  <th>Utilization P50</th><th>Throughput P50</th>
</tr></thead><tbody>"""
        prev_scenario = None
        for row in rows:
            s = row["scenario"]
            q = row["queue"]
            b10, b50, b90 = row["breach_p10"], row["breach_p50"], row["breach_p90"]
            util = row["utilization_p50"]
            tput = row["throughput_p50"]

            color_b90 = "#f72585" if b90 > 20 else ("#f4a261" if b90 > 10 else "#06d6a0")
            disp_s = s.replace("_", " ") if s != prev_scenario else ""
            prev_scenario = s

            html += f"""<tr>
  <td style="color:#8b949e;font-size:0.78rem">{disp_s}</td>
  <td>{q}</td>
  <td>{b10:.1f}%</td>
  <td>{b50:.1f}%</td>
  <td style="color:{color_b90};font-weight:600">{b90:.1f}%</td>
  <td>{util:.1f}%</td>
  <td>{tput:.1f}</td>
</tr>"""
        html += "</tbody></table>"
        return html

    # ------------------------------------------------------------------
    def _build_animation_js(self, viz_timeline, queues) -> str:
        if not viz_timeline:
            return "// No timeline data available for animation."

        max_wip = max(
            (f["wip"] for frames in viz_timeline.values() for f in frames),
            default=1
        )
        max_wip = max(max_wip, 1)

        return f"""
(function() {{
  const maxWip = {max_wip};
  const lanesDiv = document.getElementById('pipeline-lanes');
  const timeDisp = document.getElementById('time-display');
  let frameIdx = 0;
  let animTimer = null;
  let playing = false;

  // Build lane DOM
  queueNames.forEach(q => {{
    const color = QUEUE_COLORS[q] || '#4361ee';
    const div = document.createElement('div');
    div.className = 'queue-lane';
    div.innerHTML = `
      <div class="lane-label">${{q}}</div>
      <div class="lane-bar-wrap">
        <div class="lane-bar" id="bar-${{q.replace(/ /g,'_')}}" style="width:0%;background:${{color}}">0</div>
      </div>
      <div class="lane-stats" id="stats-${{q.replace(/ /g,'_')}}">WIP:0 Q:0</div>
    `;
    lanesDiv.appendChild(div);
  }});

  function renderFrame(idx) {{
    queueNames.forEach(q => {{
      const key = q;
      const frames = vizData[key];
      if (!frames || frames.length === 0) return;
      const frame = frames[Math.min(idx, frames.length - 1)];
      const pct = Math.min(100, (frame.wip / maxWip) * 100);
      const barId = 'bar-' + q.replace(/ /g, '_');
      const statsId = 'stats-' + q.replace(/ /g, '_');
      const bar = document.getElementById(barId);
      const stats = document.getElementById(statsId);
      if (bar) {{
        bar.style.width = pct + '%';
        bar.textContent = frame.wip + ' WIP';
      }}
      if (stats) {{
        stats.textContent = `Svc:${{frame.in_service}} Q:${{frame.queue_length}} Br:${{frame.breached}}`;
      }}
      const t = frames[Math.min(idx, frames.length-1)].t;
      timeDisp.textContent = `T = ${{t}} min`;
    }});
  }}

  function step() {{
    const maxFrames = Math.max(...queueNames.map(q => (vizData[q]||[]).length));
    if (frameIdx >= maxFrames) {{
      clearInterval(animTimer);
      playing = false;
      document.getElementById('btn-play').textContent = '▶ Play';
      return;
    }}
    renderFrame(frameIdx++);
  }}

  document.getElementById('btn-play').addEventListener('click', function() {{
    if (playing) {{
      clearInterval(animTimer);
      playing = false;
      this.textContent = '▶ Play';
    }} else {{
      playing = true;
      this.textContent = '⏸ Pause';
      animTimer = setInterval(step, 80);
    }}
  }});

  document.getElementById('btn-reset').addEventListener('click', function() {{
    clearInterval(animTimer);
    playing = false;
    frameIdx = 0;
    document.getElementById('btn-play').textContent = '▶ Play';
    renderFrame(0);
  }});

  renderFrame(0);
}})();
"""
