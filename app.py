import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_NAMES = ["lstm_model", "gru_model", "bilstm_model",
               "cnn_lstm_model", "transformer_model"]

def load_model(name):
    for ext in (".keras", ".h5"):
        path = f"models/{name}{ext}"
        if os.path.exists(path):
            print(f"  Loading {path}")
            return tf.keras.models.load_model(path)
    raise FileNotFoundError(f"No saved model found for: {name}")

print("Loading models...")
MODELS   = {n: load_model(n) for n in MODEL_NAMES}
SCALER_X = joblib.load("models/scaler_X.pkl")
SCALER_Y = joblib.load("models/scaler_y.pkl")
SEQ_LEN  = 24
print("All models loaded successfully.")

FEATURE_COLS = [
    "CPI", "CPI_lag1", "CPI_lag2", "CPI_lag3",
    "CPI_lag6", "CPI_lag12",
    "CPI_MA3", "CPI_MA6", "CPI_MA12",
    "CPI_pct1", "CPI_pct3", "CPI_pct12",
    "CPI_std3", "CPI_std6",
    "Month", "Quarter"
]

# ── Feature Engineering ───────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["CPI"]

    df["CPI_lag1"]  = c.shift(1)
    df["CPI_lag2"]  = c.shift(2)
    df["CPI_lag3"]  = c.shift(3)
    df["CPI_lag6"]  = c.shift(6)
    df["CPI_lag12"] = c.shift(12)

    df["CPI_MA3"]  = c.rolling(3).mean()
    df["CPI_MA6"]  = c.rolling(6).mean()
    df["CPI_MA12"] = c.rolling(12).mean()

    df["CPI_pct1"]  = c.pct_change(1)  * 100
    df["CPI_pct3"]  = c.pct_change(3)  * 100
    df["CPI_pct12"] = c.pct_change(12) * 100

    df["CPI_std3"] = c.rolling(3).std()
    df["CPI_std6"] = c.rolling(6).std()

    df["Month"]   = df.index.month
    df["Quarter"] = df.index.quarter
    return df

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_inflation(steps=6, model_name="ensemble"):
    df = pd.read_csv("data/cpi_data.csv", index_col=0, parse_dates=True)
    df = add_features(df).dropna()

    X = SCALER_X.transform(df[FEATURE_COLS].values)

    future_preds = []
    seq = X[-SEQ_LEN:].copy()

    for _ in range(steps):
        inp  = seq[np.newaxis, :]
        if model_name == "ensemble":
            raw = np.mean([
                SCALER_Y.inverse_transform(
                    m.predict(inp, verbose=0))[0][0]
                for m in MODELS.values()
            ])
        else:
            raw = float(SCALER_Y.inverse_transform(
                MODELS[model_name].predict(inp, verbose=0))[0][0])
        future_preds.append(float(raw))
        new_row      = seq[-1].copy()
        new_row[0]   = float(SCALER_X.transform(
                         np.array([[raw] + [0]*(len(FEATURE_COLS)-1)])
                       )[0][0])
        seq = np.vstack([seq[1:], new_row])

    current_cpi   = float(df["CPI"].iloc[-1])
    current_infl  = float(df["CPI_pct12"].iloc[-1])
    hist_cpi      = df["CPI"].tail(36).values.flatten().tolist()
    hist_infl     = df["CPI_pct12"].tail(36).values.flatten().tolist()
    hist_dates    = [str(d.date()) for d in df.index[-36:]]

    last_date  = df.index[-1]
    pred_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps, freq="MS"
    )
    pred_dates_str = [str(d.date()) for d in pred_dates]

    pred_infl = [
        round((future_preds[i] / df["CPI"].iloc[-12+i] - 1) * 100, 4)
        if i < 12 else
        round((future_preds[i] / future_preds[i-12] - 1) * 100, 4)
        for i in range(steps)
    ]

    trend = "RISING" if pred_infl[-1] > current_infl else (
            "FALLING" if pred_infl[-1] < current_infl else "STABLE")

    return {
        "current_cpi"    : round(current_cpi, 4),
        "current_infl"   : round(current_infl, 4),
        "pred_cpi"       : [round(p, 4) for p in future_preds],
        "pred_infl"      : pred_infl,
        "pred_dates"     : pred_dates_str,
        "hist_dates"     : hist_dates,
        "hist_cpi"       : [round(v, 4) for v in hist_cpi],
        "hist_infl"      : [round(v, 4) for v in hist_infl],
        "trend"          : trend,
        "model_used"     : model_name,
        "steps"          : steps,
    }

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Inflation Predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#0b0d15;color:#dde1f0;min-height:100vh}
    header{background:#131726;padding:1rem 2rem;border-bottom:1px solid #1f2540;
           display:flex;align-items:center;gap:.8rem}
    header h1{font-size:1.35rem;color:#7b9ef4;font-weight:700}
    header span{font-size:1.6rem}
    .container{max-width:1100px;margin:2rem auto;padding:0 1.25rem}
    .card{background:#131726;border:1px solid #1f2540;border-radius:14px;
          padding:1.5rem;margin-bottom:1.5rem}
    .card-title{font-size:.75rem;color:#55607a;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:1rem}
    .controls{display:flex;gap:.75rem;flex-wrap:wrap;align-items:flex-end}
    .ctrl-group{display:flex;flex-direction:column;gap:.35rem;flex:1;min-width:140px}
    .ctrl-group label{font-size:.75rem;color:#55607a;text-transform:uppercase;letter-spacing:.06em}
    .controls input,.controls select{
      background:#0b0d15;border:1px solid #2a2f45;color:#dde1f0;
      padding:.65rem 1rem;border-radius:8px;font-size:1rem;outline:none;
      transition:border .2s;width:100%}
    .controls input:focus,.controls select:focus{border-color:#7b9ef4}
    .controls button{background:#7b9ef4;color:#0b0d15;border:none;padding:.65rem 1.75rem;
      border-radius:8px;font-size:1rem;font-weight:700;cursor:pointer;
      transition:background .2s;align-self:flex-end}
    .controls button:hover{background:#5d7ee0}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:1rem}
    .metric{background:#0b0d15;border:1px solid #1f2540;border-radius:10px;
            padding:1rem .75rem;text-align:center}
    .metric .lbl{font-size:.7rem;color:#55607a;text-transform:uppercase;
                 letter-spacing:.07em;margin-bottom:.4rem}
    .metric .val{font-size:1.45rem;font-weight:700}
    .green{color:#3ecf8e}.red{color:#e05c5c}.yellow{color:#f0c040}
    .badge{display:inline-block;padding:.3rem 1rem;border-radius:20px;font-weight:700}
    .RISING{background:rgba(224,92,92,.12);color:#e05c5c;border:1px solid #e05c5c}
    .FALLING{background:rgba(62,207,142,.12);color:#3ecf8e;border:1px solid #3ecf8e}
    .STABLE{background:rgba(240,192,64,.12);color:#f0c040;border:1px solid #f0c040}
    .charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
    @media(max-width:700px){.charts-grid{grid-template-columns:1fr}}
    canvas{width:100%!important;max-height:300px}
    .loader{display:none;color:#7b9ef4;padding:.75rem 0}
    .error{color:#e05c5c;padding:.75rem 0;min-height:1.5rem}
    .pred-table{width:100%;border-collapse:collapse;font-size:.9rem;margin-top:.5rem}
    .pred-table th{color:#55607a;text-transform:uppercase;font-size:.72rem;
                   letter-spacing:.06em;padding:.5rem .75rem;text-align:left;
                   border-bottom:1px solid #1f2540}
    .pred-table td{padding:.5rem .75rem;border-bottom:1px solid #1a1e30}
    .pred-table tr:last-child td{border-bottom:none}
  </style>
</head>
<body>
<header><span>📈</span><h1>Deep Learning Inflation Predictor</h1></header>
<div class="container">

  <div class="card">
    <p class="card-title">Forecast settings</p>
    <div class="controls">
      <div class="ctrl-group">
        <label>Forecast horizon</label>
        <select id="steps">
          <option value="3">3 months</option>
          <option value="6" selected>6 months</option>
          <option value="12">12 months</option>
          <option value="24">24 months</option>
        </select>
      </div>
      <div class="ctrl-group">
        <label>Model</label>
        <select id="modelSel">
          <option value="ensemble">Ensemble — all models</option>
          <option value="lstm_model">LSTM</option>
          <option value="gru_model">GRU</option>
          <option value="bilstm_model">Bi-LSTM</option>
          <option value="cnn_lstm_model">CNN-LSTM</option>
          <option value="transformer_model">Transformer</option>
        </select>
      </div>
      <button onclick="runPredict()">Forecast</button>
    </div>
    <div class="loader" id="loader">⏳ Running model forecast…</div>
    <div class="error"  id="errMsg"></div>
  </div>

  <div class="card" id="resCard" style="display:none">
    <p class="card-title" id="resTitle"></p>
    <div class="metrics" id="metrics"></div>
  </div>

  <div class="charts-grid">
    <div class="card" id="cpiCard" style="display:none">
      <p class="card-title">CPI — history + forecast</p>
      <canvas id="cpiChart"></canvas>
    </div>
    <div class="card" id="inflCard" style="display:none">
      <p class="card-title">Inflation rate (YoY %) — history + forecast</p>
      <canvas id="inflChart"></canvas>
    </div>
  </div>

  <div class="card" id="tableCard" style="display:none">
    <p class="card-title">Month-by-month forecast</p>
    <table class="pred-table">
      <thead>
        <tr>
          <th>Date</th>
          <th>Forecast CPI</th>
          <th>Inflation Rate (YoY %)</th>
          <th>vs Current</th>
        </tr>
      </thead>
      <tbody id="predBody"></tbody>
    </table>
  </div>

</div>
<script>
let charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); charts[id] = null; }
}

function makeChart(id, labels, datasets, yLabel) {
  destroyChart(id);
  charts[id] = new Chart(document.getElementById(id).getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode:"index", intersect:false },
      plugins: {
        legend: { labels:{ color:"#9aa3b8", font:{size:11} } },
        tooltip: {
          backgroundColor:"#1a1f35", titleColor:"#9aa3b8", bodyColor:"#dde1f0",
          callbacks: { label: ctx => "  "+ctx.dataset.label+": "+ctx.parsed.y.toFixed(4) }
        }
      },
      scales: {
        x: { ticks:{color:"#55607a",maxTicksLimit:8,maxRotation:0},
             grid:{color:"rgba(255,255,255,.04)"} },
        y: { title:{display:true,text:yLabel,color:"#55607a",font:{size:11}},
             ticks:{color:"#55607a"},
             grid:{color:"rgba(255,255,255,.04)"} }
      }
    }
  });
}

function metric(lbl, val, cls="") {
  return '<div class="metric"><div class="lbl">'+lbl+'</div>'
       + '<div class="val '+cls+'">'+val+'</div></div>';
}
function metricBadge(lbl, signal) {
  return '<div class="metric"><div class="lbl">'+lbl+'</div>'
       + '<div class="val"><span class="badge '+signal+'">'+signal+'</span></div></div>';
}

async function runPredict() {
  const steps = document.getElementById("steps").value;
  const model = document.getElementById("modelSel").value;

  document.getElementById("loader").style.display   = "block";
  document.getElementById("errMsg").textContent     = "";
  ["resCard","cpiCard","inflCard","tableCard"].forEach(
    id => document.getElementById(id).style.display = "none"
  );

  try {
    const res  = await fetch("/predict?steps="+steps+"&model="+model);
    const data = await res.json();
    if (data.error) {
      document.getElementById("errMsg").textContent = "⚠️ "+data.error;
      return;
    }

    const inflDir  = data.pred_infl[data.pred_infl.length-1] > data.current_infl;
    const inflColor = inflDir ? "red" : "green";

    document.getElementById("resTitle").textContent =
      "Forecast: "+steps+" months  |  model: "+data.model_used;

    document.getElementById("metrics").innerHTML =
      metric("Current CPI",        data.current_cpi.toFixed(4), "")           +
      metric("Current Inflation",  data.current_infl.toFixed(2)+"%", "")      +
      metric("Forecast CPI",       data.pred_cpi[data.pred_cpi.length-1].toFixed(4),    inflColor) +
      metric("Forecast Inflation", data.pred_infl[data.pred_infl.length-1].toFixed(2)+"%", inflColor) +
      metricBadge("Trend", data.trend);

    document.getElementById("resCard").style.display = "block";

    // CPI Chart
    const cpiLabels   = [...data.hist_dates, ...data.pred_dates];
    const histCpiData = data.hist_cpi.map((v,i) => ({x:data.hist_dates[i], y:v}));
    const predCpiData = data.pred_dates.map((d,i) => ({x:d, y:data.pred_cpi[i]}));
    const joinCpi     = [{x:data.hist_dates[data.hist_dates.length-1],
                          y:data.hist_cpi[data.hist_cpi.length-1]},
                         ...predCpiData];

    makeChart("cpiChart", cpiLabels, [
      { label:"Historical CPI", data:histCpiData,
        borderColor:"#7b9ef4", backgroundColor:"rgba(123,158,244,.06)",
        borderWidth:1.8, fill:true, tension:0.35,
        pointRadius:0, pointHoverRadius:4 },
      { label:"Forecast CPI", data:joinCpi,
        borderColor:"#e05c5c", backgroundColor:"rgba(224,92,92,.06)",
        borderWidth:2, fill:false, tension:0.35, borderDash:[6,3],
        pointRadius:3, pointHoverRadius:5,
        pointBackgroundColor:"#e05c5c" }
    ], "CPI Index");

    document.getElementById("cpiCard").style.display = "block";

    // Inflation chart
    const inflLabels    = [...data.hist_dates, ...data.pred_dates];
    const histInflData  = data.hist_infl.map((v,i)=>({x:data.hist_dates[i],y:v}));
    const predInflData  = data.pred_dates.map((d,i)=>({x:d,y:data.pred_infl[i]}));
    const joinInfl      = [{x:data.hist_dates[data.hist_dates.length-1],
                            y:data.hist_infl[data.hist_infl.length-1]},
                           ...predInflData];

    makeChart("inflChart", inflLabels, [
      { label:"Historical Inflation", data:histInflData,
        borderColor:"#7b9ef4", backgroundColor:"rgba(123,158,244,.06)",
        borderWidth:1.8, fill:true, tension:0.35,
        pointRadius:0, pointHoverRadius:4 },
      { label:"Forecast Inflation", data:joinInfl,
        borderColor:"#f0c040", backgroundColor:"rgba(240,192,64,.06)",
        borderWidth:2, fill:false, tension:0.35, borderDash:[6,3],
        pointRadius:3, pointHoverRadius:5,
        pointBackgroundColor:"#f0c040" }
    ], "Inflation Rate (%)");

    document.getElementById("inflCard").style.display = "block";

    // Table
    const tbody = document.getElementById("predBody");
    tbody.innerHTML = "";
    data.pred_dates.forEach((d,i) => {
      const diff  = (data.pred_infl[i] - data.current_infl).toFixed(2);
      const up    = parseFloat(diff) >= 0;
      const arrow = up ? "▲" : "▼";
      const cls   = up ? "red" : "green";
      tbody.innerHTML +=
        "<tr><td>"+d+"</td><td>"+data.pred_cpi[i].toFixed(4)+"</td>"
        +"<td>"+data.pred_infl[i].toFixed(2)+"%</td>"
        +"<td class='"+cls+"'>"+arrow+" "+Math.abs(diff)+"%</td></tr>";
    });
    document.getElementById("tableCard").style.display = "block";

  } catch(err) {
    document.getElementById("errMsg").textContent = "Request failed: "+err.message;
  } finally {
    document.getElementById("loader").style.display = "none";
  }
}

runPredict();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict")
def predict():
    steps      = int(request.args.get("steps", 6))
    model_name = request.args.get("model", "ensemble").strip()
    try:
        return jsonify(predict_inflation(steps, model_name))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)