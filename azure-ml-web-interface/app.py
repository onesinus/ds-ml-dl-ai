import logging
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forecast-app")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
EXOGENOUS_COLS = ["ST (mm)", "WR (mm)", "ST (hr)", "NT (hr)", "WR (hr)", "NT (mm)"]

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading model (this can take a minute)...")
        _model = mlflow.pyfunc.load_model(MODEL_PATH)
        logger.info("Model loaded.")
    return _model


def forecast_range(start_date, end_date):
    fm = get_model()._model_impl.forecasting_model
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        raise ValueError("start_date must be on or before end_date")

    x = pd.DataFrame({"Date": dates})
    for c in EXOGENOUS_COLS:
        x[c] = float("nan")

    y, _ = fm.forecast(X_pred=x, y_pred=None, ignore_data_errors=True)

    results = []
    for d, v in zip(dates, y):
        results.append({"date": d.strftime("%Y-%m-%d"), "predicted_nt_mm": round(float(v), 2)})
    return results


class PredictRequest(BaseModel):
    start_date: str = Field(description="Start date, format YYYY-MM-DD")
    end_date: str = Field(description="End date, format YYYY-MM-DD")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="NT Rainfall Forecast", lifespan=lifespan)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NT Rainfall Forecast</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: .25rem; }
  .sub { opacity: .7; margin-bottom: 1.5rem; }
  form { display: flex; gap: .75rem; flex-wrap: wrap; align-items: flex-end; }
  label { font-size: .85rem; display: flex; flex-direction: column; gap: .25rem; }
  input { padding: .5rem; border-radius: 8px; border: 1px solid #888; }
  button { padding: .55rem 1.1rem; border-radius: 8px; border: 0; background: #0b6; color: #fff; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .6; cursor: wait; }
  .error { color: #c33; margin-top: 1rem; }
  table { border-collapse: collapse; margin-top: 1.5rem; width: 100%; }
  th, td { padding: .5rem .75rem; text-align: left; border-bottom: 1px solid #555; }
  th { font-weight: 600; }
  .bar { height: 18px; background: #0b6; border-radius: 4px; min-width: 2px; }
  #chart { margin-top: 1.5rem; }
</style>
</head>
<body>
  <h1>Rainfall Forecast (NT)</h1>
  <p class="sub">Predicted daily rainfall at the NT station, in millimetres.</p>
  <form id="form">
    <label>From <input type="date" id="start" required></label>
    <label>To <input type="date" id="end" required></label>
    <button type="submit">Forecast</button>
  </form>
  <div id="error" class="error"></div>
  <div id="chart"></div>
  <table id="table" style="display:none">
    <thead><tr><th>Date</th><th>Rainfall (mm)</th></tr></thead>
    <tbody></tbody>
  </table>

<script>
  const today = new Date();
  const fmt = d => d.toISOString().slice(0, 10);
  const tomorrow = new Date(today.getTime() + 86400000);
  const in7 = new Date(today.getTime() + 7 * 86400000);
  document.getElementById('start').value = fmt(tomorrow);
  document.getElementById('end').value = fmt(in7);

  document.getElementById('form').addEventListener('submit', async e => {
    e.preventDefault();
    const btn = document.querySelector('button');
    btn.disabled = true;
    document.getElementById('error').textContent = '';
    document.getElementById('chart').innerHTML = '';
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: document.getElementById('start').value,
          end_date: document.getElementById('end').value,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);

      const tbody = document.querySelector('#table tbody');
      tbody.innerHTML = '';
      let rows = '';
      const max = Math.max(...data.map(d => d.predicted_nt_mm));
      data.forEach(d => {
        const pct = max > 0 ? Math.round((d.predicted_nt_mm / max) * 100) : 0;
        rows += `<tr><td>${d.date}</td><td>${d.predicted_nt_mm} <span class="bar" style="width:${pct}px;display:inline-block;vertical-align:middle"></span></td></tr>`;
      });
      tbody.innerHTML = rows;
      document.getElementById('table').style.display = '';
    } catch (err) {
      document.getElementById('error').textContent = 'Error: ' + err.message;
    } finally {
      btn.disabled = false;
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        start = pd.Timestamp(req.start_date)
        end = pd.Timestamp(req.end_date)
    except Exception:
        raise HTTPException(400, "Dates must be in YYYY-MM-DD format.")

    if start > end:
        raise HTTPException(400, "start_date must be on or before end_date.")

    try:
        return forecast_range(start, end)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
