"""Salary Predictor website.  Run:  python app.py  ->  http://127.0.0.1:5000"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
MODEL = joblib.load(BASE_DIR / "model" / "salary_model.joblib")
with (BASE_DIR / "model" / "meta.json").open(encoding="utf-8") as meta_file:
    META = json.load(meta_file)
SKILL_COLS = META["skill_cols"]

_D = pd.read_csv(BASE_DIR / "cleaned_job_market_data_v2.csv")
_D = _D[_D.use_for_model]

# Keep the controls tied to columns that are actually present and represented
# in the training rows, rather than exposing catalogue values with no data.
META["domains"] = [d for d in META["domains"] if d in _D["Domain"].dropna().unique()]
META["levels"] = [l for l in META["levels"] if l in _D["Seniority"].dropna().unique()]
META["skills"] = [
    s for s in META["skills"]
    if "Skill_" + s in _D.columns and _D["Skill_" + s].fillna(0).sum() > 0
]
SKILL_COLS = ["Skill_" + s for s in META["skills"]]
META["skill_cols"] = SKILL_COLS
META["experience_ranges"] = {
    "Entry": {"min": 0, "max": 1, "label": "0-1 years"},
    "Mid": {"min": 2, "max": 4, "label": "2-4 years"},
    "Senior": {"min": 5, "max": 7, "label": "5-7 years"},
    "Lead/Director": {"min": 8, "max": 20, "label": "8-20 years"},
}

def skill_gains(level, years, domain, skills):
    """Average effect of adding each skill, over real postings in the chosen domain
    (more stable than one hypothetical profile, which small data makes noisy)."""
    base = _D[_D.Domain == domain]
    if len(base) < 10: base = _D
    X = base[SKILL_COLS].copy()
    for s in skills: X["Skill_" + s] = 1
    X["Seniority"], X["Domain"], X["Experience_Years"] = level, domain, float(years)
    X["Skill_Count"] = X[SKILL_COLS].sum(axis=1)
    p0 = MODEL.predict(X).mean(); out = []
    for s in META["skills"]:
        if s in skills: continue
        X2 = X.copy(); X2["Skill_" + s] = 1; X2["Skill_Count"] = X2[SKILL_COLS].sum(axis=1)
        out.append({"skill": s, "delta": float(MODEL.predict(X2).mean() - p0)})
    return sorted(out, key=lambda g: -g["delta"])[:5]

def build_row(level, years, domain, skills):
    row = {"Seniority": level, "Domain": domain,
           "Experience_Years": float(years), "Skill_Count": len(skills)}
    for c in SKILL_COLS:
        row[c] = int(c.replace("Skill_", "") in skills)
    return pd.DataFrame([row])

@app.get("/")
def index():
    return render_template("index.html", meta=META)

@app.post("/api/predict")
def predict():
    d = request.get_json(force=True)
    level, domain = d.get("level"), d.get("domain")
    skills = [s for s in d.get("skills", []) if s in META["skills"]]
    try:
        years = min(max(float(d.get("years", 0)), 0), 30)
    except (TypeError, ValueError):
        return jsonify(error="Years of experience must be a number"), 400
    if level not in META["levels"] or domain not in META["domains"]:
        return jsonify(error="Unknown level or domain"), 400
    p = float(MODEL.predict(build_row(level, years, domain, skills))[0])
    lo, hi = p * np.exp(META["interval_log_q10"]), p * np.exp(META["interval_log_q90"])
    gains = skill_gains(level, years, domain, skills)
    return jsonify(prediction=round(p, -2), low=round(lo, -2), high=round(hi, -2), suggestions=gains)

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", "5000")))
