"""
CRISPR Guardian (Simulation) — Backend API
============================================
Flask API server that powers the index.html frontend.

IMPORTANT — READ BEFORE USING:
This server does NOT run a trained machine learning model. Every "risk score,"
"confidence" value, and "hotspot" returned by these endpoints comes from a
deterministic placeholder formula seeded from the input sequence's character
codes. There is no Random Forest, XGBoost, or Transformer anywhere in this
codebase. This exists to demonstrate API/frontend wiring for a teaching demo —
it must never be used to make real decisions about gene editing, therapeutics,
or biosafety. Real off-target risk assessment requires wet-lab verification
(e.g. GUIDE-seq, mass spectrometry) and validated, peer-reviewed tools.
"""

import math
import os
import random
import webbrowser
from datetime import datetime, timedelta
from threading import Timer

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow index.html (served from file:// or any localhost port) to call this API

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["API_KEY"] = os.getenv("API_KEY", "")

# ----------------------------------------------------------------------------
# Engine constants — mirrors the weights used in index.html's JS port
# ----------------------------------------------------------------------------
WEIGHT_RF = 0.35
WEIGHT_XGB = 0.25
WEIGHT_TRANS = 0.40

VALID_GRNA_CHARS = set("ATGCU")
VALID_TARGET_CHARS = set("ATGC")


# ----------------------------------------------------------------------------
# Core simulated-formula engine
# ----------------------------------------------------------------------------
class SimulatedScoringEngine:
    """Deterministic placeholder formula. NOT a trained ML model."""

    @staticmethod
    def _sum_ord(s: str) -> int:
        return sum(ord(c) for c in s)

    @staticmethod
    def compute_similarity(seq1: str, seq2: str) -> float:
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / min_len

    @classmethod
    def generate_attention_scores(cls, sequence: str):
        """Random matrix with a boosted diagonal. Not learned attention."""
        length = len(sequence)
        seed = cls._sum_ord(sequence) % 1000
        rng = random.Random(seed)

        matrix = [[rng.random() * 0.2 for _ in range(length)] for _ in range(length)]

        for i in range(length):
            matrix[i][i] = 0.6 + rng.random() * 0.3
            if length - 4 <= i < length:
                for j in range(length):
                    matrix[i][j] += 0.25

        for i in range(length):
            row_sum = sum(matrix[i])
            if row_sum > 0:
                matrix[i] = [v / row_sum for v in matrix[i]]

        return matrix

    @staticmethod
    def check_pam_site(target: str) -> dict:
        """
        Real biological rule (not a formula): SpCas9 requires an NGG PAM
        immediately 3' of the target site. This checks the last 3 bases.
        """
        target = target.upper().strip()
        if len(target) < 3:
            return {"valid": False, "pam": "", "detail": "Target too short to contain a PAM."}
        pam = target[-3:]
        valid = len(pam) == 3 and pam[1] == "G" and pam[2] == "G"
        detail = (
            f"PAM '{pam}' matches the NGG pattern SpCas9 requires."
            if valid
            else f"PAM '{pam}' does NOT match NGG — SpCas9 cannot bind here regardless of sequence match."
        )
        return {"valid": valid, "pam": pam, "detail": detail}

    @staticmethod
    def positional_mismatch_profile(grna: str, target: str) -> dict:
        """
        Per-base mismatch map with seed-region weighting. The 3' end (closest
        to the PAM) is the seed region and is biologically most sensitive to
        mismatches.
        """
        min_len = min(len(grna), len(target))
        profile = []
        weighted_penalty = 0.0
        max_possible = 0.0
        for i in range(min_len):
            is_match = grna[i] == target[i]
            weight = 1.0 + (i / min_len) * 1.5  # ramps up toward the 3'/PAM end
            max_possible += weight
            if not is_match:
                weighted_penalty += weight
            profile.append({
                "pos": i,
                "grna_base": grna[i],
                "target_base": target[i],
                "match": is_match,
                "weight": round(weight, 2),
            })
        seed_weighted_score = (weighted_penalty / max_possible) if max_possible else 0.0
        return {"profile": profile, "seed_weighted_penalty": round(seed_weighted_score * 100, 2)}

    @classmethod
    def predict_off_target_risk(cls, grna_raw: str, target_raw: str) -> dict:
        grna = grna_raw.upper().strip()
        target = target_raw.upper().strip()

        seed_factor = (cls._sum_ord(grna) + cls._sum_ord(target)) % 50000
        rng = random.Random(seed_factor)

        similarity = cls.compute_similarity(grna, target)
        rf_risk = min(1.0, max(0.0, similarity * 0.85 + rng.gauss(0, 1) * 0.04))
        xgb_risk = min(1.0, max(0.0, similarity * 0.82 + rng.gauss(0, 1) * 0.05))
        trans_risk = min(1.0, max(0.0, similarity * 0.91 + rng.gauss(0, 1) * 0.03))

        cmp_len = min(10, len(grna), len(target))
        seed_mismatches = sum(1 for i in range(cmp_len) if grna[i] != target[i])

        final_risk = (
            rf_risk * WEIGHT_RF + xgb_risk * WEIGHT_XGB + trans_risk * WEIGHT_TRANS
        )
        if seed_mismatches == 0 and similarity > 0.75:
            final_risk = min(1.0, final_risk * 1.25)

        pam_check = cls.check_pam_site(target)
        if not pam_check["valid"]:
            final_risk = min(final_risk, 0.10)

        mismatch_data = cls.positional_mismatch_profile(grna, target)

        confidence = min(0.99, max(0.70, 0.88 + similarity * 0.10 - seed_mismatches * 0.02))

        if final_risk < 0.35:
            classification, color = "Low Risk", "#32C766"
            reasoning = (
                "Formula output: large mismatch distance relative to the simulated "
                "baseline. Low simulated probability of secondary site cleavage "
                "(not a real prediction)."
            )
        elif final_risk < 0.70:
            classification, color = "Medium Risk", "#FFB300"
            reasoning = (
                "Formula output: moderate sequence match density. The simulated "
                "formula flags this as worth a closer (real, wet-lab) look — this "
                "is not itself evidence of risk."
            )
        else:
            classification, color = "High Risk", "#FF3366"
            reasoning = (
                "Formula output: high sequence alignment by the simulated "
                "similarity metric. This is a placeholder signal only — it does "
                "not indicate an actual cleavage risk."
            )

        if not pam_check["valid"]:
            reasoning = (
                f"{pam_check['detail']} Risk is capped low because, biologically, "
                "no binding can occur without a valid PAM, regardless of what the "
                "formula's sequence-similarity score says."
            )

        return {
            "risk_score": final_risk * 100,
            "confidence": confidence * 100,
            "classification": classification,
            "color": color,
            "sites_count": int(final_risk * 18 + math.floor(rng.random() * 4)),
            "reasoning": reasoning,
            "model_breakdown": {
                "Formula Variant A (labeled 'Random Forest')": rf_risk * 100,
                "Formula Variant B (labeled 'XGBoost')": xgb_risk * 100,
                "Formula Variant C (labeled 'Transformer')": trans_risk * 100,
            },
            "pam_check": pam_check,
            "mismatch_profile": mismatch_data["profile"],
            "seed_weighted_penalty": mismatch_data["seed_weighted_penalty"],
        }

    @classmethod
    def recommend_safer_targets(cls, candidates: list, reference_grna: str) -> list:
        rows = []
        for cand in candidates:
            m = cls.predict_off_target_risk(reference_grna, cand)
            rows.append(
                {
                    "candidate": cand,
                    "risk": round(m["risk_score"], 2),
                    "confidence": round(m["confidence"], 2),
                    "hotspots": m["sites_count"],
                    "classification": m["classification"],
                    "pam_valid": m["pam_check"]["valid"],
                    "pam": m["pam_check"]["pam"],
                    "seed_weighted_penalty": m["seed_weighted_penalty"],
                }
            )
        rows.sort(key=lambda r: (not r["pam_valid"], r["risk"]))
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows


# ----------------------------------------------------------------------------
# Synthetic historical analytics
# ----------------------------------------------------------------------------
def generate_synthetic_historical_analytics():
    rng = random.Random(42)
    n = 100
    start = datetime(2026, 1, 1)
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]

    daily = [rng.randint(15, 45) for _ in range(n)]
    total = []
    running = 0
    for v in daily:
        running += v
        total.append(running)

    avg_risk = []
    cum = 0.0
    for _ in range(n):
        cum += rng.gauss(0, 1) * 0.4
        avg_risk.append(min(85.0, max(15.0, 65.0 - cum)))

    high_risk = [rng.randint(1, 8) for _ in range(n)]
    safe_validated = [rng.randint(5, 25) for _ in range(n)]

    return {
        "dates": dates,
        "total": total,
        "avg_risk": avg_risk,
        "high_risk": high_risk,
        "safe_validated": safe_validated,
    }


HISTORICAL_DATA = generate_synthetic_historical_analytics()


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------
def validate_sequence(seq: str, allowed_chars: set, field_name: str):
    if not seq or not isinstance(seq, str):
        return f"{field_name} is required and must be a non-empty string."
    cleaned = seq.upper().strip()
    bad_chars = set(cleaned) - allowed_chars
    if bad_chars:
        return (
            f"{field_name} contains invalid characters: {', '.join(sorted(bad_chars))}. "
            f"Allowed: {', '.join(sorted(allowed_chars))}."
        )
    return None


# ----------------------------------------------------------------------------
# Routes (All 7 Endpoints + Web Serving Interface Engine)
# ----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serves the frontend index.html dashboard template."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Liveness health check route query."""
    return jsonify({"status": "ok", "service": "crispr-guardian-simulation-api"})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Single gRNA/target evaluation formula metric scores."""
    data = request.get_json(silent=True) or {}
    grna = data.get("grna", "")
    target = data.get("target", "")

    err = validate_sequence(grna, VALID_GRNA_CHARS, "grna")
    if err:
        return jsonify({"error": err}), 400
    err = validate_sequence(target, VALID_TARGET_CHARS, "target")
    if err:
        return jsonify({"error": err}), 400

    result = SimulatedScoringEngine.predict_off_target_risk(grna, target)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def compare():
    """Multi-Sequence Candidate Comparator screening ranking tool."""
    data = request.get_json(silent=True) or {}
    grna = data.get("grna", "")
    candidates = data.get("candidates", [])

    err = validate_sequence(grna, VALID_GRNA_CHARS, "grna")
    if err:
        return jsonify({"error": err}), 400
        
    if not isinstance(candidates, list) or len(candidates) == 0:
        return jsonify({"error": "candidates must be a non-empty list of strings."}), 400

    cleaned_candidates = []
    for i, cand in enumerate(candidates):
        if not cand.strip():
            continue
        err = validate_sequence(cand, VALID_TARGET_CHARS, f"candidates[{i}]")
        if err:
            return jsonify({"error": err}), 400
        cleaned_candidates.append(cand.upper().strip())

    if not cleaned_candidates:
         return jsonify({"error": "No valid target candidates found."}), 400

    rows = SimulatedScoringEngine.recommend_safer_targets(cleaned_candidates, grna)
    return jsonify({"results": rows})


@app.route("/api/attention-matrix", methods=["POST"])
def attention_matrix():
    """Generates a random token heatmap weight distribution structure."""
    data = request.get_json(silent=True) or {}
    sequence = data.get("sequence", "")
    
    err = validate_sequence(sequence, VALID_GRNA_CHARS.union(VALID_TARGET_CHARS), "sequence")
    if err:
        return jsonify({"error": err}), 400
        
    matrix = SimulatedScoringEngine.generate_attention_scores(sequence)
    return jsonify({"matrix": matrix})


@app.route("/api/genome-track", methods=["GET"])
def genome_track():
    """Returns a synthetic per-chromosome coordinate risk density map curve."""
    chr_label = request.args.get("chr", "Chr 1")
    seed = ord(chr_label[-1]) if chr_label else 1
    rng = random.Random(seed)

    n = 100
    coords = [round((i / (n - 1)) * 150, 3) for i in range(n)]
    risk_density = [
        min(100.0, max(0.0, abs(math.sin(c * 0.15) * 40) + rng.random() * 45))
        for c in coords
    ]
    
    # Fixed demo hotspots
    risk_density[20] = 92.4
    risk_density[55] = 88.1
    risk_density[82] = 95.3

    return jsonify(
        {
            "chromosome": chr_label,
            "coordinates_mb": coords,
            "synthetic_risk_density": risk_density,
            "fixed_hotspot_indices": [20, 55, 82],
            "note": "Hotspot positions are hardcoded demo markers, not detected signals.",
        }
    )


@app.route("/api/historical-analytics", methods=["GET"])
def historical_analytics():
    """Returns programmatic rolling data trends charts matrix arrays."""
    return jsonify(HISTORICAL_DATA)


@app.route("/api/architecture", methods=["GET"])
def architecture():
    """Returns fixed static blueprint weight variables data logs configurations."""
    return jsonify(
        {
            "convergence_scores": {
                "Formula Variant A (labeled 'Random Forest')": 87.1,
                "Formula Variant B (labeled 'XGBoost')": 89.6,
                "Formula Variant C (labeled 'Transformer')": 94.2,
            },
            "note": "These numbers are fixed placeholders for the diagram, not measured results.",
        }
    )


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5003/")


if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=5003, debug=True)
