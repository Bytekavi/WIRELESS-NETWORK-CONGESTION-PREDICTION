"""
Wireless Network Congestion Prediction
Dynamic Full-Stack Web Application with Persistent SQLite Database
Author: Academic Machine Learning Project
"""

import os
import io
import csv
import json
import random
import datetime
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response

import database

app = Flask(__name__)

# Paths
MODELS_DIR = "models"
MODEL_FILE = os.path.join(MODELS_DIR, "congestion_model.joblib")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
COMPARISON_FILE = "model_comparison.csv"

FEATURE_COLUMNS = [
    "network_load_pct",
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct"
]

# Global variables for model
model_pipeline = None
model_metadata = {}
is_ml_model_loaded = False
active_model_name = "Rule-based Fallback"


def load_trained_model():
    """Loads the trained machine learning pipeline and metadata from disk."""
    global model_pipeline, model_metadata, is_ml_model_loaded, active_model_name

    if os.path.exists(MODEL_FILE):
        try:
            payload = joblib.load(MODEL_FILE)
            model_pipeline = payload.get("pipeline")
            active_model_name = payload.get("model_name", "Trained ML Model")
            is_ml_model_loaded = True
            print(f"[+] Loaded ML model '{active_model_name}' successfully from {MODEL_FILE}")
        except Exception as e:
            print(f"[!] Error loading model file: {e}")
            is_ml_model_loaded = False
            model_pipeline = None
            active_model_name = "Rule-based Fallback (Load Error)"
    else:
        print(f"[!] Model file '{MODEL_FILE}' not found. Using transparent rule-based fallback.")
        is_ml_model_loaded = False
        model_pipeline = None
        active_model_name = "Rule-based Fallback (Model Not Trained)"

    # Load metadata JSON if present
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception as e:
            print(f"[!] Error reading metadata JSON: {e}")
            model_metadata = {}


# Initial model load
load_trained_model()


def fallback_predict(load, throughput, latency, loss):
    """
    Transparent rule-based heuristic prediction when ML model file is unavailable.
    As per project specification section 11.3.
    """
    if load >= 75.0 or latency >= 120.0 or loss >= 5.0 or throughput <= 15.0:
        pred_class = "High"
        probs = {"Low": 0.05, "Medium": 0.15, "High": 0.80}
    elif load >= 45.0 or latency >= 50.0 or loss >= 1.5 or throughput <= 45.0:
        pred_class = "Medium"
        probs = {"Low": 0.15, "Medium": 0.70, "High": 0.15}
    else:
        pred_class = "Low"
        probs = {"Low": 0.85, "Medium": 0.12, "High": 0.03}
    return pred_class, probs


def validate_telemetry_inputs(load_raw, tp_raw, lat_raw, loss_raw):
    """
    Validates the four input telemetry measurements according to system specifications.
    Returns (cleaned_dict, error_message)
    """
    try:
        if any(v is None or str(v).strip() == "" for v in [load_raw, tp_raw, lat_raw, loss_raw]):
            return None, "All four telemetry fields are required."

        load = float(load_raw)
        tp = float(tp_raw)
        lat = float(lat_raw)
        loss = float(loss_raw)
    except (ValueError, TypeError):
        return None, "All telemetry inputs must be valid numeric values."

    if load < 0.0 or load > 100.0:
        return None, f"Network load must be between 0% and 100%. (Provided: {load}%)"

    if tp < 0.0:
        return None, f"Throughput cannot be negative. (Provided: {tp} Mbps)"

    if lat < 0.0:
        return None, f"Latency cannot be negative. (Provided: {lat} ms)"

    if loss < 0.0 or loss > 100.0:
        return None, f"Packet loss rate must be between 0% and 100%. (Provided: {loss}%)"

    return {
        "network_load_pct": round(load, 2),
        "throughput_mbps": round(tp, 2),
        "latency_ms": round(lat, 2),
        "packet_loss_pct": round(loss, 2)
    }, None


def get_recommendation_and_style(congestion_class):
    """Returns visual styling, descriptions, and network engineering recommendations."""
    if congestion_class == "Low":
        return {
            "badge_class": "badge-low",
            "color_hex": "#10b981",
            "summary": "Network operating within optimal capacity parameters.",
            "action": "Normal state. Standard quality of service (QoS) scheduling is sufficient. No active congestion mitigation required."
        }
    elif congestion_class == "Medium":
        return {
            "badge_class": "badge-medium",
            "color_hex": "#f59e0b",
            "summary": "Moderate network strain detected. Buffers are filling.",
            "action": "Caution state. Activate Active Queue Management (AQM), monitor jitter, and prepare load-balancing or traffic prioritization."
        }
    else:  # High
        return {
            "badge_class": "badge-high",
            "color_hex": "#ef4444",
            "summary": "Severe congestion detected! Potential packet drops and bufferbloat.",
            "action": "Critical state. Enforce aggressive traffic shaping, throttle background flows, trigger carrier load-balancing / radio cell handover, and alert network operations."
        }


def run_prediction(clean_inputs):
    """Executes ML prediction or fallback heuristic."""
    if model_pipeline is None and os.path.exists(MODEL_FILE):
        load_trained_model()

    if is_ml_model_loaded and model_pipeline is not None:
        df_input = pd.DataFrame([clean_inputs], columns=FEATURE_COLUMNS)
        pred_class = model_pipeline.predict(df_input)[0]

        probs = {}
        if hasattr(model_pipeline, "predict_proba"):
            raw_probs = model_pipeline.predict_proba(df_input)[0]
            classes = model_pipeline.classes_
            probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, raw_probs)}
        else:
            probs = {pred_class: 1.0}

        source = f"Machine Learning Model ({active_model_name})"
        is_fallback = False
    else:
        pred_class, probs = fallback_predict(
            clean_inputs["network_load_pct"],
            clean_inputs["throughput_mbps"],
            clean_inputs["latency_ms"],
            clean_inputs["packet_loss_pct"]
        )
        source = "Rule-based Fallback (ML Model File Not Found)"
        is_fallback = True

    confidence = round(probs.get(pred_class, 1.0) * 100, 1)
    rec_info = get_recommendation_and_style(pred_class)

    return {
        "prediction": pred_class,
        "probabilities": probs,
        "confidence": confidence,
        "source": source,
        "is_fallback": is_fallback,
        "recommendation": rec_info
    }


# ==============================================================================
# Web Routes
# ==============================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    """Main dynamic dashboard route displaying live stats, charts, inputs, and database records."""
    if not is_ml_model_loaded and os.path.exists(MODEL_FILE):
        load_trained_model()

    error = None
    prediction_result = None
    form_values = {
        "cell_id": "BS-101-North",
        "network_load_pct": 35.0,
        "throughput_mbps": 65.0,
        "latency_ms": 28.0,
        "packet_loss_pct": 0.4
    }

    if request.method == "POST":
        cell_id = request.form.get("cell_id", "BS-101-North").strip()
        raw_load = request.form.get("network_load_pct")
        raw_tp = request.form.get("throughput_mbps")
        raw_lat = request.form.get("latency_ms")
        raw_loss = request.form.get("packet_loss_pct")

        clean_inputs, err = validate_telemetry_inputs(raw_load, raw_tp, raw_lat, raw_loss)
        if err:
            error = err
            form_values = {
                "cell_id": cell_id,
                "network_load_pct": raw_load or "",
                "throughput_mbps": raw_tp or "",
                "latency_ms": raw_lat or "",
                "packet_loss_pct": raw_loss or ""
            }
        else:
            form_values = clean_inputs
            form_values["cell_id"] = cell_id
            prediction_result = run_prediction(clean_inputs)

            # Persist to real SQLite database
            rec_id = database.save_telemetry_record(
                cell_id=cell_id,
                network_load_pct=clean_inputs["network_load_pct"],
                throughput_mbps=clean_inputs["throughput_mbps"],
                latency_ms=clean_inputs["latency_ms"],
                packet_loss_pct=clean_inputs["packet_loss_pct"],
                predicted_congestion=prediction_result["prediction"],
                confidence_pct=prediction_result["confidence"],
                model_used=active_model_name if not prediction_result["is_fallback"] else "Rule Fallback",
                probs=prediction_result["probabilities"],
                is_simulation=False
            )
            prediction_result["database_record_id"] = rec_id

    # Retrieve real-time dynamic data from SQLite
    db_stats = database.get_telemetry_statistics()
    recent_logs = database.get_recent_telemetry(limit=10)
    active_alerts = database.get_active_alerts(limit=5)
    base_stations = database.get_all_base_stations()

    # Read comparison table if available
    comparison_data = []
    if os.path.exists(COMPARISON_FILE):
        try:
            comp_df = pd.read_csv(COMPARISON_FILE)
            comparison_data = comp_df.to_dict(orient="records")
        except Exception:
            comparison_data = []

    return render_template(
        "index.html",
        is_ml_model_loaded=is_ml_model_loaded,
        active_model_name=active_model_name,
        metadata=model_metadata,
        comparison_data=comparison_data,
        form_values=form_values,
        prediction_result=prediction_result,
        error=error,
        db_stats=db_stats,
        recent_logs=recent_logs,
        active_alerts=active_alerts,
        base_stations=base_stations
    )


@app.route("/history", methods=["GET"])
def history():
    """Full database telemetry history viewer with search filters."""
    cell_id = request.args.get("cell_id", "").strip() or None
    congestion = request.args.get("congestion", "").strip() or None
    limit = int(request.args.get("limit", 100))

    logs = database.get_recent_telemetry(limit=limit, cell_id=cell_id, congestion=congestion)
    base_stations = database.get_all_base_stations()
    db_stats = database.get_telemetry_statistics()

    return render_template(
        "history.html",
        logs=logs,
        base_stations=base_stations,
        selected_cell=cell_id,
        selected_congestion=congestion,
        db_stats=db_stats,
        total_shown=len(logs)
    )


@app.route("/stations", methods=["GET", "POST"])
def stations_manager():
    """Base station node management with database insertion."""
    message = None
    msg_type = "info"

    if request.method == "POST":
        cell_id = request.form.get("cell_id", "").strip()
        name = request.form.get("name", "").strip()
        location = request.form.get("location", "").strip()
        frequency_band = request.form.get("frequency_band", "").strip()
        technology = request.form.get("technology", "5G NR").strip()

        if not all([cell_id, name, location, frequency_band, technology]):
            message = "All base station fields are required."
            msg_type = "danger"
        else:
            success, msg = database.add_base_station(cell_id, name, location, frequency_band, technology)
            if success:
                message = msg
                msg_type = "success"
            else:
                message = msg
                msg_type = "danger"

    stations = database.get_all_base_stations()
    return render_template(
        "stations.html",
        stations=stations,
        message=message,
        msg_type=msg_type
    )


# ==============================================================================
# RESTful API Endpoints for Dynamic AJAX UI
# ==============================================================================

@app.route("/api/predict", methods=["POST", "GET"])
def api_predict():
    """
    RESTful prediction endpoint. Evaluates inputs and saves to the SQLite database.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
    else:
        data = request.args

    if not data:
        return jsonify({"success": False, "error": "No input telemetry provided."}), 400

    cell_id = data.get("cell_id", "BS-101-North").strip()
    clean_inputs, err = validate_telemetry_inputs(
        data.get("network_load_pct"),
        data.get("throughput_mbps"),
        data.get("latency_ms"),
        data.get("packet_loss_pct")
    )

    if err:
        return jsonify({"success": False, "error": err}), 400

    result = run_prediction(clean_inputs)

    # Persist record into real SQLite database
    rec_id = database.save_telemetry_record(
        cell_id=cell_id,
        network_load_pct=clean_inputs["network_load_pct"],
        throughput_mbps=clean_inputs["throughput_mbps"],
        latency_ms=clean_inputs["latency_ms"],
        packet_loss_pct=clean_inputs["packet_loss_pct"],
        predicted_congestion=result["prediction"],
        confidence_pct=result["confidence"],
        model_used=active_model_name if not result["is_fallback"] else "Rule Fallback",
        probs=result["probabilities"],
        is_simulation=bool(data.get("is_simulation", False))
    )

    return jsonify({
        "success": True,
        "database_record_id": rec_id,
        "cell_id": cell_id,
        "input_telemetry": clean_inputs,
        "prediction": result["prediction"],
        "confidence_pct": result["confidence"],
        "probabilities": result["probabilities"],
        "prediction_source": result["source"],
        "is_fallback": result["is_fallback"],
        "recommendation": result["recommendation"]
    })


@app.route("/api/live-data", methods=["GET"])
def api_live_data():
    """Returns dynamic data bundle for live frontend chart updates and feeds."""
    timeseries = database.get_timeseries_data(limit=15)
    stats = database.get_telemetry_statistics()
    recent = database.get_recent_telemetry(limit=8)
    alerts = database.get_active_alerts(limit=5)

    return jsonify({
        "success": True,
        "timeseries": timeseries,
        "statistics": stats,
        "recent_logs": recent,
        "active_alerts": alerts
    })


@app.route("/api/telemetry/simulate-step", methods=["POST"])
def api_simulate_step():
    """
    Simulates a live telemetry event from a random base station node,
    processes it with the ML classifier, and persists to SQLite.
    """
    stations = database.get_all_base_stations()
    station = random.choice(stations) if stations else {"cell_id": "BS-101-North"}
    cell_id = station["cell_id"]

    # Choose scenario profile randomly: 45% Low, 35% Medium, 20% High
    profile = random.choices(["Low", "Medium", "High"], weights=[0.45, 0.35, 0.20])[0]

    if profile == "Low":
        load = round(random.uniform(12.0, 42.0), 1)
        tp = round(random.uniform(50.0, 115.0), 1)
        lat = round(random.uniform(14.0, 38.0), 1)
        loss = round(random.uniform(0.05, 0.9), 2)
    elif profile == "Medium":
        load = round(random.uniform(45.0, 72.0), 1)
        tp = round(random.uniform(22.0, 52.0), 1)
        lat = round(random.uniform(48.0, 88.0), 1)
        loss = round(random.uniform(1.2, 3.8), 2)
    else:  # High
        load = round(random.uniform(76.0, 98.0), 1)
        tp = round(random.uniform(3.0, 18.0), 1)
        lat = round(random.uniform(110.0, 260.0), 1)
        loss = round(random.uniform(4.5, 14.5), 2)

    clean_inputs = {
        "network_load_pct": load,
        "throughput_mbps": tp,
        "latency_ms": lat,
        "packet_loss_pct": loss
    }

    result = run_prediction(clean_inputs)

    rec_id = database.save_telemetry_record(
        cell_id=cell_id,
        network_load_pct=load,
        throughput_mbps=tp,
        latency_ms=lat,
        packet_loss_pct=loss,
        predicted_congestion=result["prediction"],
        confidence_pct=result["confidence"],
        model_used=active_model_name if not result["is_fallback"] else "Rule Fallback",
        probs=result["probabilities"],
        is_simulation=True
    )

    return jsonify({
        "success": True,
        "database_record_id": rec_id,
        "cell_id": cell_id,
        "telemetry": clean_inputs,
        "prediction": result["prediction"],
        "confidence_pct": result["confidence"],
        "probabilities": result["probabilities"]
    })


@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def api_acknowledge_alert(alert_id):
    """Marks an alert as acknowledged in the SQLite database."""
    success = database.acknowledge_alert(alert_id)
    return jsonify({"success": success})


@app.route("/api/telemetry/<int:record_id>/delete", methods=["POST", "DELETE"])
def api_delete_telemetry(record_id):
    """Deletes a telemetry record from the database."""
    success = database.delete_telemetry_record(record_id)
    return jsonify({"success": success})


@app.route("/api/telemetry/clear-all", methods=["POST"])
def api_clear_all():
    """Clears all telemetry and alerts from the database."""
    count = database.clear_all_telemetry()
    return jsonify({"success": True, "deleted_count": count})


@app.route("/api/export-csv", methods=["GET"])
def export_csv():
    """Exports all telemetry records from SQLite database as a downloadable CSV."""
    logs = database.get_recent_telemetry(limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID", "Timestamp", "Cell_ID", "Cell_Name", "Technology",
        "Network_Load_Pct", "Throughput_Mbps", "Latency_Ms", "Packet_Loss_Pct",
        "Predicted_Congestion", "Confidence_Pct", "Model_Used", "Is_Simulation"
    ])

    for log in logs:
        writer.writerow([
            log["id"],
            log["timestamp"],
            log["cell_id"],
            log.get("cell_name", ""),
            log.get("technology", ""),
            log["network_load_pct"],
            log["throughput_mbps"],
            log["latency_ms"],
            log["packet_loss_pct"],
            log["predicted_congestion"],
            log["confidence_pct"],
            log["model_used"],
            log["is_simulation"]
        ])

    csv_data = output.getvalue()
    filename = f"wireless_telemetry_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming database and model statuses."""
    stats = database.get_telemetry_statistics()
    return jsonify({
        "status": "healthy",
        "database": "SQLite (data/wireless_network.db)",
        "is_ml_model_loaded": is_ml_model_loaded,
        "active_model": active_model_name,
        "database_records_count": stats["total_records"]
    })


if __name__ == "__main__":
    print("[*] Starting Wireless Congestion Prediction Web Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
