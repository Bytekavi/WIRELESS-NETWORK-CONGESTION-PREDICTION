"""
Automated System & Database Functional Test Suite
Wireless Network Congestion Prediction
Tests all functional cases: ML pipelines, SQLite database operations, real-time APIs, and web routes.
"""

import os
import unittest
import json
import database
from app import app, load_trained_model, validate_telemetry_inputs, fallback_predict


class TestWirelessCongestionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()
        load_trained_model()
        database.init_db()

    def test_01_training_and_db_artifacts_exist(self):
        """Verify training and database artifacts exist."""
        self.assertTrue(os.path.exists("data/network_congestion.csv"), "Dataset CSV must exist.")
        self.assertTrue(os.path.exists("models/congestion_model.joblib"), "Model file must exist.")
        self.assertTrue(os.path.exists("model_comparison.csv"), "Model comparison CSV must exist.")
        self.assertTrue(os.path.exists("models/model_metadata.json"), "Metadata JSON must exist.")
        self.assertTrue(os.path.exists("data/wireless_network.db"), "SQLite database file must exist.")

    def test_02_database_base_stations_seeded(self):
        """Verify initial base stations are seeded into SQLite."""
        stations = database.get_all_base_stations()
        self.assertGreaterEqual(len(stations), 5)
        cell_ids = [s["cell_id"] for s in stations]
        self.assertIn("BS-101-North", cell_ids)
        self.assertIn("BS-102-Central", cell_ids)

    def test_03_valid_predict_and_db_persistence(self):
        """Test prediction via API and verify it is written to the SQLite database."""
        initial_stats = database.get_telemetry_statistics()
        initial_count = initial_stats["total_records"]

        payload = {
            "cell_id": "BS-101-North",
            "network_load_pct": 20.0,
            "throughput_mbps": 85.0,
            "latency_ms": 22.0,
            "packet_loss_pct": 0.2
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "Low")
        self.assertIn("database_record_id", data)
        self.assertGreater(data["database_record_id"], 0)

        # Check DB count incremented
        updated_stats = database.get_telemetry_statistics()
        self.assertEqual(updated_stats["total_records"], initial_count + 1)

    def test_04_critical_congestion_and_alert_generation(self):
        """Test High congestion prediction and check that an alert is created in the database."""
        payload = {
            "cell_id": "BS-105-Stadium",
            "network_load_pct": 94.0,
            "throughput_mbps": 4.5,
            "latency_ms": 220.0,
            "packet_loss_pct": 11.2
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "High")

        # Verify alert was inserted into SQLite
        alerts = database.get_active_alerts(limit=5)
        self.assertGreater(len(alerts), 0)
        latest_alert = alerts[0]
        self.assertEqual(latest_alert["severity"], "Critical")
        self.assertEqual(latest_alert["cell_id"], "BS-105-Stadium")

        # Acknowledge the alert
        ack_res = self.client.post(f"/api/alerts/{latest_alert['id']}/acknowledge")
        self.assertEqual(ack_res.status_code, 200)
        self.assertTrue(ack_res.get_json()["success"])

    def test_05_negative_latency_validation(self):
        """Test validation failure on negative latency."""
        clean, err = validate_telemetry_inputs(50.0, 40.0, -15.0, 1.0)
        self.assertIsNone(clean)
        self.assertIn("Latency cannot be negative", err)

        res = self.client.post("/api/predict", json={
            "cell_id": "BS-101-North",
            "network_load_pct": 50.0,
            "throughput_mbps": 40.0,
            "latency_ms": -15.0,
            "packet_loss_pct": 1.0
        })
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["success"])

    def test_06_load_above_100_validation(self):
        """Test validation failure on load > 100%."""
        clean, err = validate_telemetry_inputs(125.0, 40.0, 30.0, 1.0)
        self.assertIsNone(clean)
        self.assertIn("Network load must be between 0% and 100%", err)

    def test_07_simulate_step_api(self):
        """Test live telemetry burst simulation endpoint."""
        res = self.client.post("/api/telemetry/simulate-step")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("database_record_id", data)
        self.assertIn(data["prediction"], ["Low", "Medium", "High"])

    def test_08_live_data_api(self):
        """Test live dynamic data bundle API for dashboard charts and counters."""
        res = self.client.get("/api/live-data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("timeseries", data)
        self.assertIn("statistics", data)
        self.assertIn("recent_logs", data)

    def test_09_export_csv(self):
        """Test CSV database export endpoint."""
        res = self.client.get("/api/export-csv")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn(b"Predicted_Congestion", res.data)
        self.assertIn(b"Cell_ID", res.data)

    def test_10_routes_render_html(self):
        """Verify all dynamic web pages render with 200 OK."""
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"Wireless Network Congestion Prediction", res_index.data)

        res_history = self.client.get("/history")
        self.assertEqual(res_history.status_code, 200)
        self.assertIn(b"Database Telemetry Log Explorer", res_history.data)

        res_stations = self.client.get("/stations")
        self.assertEqual(res_stations.status_code, 200)
        self.assertIn(b"Registered Base Stations", res_stations.data)

    def test_11_delete_record_api(self):
        """Test deleting a record from SQLite database."""
        # Insert a sample
        rec_id = database.save_telemetry_record(
            cell_id="BS-102-Central",
            network_load_pct=30.0,
            throughput_mbps=70.0,
            latency_ms=25.0,
            packet_loss_pct=0.3,
            predicted_congestion="Low",
            confidence_pct=99.0,
            model_used="Random Forest",
            probs={"Low": 0.99, "Medium": 0.01, "High": 0.0}
        )
        res = self.client.post(f"/api/telemetry/{rec_id}/delete")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

    def test_12_add_base_station(self):
        """Test registering a new base station into SQLite."""
        import time
        unique_id = f"BS-TEST-{int(time.time() * 1000)}"
        success, msg = database.add_base_station(
            cell_id=unique_id,
            name="Test Verification Tower",
            location="Lab Sector 1",
            frequency_band="3.6 GHz",
            technology="5G NR"
        )
        self.assertTrue(success)
        stations = database.get_all_base_stations()
        cell_ids = [s["cell_id"] for s in stations]
        self.assertIn(unique_id, cell_ids)


if __name__ == "__main__":
    unittest.main()
