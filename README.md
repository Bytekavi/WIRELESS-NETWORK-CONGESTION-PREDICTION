# Wireless Network Congestion Prediction Using Machine Learning Algorithms

> **Academic Machine Learning Project Prototype**  
> A real-time, dynamic web application backed by a persistent **SQLite relational database** and supervised machine learning classification algorithms (**Random Forest**, **Decision Tree**, **Gradient Boosting**).

---

## 1. Project Overview & Dynamic Architecture

Wireless cellular networks (4G LTE, 5G NR, Wi-Fi 6) experience congestion due to traffic bursts, high subscriber density, and backhaul limits. This system provides an end-to-end, dynamic software platform that:
1. Ingests network telemetry measurements (Network Load, Throughput, Latency, and Packet Loss).
2. Classifies network state into **`Low`**, **`Medium`**, or **`High`** congestion using a trained machine learning model.
3. Automatically persists every measurement and inference into a **real SQLite database** (`data/wireless_network.db`).
4. Streams real-time telemetry from registered base stations into interactive **Chart.js dynamic visual charts**.
5. Manages base station nodes, logs incident alerts for critical congestion, and exports telemetry logs to **CSV**.

```
                           ┌────────────────────────────────────────────────────────┐
                           │               Real SQLite Database                     │
                           │           (data/wireless_network.db)                   │
                           │  - base_stations (Registered Cell Towers)              │
                           │  - telemetry_logs (Chronological Measurements)         │
                           │  - alerts (Critical / Warning Incidents)               │
                           └──────────────────────────┬─────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [Live RESTful APIs / Ingestion]                                [Dynamic Web Dashboard]
         - POST /api/predict                                            - / (Live Charts, Sliders & Feed)
         - POST /api/telemetry/simulate-step                            - /history (Database Log Explorer)
         - GET  /api/live-data                                          - /stations (Base Station Directory)
         - GET  /api/export-csv                                         - Real-Time Chart.js Visualizations
                       ▲                                                             ▲
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                    [Trained ML Classifier Pipeline]
                                  Random Forest (99.83% Test Accuracy)
```

---

## 2. Real Database Schema (`data/wireless_network.db`)

The database is built on **SQLite 3** with foreign key constraints and query indexes:

### 1. `base_stations` Table
Stores cell tower radio specifications:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `cell_id` (TEXT UNIQUE) &mdash; e.g. `BS-101-North`, `BS-102-Central`, `BS-105-Stadium`
- `name` (TEXT) &mdash; Facility or landmark name
- `location` (TEXT) &mdash; Geographic district
- `frequency_band` (TEXT) &mdash; e.g. `3.5 GHz (n78)`, `28 GHz (mmWave)`, `700 MHz`
- `technology` (TEXT) &mdash; `5G NR`, `4G LTE`, `Wi-Fi 6`
- `status` (TEXT) &mdash; `Active`, `Maintenance`, `Degraded`
- `created_at` (TIMESTAMP)

### 2. `telemetry_logs` Table
Stores every live telemetry packet and model prediction:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `cell_id` (TEXT, FOREIGN KEY)
- `timestamp` (DATETIME DEFAULT CURRENT_TIMESTAMP)
- `network_load_pct` (REAL)
- `throughput_mbps` (REAL)
- `latency_ms` (REAL)
- `packet_loss_pct` (REAL)
- `predicted_congestion` (TEXT: `Low`, `Medium`, `High`)
- `confidence_pct` (REAL)
- `model_used` (TEXT)
- `prob_low`, `prob_medium`, `prob_high` (REAL)
- `is_simulation` (INTEGER: 0 for manual, 1 for live stream)

### 3. `alerts` Table
Generated automatically whenever `Medium` or `High` congestion is detected:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `telemetry_id` (INTEGER, FOREIGN KEY)
- `cell_id` (TEXT)
- `severity` (`Warning`, `Critical`)
- `message` (TEXT)
- `is_acknowledged` (BOOLEAN DEFAULT 0)
- `created_at` (TIMESTAMP)

---

## 3. Dynamic Website Features

### 1. Live Dashboard (`/`)
- **Real-Time Dynamic Charts**:
  - **Latency & Throughput Timeseries**: Live dual-axis line chart updating dynamically from SQLite.
  - **Congestion Breakdown Doughnut Chart**: Proportional distribution of Low, Medium, and High states.
- **KPI Counter Cards**: Real-time total database records, active alerts, average network load, and average latency.
- **Live Stream Simulator**:
  - `▶ Start Real-Time Stream`: Automatically generates telemetry packets from random cells every 2.6 seconds, runs ML inference, and writes to SQLite without page reloads!
  - `⚡ Inject 1 Packet`: Single-step simulation burst.
- **Incident Alert Center**: Unacknowledged critical congestion alerts with one-click acknowledgment.
- **Recent Telemetry Table**: Real-time live feed showing latest SQLite rows with per-row deletion.

### 2. Telemetry Log Explorer (`/history`)
- Comprehensive tabular browser of all database records.
- Filtering by **Base Station** and **Congestion Severity**.
- Direct **CSV Export** (`/api/export-csv`) and full database purge controls.

### 3. Base Station Registry (`/stations`)
- View all active base station towers and their radio characteristics.
- Add and register new custom cell towers into SQLite.

---

## 4. Machine Learning Benchmark Results

Evaluated on a held-out stratified test set (600 test samples, 20% holdout):

| Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 99.33% | 0.9937 | 0.9934 | 0.9935 | Baseline |
| **Random Forest** | **99.83%** | **0.9984** | **0.9984** | **0.9984** | **Selected Best** |
| **Gradient Boosting** | 99.50% | 0.9952 | 0.9952 | 0.9952 | Comparison |

### Confusion Matrix (Random Forest Test Set):
```
Actual \ Predicted     Low       Medium     High
Low (210 samples)      210          0         0     (Recall: 100.0%)
Medium (210 samples)     1        209         0     (Recall: 99.5%)
High (180 samples)       0          0       180     (Recall: 100.0%)
```

---

## 5. Project Directory Structure

```
wmc project/
│
├── app.py                      # Dynamic Flask web app & REST API
├── database.py                 # SQLite database schema, connections & queries
├── train_model.py              # ML data generator, training & evaluation
├── test_system.py              # 12 automated unit and integration tests
├── requirements.txt            # Python dependencies
├── model_comparison.csv        # Benchmark metrics
├── README.md                   # Full documentation
│
├── data/
│   ├── network_congestion.csv  # Synthetic dataset (3,000 records)
│   └── wireless_network.db     # SQLite persistent database
│
├── models/
│   ├── congestion_model.joblib # Serialized best model pipeline
│   └── model_metadata.json     # Detailed evaluation metrics
│
└── templates/
    ├── index.html              # Live dynamic dashboard with Chart.js
    ├── history.html            # Database log explorer & filters
    └── stations.html           # Base station node registry
```

---

## 6. How to Run the Project

### Step 1: Open Terminal in Project Directory
```powershell
cd "d:\projects\wmc project"
```

### Step 2: Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### Step 3: Train Models (Optional, already trained)
```powershell
python train_model.py
```

### Step 4: Run Automated Tests
```powershell
python test_system.py
```

### Step 5: Start the Dynamic Web Application
```powershell
python app.py
```
Open **`http://127.0.0.1:5000`** in any web browser.

---

## 7. Interactive Features to Test in Browser

1. **Test One-Click Scenarios**: Click `🟢 Low (Off-Peak)`, `🟡 Medium (Office)`, or `🔴 High (Rush Hour)` on the dashboard and click **Run Prediction & Save to DB**.
2. **Start Real-Time Telemetry Stream**: Click `▶ Start Real-Time Stream` to watch the site stream simulated wireless packets into the SQLite database every 2.6 seconds while the dynamic charts and table update live.
3. **Inspect Real Database Logs**: Visit `/history` to filter logs by cell tower and severity.
4. **Export to CSV**: Click `📥 Export CSV` to download the entire database as a spreadsheet.
5. **Add a Cell Tower**: Visit `/stations` to register a new 5G/4G base station.
