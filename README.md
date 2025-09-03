# Drone_Flight_Anomaly_Detection


# Drone Flight Anomaly Dashboard - Defense Analytics

🚁 A comprehensive Streamlit dashboard to detect drone flight anomalies, visualize telemetry data in 3D, and provide actionable insights. Ideal for defense analytics teams.

## Features

- Upload single or multiple drone telemetry CSV files.
- Anomaly detection using:
  - Isolation Forest
  - Bi-LSTM Autoencoder
- Parameter-specific filters (Altitude, Battery, GPS drift)
- Risk level scoring (Normal, Medium, High)
- Rolling mean charts of telemetry parameters with anomaly highlights
- 3D flight path visualization with interactive rotation, zoom, and pan
- Parameter analysis with recommended actions
- Download anomaly reports in CSV format
- Sidebar filter & configuration panel
- Telemetry parameter guide for user clarity

## Telemetry Parameters

| Parameter | Description |
|-----------|-------------|
| timestamp | Time of each flight record |
| altitude | Height above ground (meters). Spikes/drops indicate issues |
| velocity | Drone speed (m/s). Sudden increases indicate control/environment issues |
| yaw | Rotation around vertical axis (degrees). Deviations indicate navigation errors |
| pitch | Rotation around lateral axis (degrees). Indicates tilt/stability anomalies |
| battery | Remaining battery (%). Low values can lead to mission failure |
| gps_drift | GPS deviation (meters). High values indicate interference or signal loss |
| latitude & longitude | Drone position in geo-coordinates (optional for 3D path visualization) |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drone-anomaly-dashboard.git
cd drone-anomaly-dashboard
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate       # Windows
```
3. Install required packages:
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Upload drone telemetry CSV(s).
2. Select the anomaly detection model (Isolation Forest or Bi-LSTM Autoencoder).
3. Adjust filters and thresholds in the sidebar as needed.
4. View metrics, charts, 3D flight path, and parameter analysis.
5. Download anomaly reports if detected.

## Notes

- Supports both small and large datasets with automatic handling.
- Bi-LSTM Autoencoder training uses default 5 epochs (adjustable in code).
- Ensure CSV files have required columns: `timestamp, altitude, velocity, yaw, pitch, battery, gps_drift`.
- For 3D visualization, `latitude` and `longitude` columns are required.

## License

MIT License

