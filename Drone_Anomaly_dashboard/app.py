import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import random

# -------------------------------
# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -------------------------------
# Page Config
st.set_page_config(page_title="Drone Flight Anomaly Dashboard - Defense Analytics", layout="wide")

# -------------------------------
# Main Header
st.markdown(
    """
    <h1 style='text-align: center; color: #0B3D91;'>🚁 Drone Flight Anomaly Dashboard</h1>
    <p style='text-align: center; font-size:16px;'>Defense Analytics | Upload drone telemetry CSV(s) to detect anomalies and generate actionable insights</p>
    """, unsafe_allow_html=True
)

# -------------------------------
# Telemetry Parameters Guide
st.markdown("### 📘 Telemetry Parameters Guide")
with st.expander("Click to understand drone telemetry parameters"):
    st.markdown("""
    - **timestamp**: Time for each flight record.
    - **altitude**: Height above ground (meters). Spikes/drops indicate issues.
    - **velocity**: Drone speed (m/s). Sudden increases indicate control/environment issues.
    - **yaw**: Rotation around vertical axis (degrees). Deviations indicate navigation errors.
    - **pitch**: Rotation around lateral axis (degrees). Indicates tilt/stability anomalies.
    - **battery**: Remaining battery (%). Low values can lead to mission failure.
    - **gps_drift**: GPS deviation (meters). High values indicate interference or signal loss.
    - **latitude** and **longitude**: Drone position in geo-coordinates (optional for 3D path visualization)
    """)

# -------------------------------
# File Upload
uploaded_files = st.file_uploader(
    "📂 Upload one or more Drone Telemetry CSVs",
    type="csv",
    accept_multiple_files=True,
    key="multi_upload",
    help="Upload telemetry CSVs with timestamp, altitude, velocity, yaw, pitch, battery, gps_drift, optionally latitude and longitude"
)

# -------------------------------
if uploaded_files:
    # Read and combine all CSVs
    df_list = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(df_list, ignore_index=True)

    st.markdown(f"### Combined Data Preview ({len(uploaded_files)} files)")
    st.dataframe(df.head(10), use_container_width=True)

    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

    # -------------------------------
    # Sidebar Controls
    st.sidebar.header("⚙️ Filter & Configuration Panel")
    model_choice = st.sidebar.radio("Select Anomaly Detection Model to Display", ["Isolation Forest", "Bi-LSTM Autoencoder"])
    timesteps = st.sidebar.slider("Timesteps (for LSTM)", 1, 20, 5)  # Increased max timesteps for better sequence learning
    threshold_factor = st.sidebar.slider("Threshold Multiplier (for LSTM)", 1.0, 3.0, 2.0, 0.1)

    st.sidebar.markdown("### Risk Levels")
    risk_normal = st.sidebar.checkbox("Normal", value=True)
    risk_medium = st.sidebar.checkbox("Medium", value=True)
    risk_high = st.sidebar.checkbox("High", value=True)
    selected_risks = [r for r, flag in zip(["Normal","Medium","High"], [risk_normal, risk_medium, risk_high]) if flag]

    st.sidebar.markdown("### Parameter-specific Anomalies")
    filter_altitude = st.sidebar.checkbox("Altitude > Threshold")
    filter_battery = st.sidebar.checkbox("Battery < Threshold")
    filter_gps = st.sidebar.checkbox("High GPS drift")

    with st.sidebar.expander("Advanced Threshold Adjustments"):
        alt_threshold = st.slider("Altitude Threshold", min_value=0, max_value=1000, value=400)
        battery_threshold = st.slider("Battery Threshold", min_value=0, max_value=100, value=50)
        gps_threshold = st.slider("GPS Drift Threshold", min_value=0.0, max_value=5.0, value=0.5)

    # -------------------------------
    # Preprocessing
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    features = df.drop(columns=['timestamp'], errors='ignore')
    numeric_features = features.select_dtypes(include=np.number)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    # Initialize anomaly columns
    df['Anomaly_IF'] = 0
    df['Anomaly_LSTM'] = 0

    # -------------------------------
    # Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=SEED)
    clf.fit(scaled_features)
    df['Anomaly_IF'] = (clf.predict(scaled_features) == -1).astype(int)

    # -------------------------------
    # Bi-LSTM Autoencoder
    # Adjust timesteps to be valid
    timesteps = min(timesteps, len(scaled_features)//2)
    if timesteps < 1:
        timesteps = 1

    # Prepare sequences
    X_seq = []
    for i in range(len(scaled_features)-timesteps):
        X_seq.append(scaled_features[i:i+timesteps])
    X_seq = np.array(X_seq)
    n_features = X_seq.shape[2]

    # Model architecture with dropout for regularization
    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)
    encoded = Dropout(0.2)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    st.info("Training ... ⏳")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_seq, X_seq,
        epochs=50,  # Increased epochs for better training
        batch_size=16,  # Increased batch size for stability
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    st.success("Training Complete ✅")

    # Predict and calculate MSE
    X_pred = model.predict(X_seq)
    mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1,2))

    # Use percentile-based threshold for better anomaly detection
    percentile = 95
    threshold = np.percentile(mse, percentile)
    anomaly_indices = np.where(mse > threshold)[0] + timesteps
    df.loc[anomaly_indices, 'Anomaly_LSTM'] = 1

    # -------------------------------
    # Compute Risk Scores
    def compute_risk(anomaly_col):
        def risk_score(row):
            score = 0
            if row[anomaly_col] == 0:
                return 0
            if row.get('altitude',0) > alt_threshold: score += 3
            if row.get('battery',100) < battery_threshold: score += 4
            if row.get('gps_drift',0) > gps_threshold: score += 2
            if row.get('velocity',0) > 15: score += 2
            if abs(row.get('yaw',0)) > 1 or abs(row.get('pitch',0)) > 1: score += 2
            return score

        df['RiskScore_' + anomaly_col] = df.apply(risk_score, axis=1)
        def risk_level(row):
            if row['RiskScore_' + anomaly_col] >= 6: return 'High'
            elif row['RiskScore_' + anomaly_col] >= 3: return 'Medium'
            else: return 'Normal'
        df['Risk_' + anomaly_col] = df.apply(risk_level, axis=1)

    compute_risk('Anomaly_IF')
    compute_risk('Anomaly_LSTM')

    # -------------------------------
    # Select Model for Display
    anomaly_col = 'Anomaly_IF' if model_choice=="Isolation Forest" else 'Anomaly_LSTM'
    risk_col = 'Risk_' + anomaly_col
    riskscore_col = 'RiskScore_' + anomaly_col

    # -------------------------------
    # Apply Sidebar Filters
    df_filtered = df.copy()
    if selected_risks:
        df_filtered = df_filtered[df_filtered[risk_col].isin(selected_risks)]
    if filter_altitude and 'altitude' in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered['altitude'] > alt_threshold) | (df_filtered[anomaly_col]==1)]
    if filter_battery and 'battery' in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered['battery'] < battery_threshold) | (df_filtered[anomaly_col]==1)]
    if filter_gps and 'gps_drift' in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered['gps_drift'] > gps_threshold) | (df_filtered[anomaly_col]==1)]

    anomaly_df = df_filtered[df_filtered[anomaly_col]==1]

    # -------------------------------
    # Metrics
    st.markdown("### Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Points", len(df_filtered))
    col2.metric("Detected Anomalies", anomaly_df.shape[0])
    col3.metric("Average Risk Score", round(df_filtered[riskscore_col].mean(),2))

    # -------------------------------
    # Download CSV
    if not anomaly_df.empty:
        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Anomaly Report",
            data=csv,
            file_name="anomaly_report.csv",
            mime="text/csv"
        )

    # -------------------------------
    # Feature Charts
    feature_cols = df_filtered.select_dtypes(include=np.number).columns.drop([anomaly_col, risk_col, riskscore_col], errors='ignore')
    for feat in feature_cols:
        with st.expander(f"{feat} over Time"):
            df_filtered[f"{feat}_rolling"] = df_filtered[feat].rolling(3, min_periods=1).mean()
            fig = px.line(df_filtered, x='timestamp', y=[feat, f"{feat}_rolling"], title=f"{feat} & Rolling Mean")
            if not anomaly_df.empty:
                fig.add_scatter(
                    x=anomaly_df['timestamp'],
                    y=anomaly_df[feat],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8)
                )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # 3D Flight Path
    if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
        st.markdown("### 🗺️ 3D Drone Flight Path")
        try:
            fig3d = px.scatter_3d(
                df_filtered,
                x='longitude',
                y='latitude',
                z='altitude',
                color=df_filtered[anomaly_col].apply(lambda x: 'Anomaly' if x==1 else 'Normal'),
                color_discrete_map={'Normal':'blue','Anomaly':'red'},
                size_max=8,
                hover_data=['timestamp','velocity','yaw','pitch','battery','gps_drift']
            )
            fig3d.update_traces(marker=dict(size=4, line=dict(width=1, color='DarkSlateGrey')))
            fig3d.update_layout(scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Altitude (m)'
            ))
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating 3D flight path: {e}")
    else:
        st.info("Latitude and Longitude columns not detected. 3D flight path visualization unavailable.")

    # -------------------------------
    # Parameter Analysis & Recommended Actions
    st.markdown("### 📊 Parameter Analysis & Recommended Actions")
    if anomaly_df.empty:
        st.info("No anomalies detected after applying filters.")
    else:
        analysis_summary = []
        if 'altitude' in anomaly_df.columns:
            count = (anomaly_df['altitude'] > alt_threshold).sum()
            if count>0: analysis_summary.append(f"- 🚨 Altitude spikes detected in {count} instances.")
        if 'battery' in anomaly_df.columns:
            count = (anomaly_df['battery'] < battery_threshold).sum()
            if count>0: analysis_summary.append(f"- 🔋 Low battery detected in {count} instances.")
        if 'gps_drift' in anomaly_df.columns:
            count = (anomaly_df['gps_drift'] > gps_threshold).sum()
            if count>0: analysis_summary.append(f"- 📡 GPS drift anomalies detected in {count} instances.")
        if 'velocity' in anomaly_df.columns:
            count = (anomaly_df['velocity'] > 15).sum()
            if count>0: analysis_summary.append(f"- ⚡ Velocity spikes detected in {count} instances.")
        count_yaw = (anomaly_df['yaw'].abs() > 1).sum() if 'yaw' in anomaly_df.columns else 0
        count_pitch = (anomaly_df['pitch'].abs() > 1).sum() if 'pitch' in anomaly_df.columns else 0
        total = count_yaw + count_pitch
        if total>0: analysis_summary.append(f"- 🧭 Orientation anomalies detected in {total} instances.")

        for line in analysis_summary:
            st.warning(line)

        st.markdown("#### Recommended Actions")
        for line in analysis_summary:
            if "Altitude" in line: st.write("- Check drone sensors and flight path for altitude spikes.")
            if "Battery" in line: st.write("- Recharge drone battery and check battery health.")
            if "GPS" in line: st.write("- Calibrate GPS or avoid interference zones.")
            if "Velocity" in line: st.write("- Inspect control signals or environmental factors.")
            if "Orientation" in line: st.write("- Check stabilization system and sensors.")
