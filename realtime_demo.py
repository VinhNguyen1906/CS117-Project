from collections import deque
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil


class RealTimeRiskMonitor:
    """Simulate a streaming sensor pipeline with rolling window features."""

    def __init__(self, model, scaler, feature_names, raw_feature_names, threshold=0.5, window_size=10):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.raw_feature_names = raw_feature_names
        self.threshold = threshold
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.results = []

    def _build_feature_vector(self):
        if len(self.window) == 0:
            return None

        window_df = pd.DataFrame(list(self.window))
        feature_values = {}

        for col in self.raw_feature_names:
            values = window_df[col].astype(float).to_numpy()
            feature_values[f"{col}_mean"] = float(np.mean(values))
            feature_values[f"{col}_std"] = float(np.std(values, ddof=0))
            feature_values[f"{col}_min"] = float(np.min(values))
            feature_values[f"{col}_max"] = float(np.max(values))
            feature_values[f"{col}_range"] = float(np.max(values) - np.min(values))

        latest = window_df.iloc[-1]
        feature_values["Fan_Temp_Ratio"] = float(latest["Fan_RPM"] / (latest["Temperature_C"] + 1e-6))

        vector = []
        for name in self.feature_names:
            if name in feature_values:
                vector.append(feature_values[name])
            elif name in window_df.columns:
                vector.append(float(latest[name]))
            else:
                raise KeyError(f"Feature {name} is missing from the streaming feature builder")

        feature_df = pd.DataFrame([vector], columns=self.feature_names)
        return feature_df

    def update(self, sample):
        self.window.append(sample)
        feature_vector = self._build_feature_vector()
        if feature_vector is None:
            return None

        scaled_vector = pd.DataFrame(
            self.scaler.transform(feature_vector),
            columns=self.feature_names,
        )

        proba = self.model.predict_proba(scaled_vector)[0]
        classes = list(self.model.classes_)
        if 0 in classes and 1 in classes and 2 in classes:
            risk_score = float(proba[classes.index(1)] + proba[classes.index(2)])
        else:
            risk_score = float(np.max(proba))

        predicted_label = int(self.model.predict(scaled_vector)[0])
        predicted_state = "Risk" if risk_score >= self.threshold else "Normal"

        result = {
            "risk_score": risk_score,
            "predicted_label": predicted_label,
            "predicted_state": predicted_state,
            "prob_normal": float(proba[classes.index(0)]) if 0 in classes else float(proba[0]),
            "prob_warning": float(proba[classes.index(1)]) if 1 in classes else 0.0,
            "prob_critical": float(proba[classes.index(2)]) if 2 in classes else 0.0,
        }
        self.results.append(result)
        return result


def read_live_system_metrics():
    cpu_load = psutil.cpu_percent(interval=None)

    temperature = None
    try:
        temps = []
        for entries in psutil.sensors_temperatures().values():
            for entry in entries:
                if getattr(entry, "current", None) is not None:
                    temps.append(float(entry.current))
        if temps:
            temperature = float(np.mean(temps))
    except Exception:
        temperature = None

    fan_rpm = None
    try:
        fans = []
        for entries in psutil.sensors_fans().values():
            for entry in entries:
                if getattr(entry, "current", None) is not None:
                    fans.append(float(entry.current))
        if fans:
            fan_rpm = float(np.mean(fans))
    except Exception:
        fan_rpm = None

    voltage = None
    try:
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery is not None and getattr(battery, "power_plugged", None) is not None:
                voltage = 12.0
    except Exception:
        voltage = None

    if temperature is None:
        temperature = 35.0 + min(25.0, cpu_load / 100.0 * 20.0)
    if fan_rpm is None:
        fan_rpm = 1200.0 + cpu_load * 3.5
    if voltage is None:
        voltage = 1.2 + (cpu_load / 100.0) * 0.08

    return {
        "CPU_Load_Pct": float(cpu_load),
        "Temperature_C": float(temperature),
        "Fan_RPM": float(fan_rpm),
        "Voltage_V": float(voltage),
    }


def run_stream_demo(stream_df, model, scaler, feature_names, raw_feature_names, threshold=0.5, window_size=10):
    monitor = RealTimeRiskMonitor(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        raw_feature_names=raw_feature_names,
        threshold=threshold,
        window_size=window_size,
    )

    rows = []
    for idx, row in stream_df.iterrows():
        sample = row.to_dict()
        result = monitor.update(sample)

        if {"Date", "Time"}.issubset(stream_df.columns):
            try:
                ts = pd.to_datetime(f"{row['Date']} {row['Time']}", errors="coerce")
            except Exception:
                ts = pd.Timestamp(idx)
        else:
            ts = pd.Timestamp(idx)

        row_result = {
            "timestamp": ts,
            "sample_index": idx,
            **result,
        }
        rows.append(row_result)

    results_df = pd.DataFrame(rows)
    return results_df


def run_live_sensor_demo(model, scaler, feature_names, raw_feature_names, threshold=0.5, window_size=10, duration_seconds=60, interval_seconds=1, verbose=False, max_samples=None, stop_condition=None, stop_event=None):
    monitor = RealTimeRiskMonitor(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        raw_feature_names=raw_feature_names,
        threshold=threshold,
        window_size=window_size,
    )

    rows = []
    deadline = time.time() + duration_seconds if duration_seconds is not None else None
    sample_count = 0

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if max_samples is not None and sample_count >= max_samples:
            break
        if stop_condition is not None and stop_condition(rows):
            break
        if deadline is not None and time.time() >= deadline:
            break

        sample = read_live_system_metrics()
        result = monitor.update(sample)
        if result is not None:
            sample_count += 1
            rows.append({
                "timestamp": pd.Timestamp.now(),
                "sample_index": len(rows),
                **result,
            })
            if verbose:
                print(f"[{rows[-1]['timestamp']}] risk={rows[-1]['risk_score']:.3f} state={rows[-1]['predicted_state']}")
        time.sleep(interval_seconds)

    return pd.DataFrame(rows)


def plot_stream_results(results_df, threshold, title="Real-time risk demo"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(results_df["timestamp"], results_df["risk_score"], color="#1f77b4", linewidth=1.4, label="Risk score")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="Threshold")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Risk score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
