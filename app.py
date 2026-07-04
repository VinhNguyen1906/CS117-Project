import pickle
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

project_dir = Path(__file__).resolve().parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from realtime_demo import RealTimeRiskMonitor, read_live_system_metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class LiveMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Fan Risk Monitor")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.bind("<KeyPress>", self.handle_key)

        self.stop_event = Event()
        self.running = False
        self.start_time = None
        self.interval_seconds = 1.0

        self.sample_indices = []
        self.risk_scores = []
        self.log_lines = []

        self._build_ui()
        self._prepare_model()
        self.start_monitoring()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Live Fan Risk Monitor", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(top, text="Nhấn ` hoặc q để dừng. Chạy liên tục cho đến khi bạn dừng thủ công.", font=("Segoe UI", 10)).pack(anchor="w")

        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(6, 0))
        self.status_var = tk.StringVar(value="Đang khởi động...")
        self.latest_state_var = tk.StringVar(value="State: --")
        self.latest_score_var = tk.StringVar(value="Score: --")

        ttk.Label(controls, textvariable=self.status_var, foreground="#1f4e79").pack(side="left")
        ttk.Label(controls, textvariable=self.latest_state_var, foreground="#0b6e4f", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(18, 6))
        ttk.Label(controls, textvariable=self.latest_score_var, foreground="#8a2be2", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(controls, text="Dừng", command=self.stop_monitoring).pack(side="right")

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        self.figure, self.ax = plt.subplots(figsize=(8, 4.5), dpi=100)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.line, = self.ax.plot([], [], color="#1f77b4", linewidth=2, label="Risk score")
        self.ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Threshold")
        self.ax.set_title("Risk score theo thời gian")
        self.ax.set_xlabel("Sample index")
        self.ax.set_ylabel("Risk score")
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")

        right = ttk.Frame(body, width=320)
        right.pack(side="right", fill="y")
        right.grid_columnconfigure(0, weight=1)
        ttk.Label(right, text="Log", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self.log_box = tk.Text(right, height=30, width=42, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)

    def _prepare_model(self):
        artifact_path = project_dir / "trained_models.pkl"
        if not artifact_path.exists():
            raise FileNotFoundError("Không tìm thấy trained_models.pkl. Hãy chạy lại các cell huấn luyện trong demo.ipynb trước khi chạy app.py.")

        with artifact_path.open("rb") as f:
            payload = pickle.load(f)

        self.model = payload.get("model")
        self.ensemble_models = {
            "rf_calibrated": payload.get("rf_calibrated", payload.get("model")),
            "isolation_forest": payload["isolation_forest"],
            "logistic_regression": payload["logistic_regression"],
        }
        self.scaler = payload["scaler"]
        self.feature_names = payload.get("feature_names", [])
        self.raw_feature_names = payload.get("raw_feature_names", [])
        self.monitor = RealTimeRiskMonitor(
            model=self.model,
            scaler=self.scaler,
            feature_names=self.feature_names,
            raw_feature_names=self.raw_feature_names,
            threshold=payload.get("threshold", 0.5),
            window_size=10,
        )

        self.log("Đã tải mô hình huấn luyện sẵn từ demo.ipynb.")

    def log(self, message):
        self.log_lines.append(message)
        if len(self.log_lines) > 120:
            self.log_lines = self.log_lines[-120:]

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", "\n".join(self.log_lines[-80:]))
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def start_monitoring(self):
        self.running = True
        self.start_time = time.time()
        self.status_var.set("Đang giám sát...")
        self.log("Bắt đầu giám sát live...")
        self._tick()

    def stop_monitoring(self):
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        self.status_var.set("Đã dừng")
        self.log("Đã dừng giám sát.")
        self._save_results()

    def close_app(self):
        self.stop_monitoring()
        self.root.destroy()

    def handle_key(self, event):
        if event.char in {"`", "q", "Q"}:
            self.stop_monitoring()

    def _tick(self):
        if not self.running:
            return

        sample = read_live_system_metrics()
        result = self.monitor.update(sample)
        if result is not None:
            rf_score = float(result["risk_score"])

            feature_df = pd.DataFrame([sample], columns=self.raw_feature_names)
            feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0.0)
            scaled_df = pd.DataFrame(self.scaler.transform(feature_df), columns=self.feature_names)

            try:
                iso_pred = self.ensemble_models["isolation_forest"].predict(scaled_df)[0]
                iso_risk = 1.0 if iso_pred == -1 else 0.0
            except Exception:
                iso_risk = 0.0

            try:
                logit_proba = self.ensemble_models["logistic_regression"].predict_proba(scaled_df)[0]
                logit_risk = float(logit_proba[1] + logit_proba[2]) if len(logit_proba) >= 3 else float(logit_proba[1])
            except Exception:
                logit_risk = 0.0

            ensemble_score = float(np.clip((rf_score + iso_risk + logit_risk) / 3.0, 0.0, 1.0))
            state = "Risk" if ensemble_score >= 0.5 else "Normal"

            self.sample_indices.append(len(self.sample_indices))
            self.risk_scores.append(ensemble_score)
            if len(self.sample_indices) > 200:
                self.sample_indices = self.sample_indices[-200:]
                self.risk_scores = self.risk_scores[-200:]

            self.line.set_data(self.sample_indices, self.risk_scores)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()

            ts = time.strftime("%H:%M:%S")
            self.latest_state_var.set(f"State: {state}")
            self.latest_score_var.set(f"Score: {ensemble_score:.3f}")
            self.log(f"[{ts}] rf={rf_score:.3f} | iso={iso_risk:.3f} | logit={logit_risk:.3f} | ensemble={ensemble_score:.3f} | state={state}")

        self.root.after(int(self.interval_seconds * 1000), self._tick)

    def _save_results(self):
        if not self.risk_scores:
            return
        out_path = project_dir / "live_monitor_results.csv"
        df = pd.DataFrame({
            "sample_index": self.sample_indices,
            "risk_score": self.risk_scores,
        })
        df.to_csv(out_path, index=False)
        self.log(f"Đã lưu kết quả vào {out_path.name}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveMonitorApp(root)
    root.mainloop()
