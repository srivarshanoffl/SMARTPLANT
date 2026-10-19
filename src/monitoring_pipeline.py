"""
SmartPlant AI - Energy Monitoring Pipeline & Historical Replay
--------------------------------------------------------------
Coordinates real-time and simulated actual vs. predicted energy monitoring,
error tracking, anomaly detection, and alert dispatch.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from src.predict import predict_next_interval, BASE_REQUIRED_FEATURES
except ImportError:
    from predict import predict_next_interval, BASE_REQUIRED_FEATURES
try:
    from src.anomaly_detector import SmartPlantAnomalyDetector, AnomalyConfig
except ImportError:
    from anomaly_detector import SmartPlantAnomalyDetector, AnomalyConfig
try:
    from src.alert_engine import AlertEngine
except ImportError:
    from alert_engine import AlertEngine


class MonitoringIntervalRecord:
    """Historical audit record for a single monitored 15-minute interval."""
    def __init__(self,
                 timestamp,
                 interval_index,
                 actual_usage_kWh,
                 predicted_usage_kWh,
                 error_kWh,
                 absolute_error_kWh,
                 percentage_error=None,
                 is_anomaly=False,
                 anomaly_type='normal',
                 severity='None',
                 alert_id=None,
                 load_type='Medium_Load',
                 is_simulation=False):
        self.timestamp = str(timestamp)
        self.interval_index = interval_index
        self.actual_usage_kWh = actual_usage_kWh
        self.predicted_usage_kWh = predicted_usage_kWh
        self.error_kWh = error_kWh
        self.absolute_error_kWh = absolute_error_kWh
        self.percentage_error = percentage_error
        self.is_anomaly = is_anomaly
        self.anomaly_type = anomaly_type
        self.severity = severity
        self.alert_id = alert_id
        self.load_type = load_type
        self.is_simulation = is_simulation

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'interval_index': self.interval_index,
            'actual_usage_kWh': round(self.actual_usage_kWh, 2) if self.actual_usage_kWh is not None else None,
            'predicted_usage_kWh': round(self.predicted_usage_kWh, 2) if self.predicted_usage_kWh is not None else None,
            'error_kWh': round(self.error_kWh, 2) if self.error_kWh is not None else None,
            'absolute_error_kWh': round(self.absolute_error_kWh, 2) if self.absolute_error_kWh is not None else None,
            'percentage_error': round(self.percentage_error, 1) if self.percentage_error is not None else None,
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'alert_id': self.alert_id,
            'load_type': self.load_type,
            'is_simulation': self.is_simulation
        }


class SmartPlantMonitoringCenter:
    """
    Central controller orchestrating:
    1. Forecast retrieval for interval t+1 at time t
    2. Actual observation capture at time t+1
    3. Error & anomaly calculation
    4. Alert dispatch & historical replay simulation
    """
    def __init__(self,
                 detector=None,
                 alert_engine=None,
                 calibration_path=None,
                 alerts_path=None,
                 history_path=None,
                 auto_load=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'smartplant_output')
        os.makedirs(output_dir, exist_ok=True)

        self.calibration_path = calibration_path or os.path.join(output_dir, 'anomaly_detector_calibration.json')
        self.alerts_path = alerts_path or os.path.join(output_dir, 'alerts_store.json')
        self.history_path = history_path or os.path.join(output_dir, 'monitoring_history.json')
        self.auto_load = auto_load

        # Load or instantiate detector
        if detector:
            self.detector = detector
        elif os.path.exists(self.calibration_path):
            self.detector = SmartPlantAnomalyDetector.load_calibration(self.calibration_path)
        else:
            self.detector = SmartPlantAnomalyDetector()

        # Load or instantiate alert engine
        self.alert_engine = alert_engine or AlertEngine(storage_path=self.alerts_path)

        self.monitoring_history = []
        self.current_history_buffer = []  # rolling usage history for predict.py
        self.pending_prediction = None    # Prediction generated at t for interval t+1
        self.pending_meta = None
        self.interval_counter = 0

        if self.auto_load and self.history_path and os.path.exists(self.history_path):
            self._load_history()

    def record_prediction(self, features_dict, history=None):
        """
        Record a forecast for the NEXT 15-minute interval BEFORE it occurs.
        """
        if history is not None:
            self.current_history_buffer = list(history)

        pred, meta = predict_next_interval(features_dict, history=self.current_history_buffer, return_metadata=True)
        self.pending_prediction = pred
        self.pending_meta = meta
        return pred, meta

    def observe_interval(self, actual_usage_kWh, timestamp=None, telemetry_dict=None, is_simulation=False):
        """
        Process the actual observation once the interval completes.
        Compares prediction made for this interval against the actual reading.
        """
        telemetry_dict = telemetry_dict or {}
        self.interval_counter += 1
        ts = str(timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # If cold-start without prior prediction, make on-the-fly forecast using features
        if self.pending_prediction is None:
            if all(k in telemetry_dict for k in BASE_REQUIRED_FEATURES):
                pred, meta = self.record_prediction(telemetry_dict)
            else:
                pred = actual_usage_kWh  # fallback baseline
        else:
            pred = self.pending_prediction

        # Reset pending prediction for next cycle
        self.pending_prediction = None

        # Detect anomalies
        anomaly_res = self.detector.detect(actual_usage_kWh, pred, telemetry=telemetry_dict)

        # Dispatch alerts
        alert_rec, is_new = self.alert_engine.process_anomaly(
            anomaly_res,
            timestamp=ts,
            interval_index=self.interval_counter,
            actual_usage=actual_usage_kWh,
            predicted_usage=pred
        )

        alert_id = alert_rec.alert_id if alert_rec else None

        # Create record
        rec = MonitoringIntervalRecord(
            timestamp=ts,
            interval_index=self.interval_counter,
            actual_usage_kWh=actual_usage_kWh,
            predicted_usage_kWh=pred,
            error_kWh=anomaly_res.error_kWh,
            absolute_error_kWh=anomaly_res.absolute_error_kWh,
            percentage_error=anomaly_res.percentage_error,
            is_anomaly=anomaly_res.is_anomaly,
            anomaly_type=anomaly_res.anomaly_type,
            severity=anomaly_res.severity,
            alert_id=alert_id,
            load_type=telemetry_dict.get('Load_Type', 'Medium_Load'),
            is_simulation=is_simulation
        )

        self.monitoring_history.append(rec)
        # Update rolling buffer
        if actual_usage_kWh is not None and np.isfinite(actual_usage_kWh):
            self.current_history_buffer.append(float(actual_usage_kWh))
            if len(self.current_history_buffer) > 672:
                self.current_history_buffer = self.current_history_buffer[-672:]

        return rec, alert_rec

    def step_replay(self, row_dict, is_simulation=True):
        """
        Single step of historical simulation replay.
        Given a row from dataset, updates actual and prepares next interval.
        """
        actual_now = float(row_dict['Usage_kWh'])
        ts = str(row_dict.get('date', f"Interval_{self.interval_counter}"))

        # Observe interval using previously predicted value
        rec, alert = self.observe_interval(actual_now, timestamp=ts, telemetry_dict=row_dict, is_simulation=is_simulation)

        # Now predict for the subsequent interval
        self.record_prediction(row_dict)

        return rec, alert

    def run_replay_batch(self, rows_list, is_simulation=True):
        """Replay a batch of historical records chronologically."""
        results = []
        for r in rows_list:
            rec, alert = self.step_replay(r, is_simulation=is_simulation)
            results.append((rec, alert))
        self.save_history()
        return results

    def save_history(self, max_records=2000):
        """Save recent monitoring history to JSON."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        recent = self.monitoring_history[-max_records:]
        data = [r.to_dict() for r in recent]
        with open(self.history_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_history(self):
        """Load history from JSON."""
        try:
            with open(self.history_path, 'r') as f:
                data = json.load(f)
            self.monitoring_history = [MonitoringIntervalRecord(**d) for d in data]
            self.interval_counter = len(self.monitoring_history)
        except Exception:
            self.monitoring_history = []

    def get_recent_history(self, limit=100):
        """Return most recent monitoring intervals."""
        return [r.to_dict() for r in self.monitoring_history[-limit:]]

    def clear(self):
        """Reset monitoring center."""
        self.monitoring_history = []
        self.current_history_buffer = []
        self.pending_prediction = None
        self.interval_counter = 0
        self.alert_engine.clear()
        if os.path.exists(self.history_path):
            os.remove(self.history_path)
