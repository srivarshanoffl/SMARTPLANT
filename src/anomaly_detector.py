"""
SmartPlant AI - Anomaly Detection Module
----------------------------------------
Implements robust, statistically grounded anomaly detection for 15-minute
steel plant electrical telemetry and forecasting deviations.

Methodology:
- Median and Median Absolute Deviation (MAD) on historical residuals:
    MAD = median(|e - median(e)|)
    Robust Z = 0.6745 * (e - median(e)) / MAD
- Quantile-calibrated severity thresholds (Low, Medium, High, Critical)
  fit strictly on historical training/calibration data without future leakage.
- Distinguishes:
    1. Data Quality Issues (NaN, Inf, negatives, invalid telemetry)
    2. Forecast Deviations (significant discrepancy between model expectation and meter)
    3. Consumption Anomalies (extreme consumption spikes or unexpected drops)
"""

import os
import json
import math
import numpy as np
import pandas as pd


class AnomalyConfig:
    """Configurable thresholds for anomaly detection."""
    def __init__(self,
                 mad_multiplier_low=2.0,
                 mad_multiplier_med=3.0,
                 mad_multiplier_high=4.5,
                 mad_multiplier_crit=6.0,
                 min_pred_for_pct_error=1.0,
                 usage_high_percentile=99.5,
                 usage_low_percentile=0.5):
        self.mad_multiplier_low = mad_multiplier_low
        self.mad_multiplier_med = mad_multiplier_med
        self.mad_multiplier_high = mad_multiplier_high
        self.mad_multiplier_crit = mad_multiplier_crit
        self.min_pred_for_pct_error = min_pred_for_pct_error
        self.usage_high_percentile = usage_high_percentile
        self.usage_low_percentile = usage_low_percentile

    def to_dict(self):
        return {
            'mad_multiplier_low': self.mad_multiplier_low,
            'mad_multiplier_med': self.mad_multiplier_med,
            'mad_multiplier_high': self.mad_multiplier_high,
            'mad_multiplier_crit': self.mad_multiplier_crit,
            'min_pred_for_pct_error': self.min_pred_for_pct_error,
            'usage_high_percentile': self.usage_high_percentile,
            'usage_low_percentile': self.usage_low_percentile,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class AnomalyResult:
    """Result of anomaly inspection for a single 15-minute interval."""
    def __init__(self,
                 is_anomaly=False,
                 anomaly_type='normal',
                 severity='None',
                 error_kWh=0.0,
                 absolute_error_kWh=0.0,
                 percentage_error=None,
                 z_score_mad=0.0,
                 explanation='',
                 suggested_action='',
                 details=None):
        self.is_anomaly = is_anomaly
        self.anomaly_type = anomaly_type  # 'normal', 'forecast_deviation', 'consumption_spike', 'consumption_drop', 'data_quality'
        self.severity = severity          # 'None', 'Low', 'Medium', 'High', 'Critical'
        self.error_kWh = error_kWh
        self.absolute_error_kWh = absolute_error_kWh
        self.percentage_error = percentage_error
        self.z_score_mad = z_score_mad
        self.explanation = explanation
        self.suggested_action = suggested_action
        self.details = details or {}

    def to_dict(self):
        return {
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'error_kWh': round(self.error_kWh, 4) if self.error_kWh is not None else None,
            'absolute_error_kWh': round(self.absolute_error_kWh, 4) if self.absolute_error_kWh is not None else None,
            'percentage_error': round(self.percentage_error, 2) if self.percentage_error is not None else None,
            'z_score_mad': round(self.z_score_mad, 3) if self.z_score_mad is not None else None,
            'explanation': self.explanation,
            'suggested_action': self.suggested_action,
            'details': self.details
        }


class SmartPlantAnomalyDetector:
    """
    Intelligent Anomaly Detection Engine for SmartPlant AI.
    Fits calibrated residual and telemetry thresholds strictly from training data.
    """
    def __init__(self, config=None):
        self.config = config or AnomalyConfig()
        self.is_fitted = False
        self.calibration_stats = {}

    def fit(self, actual_train, predicted_train, load_types=None):
        """
        Fit error distributions and consumption boundaries on historical training data.
        Never use test data or future observations.
        """
        act = np.asarray(actual_train, dtype=float)
        pred = np.asarray(predicted_train, dtype=float)

        # Filter valid pairs
        valid_mask = np.isfinite(act) & np.isfinite(pred) & (act >= 0)
        act = act[valid_mask]
        pred = pred[valid_mask]

        if len(act) < 100:
            raise ValueError(f"Insufficient training observations for calibration ({len(act)}). Minimum 100 required.")

        residuals = act - pred
        abs_residuals = np.abs(residuals)

        # Global residual statistics
        med_res = float(np.median(residuals))
        mad_res = float(np.median(np.abs(residuals - med_res)))
        # Bounded MAD to prevent division by zero in deterministic subsets
        mad_res = max(mad_res, 0.5)

        # Global consumption statistics
        usage_p_low = float(np.percentile(act, self.config.usage_low_percentile))
        usage_p_high = float(np.percentile(act, self.config.usage_high_percentile))
        usage_med = float(np.median(act))

        # Absolute error quantile thresholds
        err_p90 = float(np.percentile(abs_residuals, 90.0))
        err_p95 = float(np.percentile(abs_residuals, 95.0))
        err_p99 = float(np.percentile(abs_residuals, 99.0))
        err_p998 = float(np.percentile(abs_residuals, 99.8))

        stats = {
            'n_samples': int(len(act)),
            'global_residual_median': med_res,
            'global_residual_mad': mad_res,
            'global_error_quantiles': {
                'p90': err_p90,
                'p95': err_p95,
                'p99': err_p99,
                'p998': err_p998
            },
            'global_usage_bounds': {
                'low': usage_p_low,
                'high': usage_p_high,
                'median': usage_med
            },
            'load_type_bounds': {}
        }

        # Load-type specific distributions if load_types provided
        if load_types is not None:
            lt_series = pd.Series(load_types)[valid_mask].values
            for lt in np.unique(lt_series):
                lt_mask = (lt_series == lt)
                if np.sum(lt_mask) >= 50:
                    lt_act = act[lt_mask]
                    lt_res = residuals[lt_mask]
                    lt_med_res = float(np.median(lt_res))
                    lt_mad_res = max(float(np.median(np.abs(lt_res - lt_med_res))), 0.5)
                    stats['load_type_bounds'][str(lt)] = {
                        'count': int(np.sum(lt_mask)),
                        'usage_p05': float(np.percentile(lt_act, 1.0)),
                        'usage_p99': float(np.percentile(lt_act, 99.0)),
                        'residual_median': lt_med_res,
                        'residual_mad': lt_mad_res,
                        'err_p95': float(np.percentile(np.abs(lt_res), 95.0)),
                        'err_p99': float(np.percentile(np.abs(lt_res), 99.0)),
                    }

        self.calibration_stats = stats
        self.is_fitted = True
        return self

    def detect(self, actual_usage, predicted_usage, telemetry=None):
        """
        Evaluate a single completed 15-minute interval for anomalies.
        Ensures prediction corresponds strictly to the same target interval.
        """
        telemetry = telemetry or {}

        # ---------------------------------------------------------
        # 1. Data Quality Checks
        # ---------------------------------------------------------
        if actual_usage is None or predicted_usage is None:
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type='data_quality',
                severity='High',
                explanation='Telemetry reading or forecast prediction is missing for this interval.',
                suggested_action='Inspect billing meter network connection and ensure forecasting model service is active.'
            )

        try:
            act = float(actual_usage)
            pred = float(predicted_usage)
        except (ValueError, TypeError):
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type='data_quality',
                severity='Critical',
                explanation=f'Non-numeric telemetry received (actual={actual_usage}, predicted={predicted_usage}).',
                suggested_action='Verify data ingestion pipeline and data type formatting from substation RTU.'
            )

        if math.isnan(act) or math.isinf(act):
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type='data_quality',
                severity='Critical',
                explanation=f'Sensor returned non-finite reading ({act}) on main revenue meter.',
                suggested_action='Check for CT/PT transducer saturation, power interruption, or meter firmware fault.'
            )

        if act < 0.0:
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type='data_quality',
                severity='High',
                explanation=f'Negative energy usage detected ({act:.2f} kWh). Steel plant facility is a load, not a generator.',
                suggested_action='Inspect meter polarity and check for reverse current flow alarms on incomer circuit breaker.'
            )

        # ---------------------------------------------------------
        # 2. Prediction Error Calculation
        # ---------------------------------------------------------
        error_kWh = act - pred
        abs_error_kWh = abs(error_kWh)

        pct_error = None
        if pred >= self.config.min_pred_for_pct_error:
            pct_error = (error_kWh / pred) * 100.0

        # If not calibrated yet, use transparent heuristic fallback
        if not self.is_fitted:
            # Cold-start heuristic bounds
            is_anom = abs_error_kWh > 20.0
            sev = 'Medium' if is_anom else 'None'
            return AnomalyResult(
                is_anomaly=is_anom,
                anomaly_type='forecast_deviation' if is_anom else 'normal',
                severity=sev,
                error_kWh=error_kWh,
                absolute_error_kWh=abs_error_kWh,
                percentage_error=pct_error,
                explanation='Uncalibrated heuristic evaluation: prediction difference exceeds 20.0 kWh threshold.' if is_anom else 'Normal consumption within default uncalibrated envelope.',
                suggested_action='Review operational log for shift change.' if is_anom else 'None'
            )

        # ---------------------------------------------------------
        # 3. Robust Z-score & Severity Calibration
        # ---------------------------------------------------------
        load_type = telemetry.get('Load_Type')
        lt_stats = self.calibration_stats.get('load_type_bounds', {}).get(str(load_type))

        if lt_stats is not None:
            med_res = lt_stats['residual_median']
            mad_res = lt_stats['residual_mad']
            p90 = lt_stats.get('err_p95', 10.0) * 0.8
            p95 = lt_stats.get('err_p95', 12.0)
            p99 = lt_stats.get('err_p99', 20.0)
            p998 = p99 * 1.35
            usage_low = lt_stats['usage_p05']
            usage_high = lt_stats['usage_p99']
        else:
            med_res = self.calibration_stats['global_residual_median']
            mad_res = self.calibration_stats['global_residual_mad']
            q = self.calibration_stats['global_error_quantiles']
            p90 = q['p90']
            p95 = q['p95']
            p99 = q['p99']
            p998 = q['p998']
            usage_low = self.calibration_stats['global_usage_bounds']['low']
            usage_high = self.calibration_stats['global_usage_bounds']['high']

        # Robust Z-score using MAD (0.6745 scales MAD to std of Gaussian)
        z_score = 0.6745 * (error_kWh - med_res) / mad_res
        abs_z = abs(z_score)

        # Determine Severity based on calibrated MAD multiplier and quantile bounds
        if abs_z >= self.config.mad_multiplier_crit or abs_error_kWh >= p998:
            severity = 'Critical'
        elif abs_z >= self.config.mad_multiplier_high or abs_error_kWh >= p99:
            severity = 'High'
        elif abs_z >= self.config.mad_multiplier_med or abs_error_kWh >= p95:
            severity = 'Medium'
        elif abs_z >= self.config.mad_multiplier_low or abs_error_kWh >= p90:
            severity = 'Low'
        else:
            severity = 'None'

        is_anomaly = (severity != 'None')

        # ---------------------------------------------------------
        # 4. Anomaly Categorization & Explanations
        # ---------------------------------------------------------
        anomaly_type = 'normal'
        explanation = 'Energy consumption matches statistical forecast within expected operating envelope.'
        suggested_action = 'No action required. Normal steady-state operation.'

        if is_anomaly:
            # Check if this is an extreme consumption spike or drop
            if act > usage_high and error_kWh > 0:
                anomaly_type = 'consumption_spike'
                explanation = f'Extreme energy spike: Usage ({act:.2f} kWh) reached {self.config.usage_high_percentile}th percentile of historical operations (+{error_kWh:.2f} kWh above forecast).'
                suggested_action = 'Check melt-shop operational log to confirm if multiple electric arc furnaces or secondary refining stands started concurrently.'
            elif act < usage_low and pred > usage_low:
                anomaly_type = 'consumption_drop'
                explanation = f'Severe energy drop: Usage ({act:.2f} kWh) fell below historical {self.config.usage_low_percentile}th percentile (-{abs(error_kWh):.2f} kWh below forecast).'
                suggested_action = 'Inspect SCADA breaker trip events, emergency stop logs, or unannounced ladle changeover delays.'
            else:
                # Forecast deviation
                anomaly_type = 'forecast_deviation'
                direction = 'above' if error_kWh > 0 else 'below'
                explanation = f'Substantial forecast deviation: Actual consumption ({act:.2f} kWh) is {abs_error_kWh:.2f} kWh {direction} predicted baseline ({pred:.2f} kWh; Robust Z = {z_score:+.2f}).'
                suggested_action = 'Cross-reference shift changeover timing and review secondary auxiliary loads (rolling mill pumps, cooling towers, dust collection fans).'

        # Extra telemetry checks: Power Factor Warning
        pf_lag = telemetry.get('Lagging_Current_Power_Factor')
        if pf_lag is not None:
            try:
                f_pf = float(pf_lag)
                if f_pf < 70.0 and act > 15.0:
                    explanation += f' Note: Lagging Power Factor is critically degraded ({f_pf:.1f}%).'
                    suggested_action += ' Inspect capacitor bank steps and automatic harmonic filter switching.'
            except Exception:
                pass

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            error_kWh=error_kWh,
            absolute_error_kWh=abs_error_kWh,
            percentage_error=pct_error,
            z_score_mad=z_score,
            explanation=explanation,
            suggested_action=suggested_action,
            details={
                'actual_usage': act,
                'predicted_usage': pred,
                'load_type': load_type,
                'robust_z': z_score,
                'threshold_p95': p95,
                'threshold_p99': p99
            }
        )

    def save_calibration(self, filepath):
        """Save calibration statistics to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'config': self.config.to_dict(),
            'stats': self.calibration_stats,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'w') as f:
            json.dump(bundle, f, indent=4)

    evaluate_interval = detect

    @classmethod
    def load_calibration(cls, filepath=None):
        """Load calibrated detector from JSON."""
        if filepath is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), '..', 'models', 'anomaly_detector_calibration.json'),
                os.path.join(os.path.dirname(__file__), '..', 'smartplant_output', 'anomaly_detector_calibration.json'),
                os.path.join(os.path.dirname(__file__), 'models', 'anomaly_detector_calibration.json'),
                os.path.join(os.path.dirname(__file__), 'smartplant_output', 'anomaly_detector_calibration.json'),
                os.path.join('models', 'anomaly_detector_calibration.json'),
                os.path.join('smartplant_output', 'anomaly_detector_calibration.json'),
            ]
            for c in candidates:
                if os.path.exists(c):
                    filepath = c
                    break
        if filepath is None or not os.path.exists(filepath):
            return cls()
        with open(filepath, 'r') as f:
            bundle = json.load(f)
        cfg = AnomalyConfig.from_dict(bundle.get('config', {}))
        detector = cls(config=cfg)
        detector.calibration_stats = bundle.get('stats', {})
        detector.is_fitted = bundle.get('is_fitted', False)
        return detector
