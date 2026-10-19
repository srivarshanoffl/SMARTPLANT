import os
import math
import warnings
import joblib
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Compatibility shim: models serialised with sklearn 1.6.1 use a
# private class (_RemainderColsList) removed in 1.7+.
# Must be injected BEFORE any joblib.load() call.
# ------------------------------------------------------------------
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ------------------------------------------------------------------
# Feature-engineering schema (must match training exactly)
# ------------------------------------------------------------------
BASE_REQUIRED_FEATURES = [
    'Usage_kWh',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM',
    'Load_Type',
    'WeekStatus',
    'Day_of_week',
]

ENGINEERED_NUM_COLS = [
    'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_16', 'lag_32', 'lag_96', 'lag_192',
    'roll_mean_4', 'roll_std_4', 'roll_mean_16', 'roll_std_16', 'roll_mean_96',
    'roll_max_4', 'roll_min_4',
    'delta_1', 'delta_4',
    'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'is_weekend',
    'pf_lag', 'reactive_ratio', 'nsm_norm',
    'current_ramp', 'apparent_power', 'high_load_streak', 'is_tariff_transition',
    'q_delta_1', 'pf_delta_1', 'apparent_power_delta_1', 'ramp_acceleration',
    'delta_from_yesterday', 'delta_from_last_week', 'load_type_duration', 'is_high_risk_hour',
]

ALL_NUM_COLS = [
    'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
    'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM',
] + ENGINEERED_NUM_COLS

ALL_CAT_COLS = ['Load_Type', 'WeekStatus', 'Day_of_week']

# ------------------------------------------------------------------
# Input Validation Helper
# ------------------------------------------------------------------
def _validate_input_features(features_dict, history=None):
    """
    Validate input features and historical data for schema completeness,
    finite numeric types, and physical plausibility in a steel plant context.
    """
    missing = [col for col in BASE_REQUIRED_FEATURES if col not in features_dict]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    numeric_checks = {
        'Usage_kWh': (0.0, 1000.0, "Usage_kWh must be non-negative and finite"),
        'Lagging_Current_Reactive.Power_kVarh': (0.0, 1000.0, "Lagging reactive power must be non-negative and finite"),
        'Leading_Current_Reactive_Power_kVarh': (0.0, 1000.0, "Leading reactive power must be non-negative and finite"),
        'CO2(tCO2)': (0.0, 10.0, "CO2 emission must be non-negative and finite"),
        'Lagging_Current_Power_Factor': (0.0, 100.0, "Lagging power factor must be between 0.0 and 100.0"),
        'Leading_Current_Power_Factor': (0.0, 100.0, "Leading power factor must be between 0.0 and 100.0"),
        'NSM': (0.0, 86400.0, "NSM (seconds from midnight) must be between 0 and 86400"),
    }

    for col, (low, high, msg) in numeric_checks.items():
        val = features_dict[col]
        try:
            f_val = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{col}' must be a numeric value, got '{val}'.")
        
        if math.isnan(f_val) or math.isinf(f_val):
            raise ValueError(f"Feature '{col}' contains non-finite value ({f_val}). Explicit sensor telemetry required.")
        
        if f_val < low or f_val > high:
            raise ValueError(f"{msg}; got {f_val}.")

    if history is not None:
        if not isinstance(history, (list, tuple, np.ndarray, pd.Series)):
            raise TypeError("history must be an iterable list, array, or Series of past Usage_kWh floats.")
        for idx, h_val in enumerate(history):
            try:
                f_h = float(h_val)
            except (TypeError, ValueError):
                raise ValueError(f"History index {idx} contains non-numeric value: '{h_val}'.")
            if math.isnan(f_h) or math.isinf(f_h):
                raise ValueError(f"History index {idx} contains non-finite value ({f_h}).")
            if f_h < 0.0:
                raise ValueError(f"History index {idx} contains negative energy usage ({f_h}).")


def _add_engineered_features_single(row_dict, history=None, extra_telemetry=None):
    d = dict(row_dict)
    usage_now = float(d['Usage_kWh'])

    if history is None:
        history = []
    else:
        history = list(history)

    def lag(n):
        idx = len(history) - n
        if idx >= 0:
            return float(history[idx])
        return usage_now

    d['lag_1']   = lag(1)
    d['lag_2']   = lag(2)
    d['lag_4']   = lag(4)
    d['lag_8']   = lag(8)
    d['lag_16']  = lag(16)
    d['lag_32']  = lag(32)
    d['lag_96']  = lag(96)
    d['lag_192'] = lag(192)

    hist_arr = np.array(history, dtype=float) if len(history) > 0 else np.array([usage_now])

    def roll_stat(n, fn):
        window = hist_arr[-n:] if len(hist_arr) >= 1 else np.array([usage_now])
        return float(fn(window))

    d['roll_mean_4']  = roll_stat(4,  np.mean)
    d['roll_std_4']   = float(np.std(hist_arr[-4:], ddof=1))  if len(hist_arr) >= 2 else 0.0
    d['roll_mean_16'] = roll_stat(16, np.mean)
    d['roll_std_16']  = float(np.std(hist_arr[-16:], ddof=1)) if len(hist_arr) >= 2 else 0.0
    d['roll_mean_96'] = roll_stat(96, np.mean)
    d['roll_max_4']   = roll_stat(4,  np.max)
    d['roll_min_4']   = roll_stat(4,  np.min)

    d['delta_1'] = lag(1) - lag(2)
    d['delta_4'] = lag(1) - lag(5)

    nsm = float(d['NSM'])
    frac_day = nsm / 86400.0
    d['sin_hour']  = float(np.sin(2 * np.pi * frac_day))
    d['cos_hour']  = float(np.cos(2 * np.pi * frac_day))
    d['nsm_norm']  = frac_day

    _dow_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6,
    }
    dow_idx = _dow_map.get(str(d.get('Day_of_week', 'Monday')), 0)
    d['sin_dow']   = float(np.sin(2 * np.pi * dow_idx / 7))
    d['cos_dow']   = float(np.cos(2 * np.pi * dow_idx / 7))
    d['is_weekend'] = int(dow_idx >= 5)

    d['pf_lag']          = float(d['Lagging_Current_Power_Factor']) * d['lag_1']
    d['reactive_ratio']  = float(d['Lagging_Current_Reactive.Power_kVarh']) / (usage_now + 1e-3)
    d['current_ramp']    = usage_now - d['lag_1']
    d['apparent_power']  = float(np.sqrt(usage_now**2 + float(d['Lagging_Current_Reactive.Power_kVarh'])**2))

    if usage_now <= 40.0:
        streak = 0
    else:
        streak = 1
        for v in reversed(history):
            if float(v) > 40.0:
                streak += 1
            else:
                break
    d['high_load_streak'] = streak

    next_nsm = (int(d['NSM']) + 900) % 86400
    tariff_switch_nsm = {33300, 36900, 44100, 47700, 62100, 72900, 80100, 83700}
    d['is_tariff_transition'] = int(next_nsm in tariff_switch_nsm)

    # 8 Enhanced precursor telemetry features
    extra = extra_telemetry or {}
    d['q_delta_1']              = float(extra.get('q_delta_1', 0.0))
    d['pf_delta_1']             = float(extra.get('pf_delta_1', 0.0))
    d['apparent_power_delta_1'] = float(extra.get('apparent_power_delta_1', d['current_ramp']))
    d['ramp_acceleration']      = float(d['current_ramp'] - (d['lag_1'] - d['lag_2']))
    d['delta_from_yesterday']   = float(usage_now - d['lag_96'])
    d['delta_from_last_week']   = float(usage_now - lag(672))
    d['load_type_duration']     = int(extra.get('load_type_duration', 1))
    hour = int(d['NSM']) // 3600
    d['is_high_risk_hour']      = int(hour in (8, 9, 14, 15, 16))

    return d


# ------------------------------------------------------------------
# Model Loader & In-Memory Cache
# ------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_MODEL_FILENAMES = {
    'breakthrough': 'energy_model_breakthrough.joblib',
    'ensemble':  'energy_model_ensemble.joblib',
    'optimized': 'energy_model_optimized.joblib',
    'improved':  'energy_model_improved.joblib',
    'baseline':  'energy_model.joblib',
}

_MODEL_LABELS = {
    'breakthrough': ('breakthrough', True),
    'ensemble':  ('ensemble',  True),
    'optimized': ('optimized', True),
    'improved':  ('improved',  True),
    'baseline':  ('baseline',  False),
}

_CACHED_MODELS = {}


def _resolve_model_path(filename):
    candidates = [
        os.path.join(_CURRENT_DIR, '..', 'models', filename),
        os.path.join(_CURRENT_DIR, '..', 'smartplant_output', filename),
        os.path.join(_CURRENT_DIR, 'models', filename),
        os.path.join(_CURRENT_DIR, 'smartplant_output', filename),
        os.path.join(_CURRENT_DIR, '..', 'SmartPlantAI_Project', 'smartplant_output', filename),
        os.path.join(_CURRENT_DIR, '..', 'smartplant-ai', 'smartplant_output', filename),
        os.path.join('SmartPlantAI_Project', 'smartplant_output', filename),
        os.path.join('smartplant-ai', 'smartplant_output', filename),
        os.path.join('smartplant_output', filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def get_prediction_pipeline(model_name=None, force_reload=False):
    global _CACHED_MODELS

    checked_paths = []
    items_to_check = (
        [(model_name, _MODEL_FILENAMES[model_name])]
        if model_name and model_name in _MODEL_FILENAMES
        else list(_MODEL_FILENAMES.items())
    )

    for tag, filename in items_to_check:
        path = _resolve_model_path(filename)
        checked_paths.append(path)
        if os.path.exists(path):
            if not force_reload and tag in _CACHED_MODELS:
                model_obj, is_eng = _CACHED_MODELS[tag]
                return model_obj, tag, is_eng

            model_obj = joblib.load(path)
            _, is_eng = _MODEL_LABELS[tag]
            _CACHED_MODELS[tag] = (model_obj, is_eng)
            return model_obj, tag, is_eng

    raise FileNotFoundError(
        f"No trained energy forecasting model found (requested: {model_name}). "
        'Checked paths:\n' + '\n'.join(checked_paths)
    )


# ------------------------------------------------------------------
# Predict function (Single Interval)
# ------------------------------------------------------------------
def predict_next_interval(features_dict, history=None, model_name=None, extra_telemetry=None, return_metadata=False):
    _validate_input_features(features_dict, history=history)

    model_obj, model_tag, is_engineered = get_prediction_pipeline(model_name=model_name)

    h_len = len(history) if history is not None else 0
    if h_len >= 192:
        inference_mode = 'full_history'
    elif h_len >= 4:
        inference_mode = 'partial_history'
    else:
        inference_mode = 'cold_start_fallback'

    usage_now = float(features_dict['Usage_kWh'])

    if is_engineered:
        row = _add_engineered_features_single(features_dict, history=history, extra_telemetry=extra_telemetry)
        input_df = pd.DataFrame([row])[ALL_NUM_COLS + ALL_CAT_COLS]

        if isinstance(model_obj, dict) and model_obj.get('type') == 'delta_ensemble':
            models  = model_obj['models']
            weights = model_obj['weights']
            prep    = model_obj['preprocessor']
            num_cols = model_obj.get('num_cols', ALL_NUM_COLS)
            cat_cols = model_obj.get('cat_cols', ALL_CAT_COLS)
            X_prep  = prep.transform(input_df[num_cols + cat_cols])

            p_xgb = float(models['xgboost'].predict(X_prep)[0])
            p_lgb = float(models['lightgbm'].predict(X_prep)[0])
            delta_hat = weights['xgboost'] * p_xgb + weights['lightgbm'] * p_lgb
            prediction = usage_now + delta_hat

        elif isinstance(model_obj, dict) and model_obj.get('type') == 'weighted_ensemble':
            models  = model_obj['models']
            weights = model_obj['weights']
            prep    = model_obj['preprocessor']
            num_cols = model_obj.get('num_cols', ALL_NUM_COLS)
            cat_cols = model_obj.get('cat_cols', ALL_CAT_COLS)
            X_prep  = prep.transform(input_df[num_cols + cat_cols])

            p_lgb = float(models['lightgbm'].predict(X_prep)[0])
            p_xgb = float(models['xgboost'].predict(X_prep)[0])
            p_rf  = float(models['rf'].predict(input_df)[0])
            prediction = weights['lightgbm'] * p_lgb + weights['xgboost'] * p_xgb + weights['rf'] * p_rf

        elif isinstance(model_obj, dict) and model_obj.get('type') == 'manual_hgb':
            ohe      = model_obj['ohe']
            model    = model_obj['model']
            num_part = input_df[model_obj['num_cols']].values
            cat_part = ohe.transform(input_df[model_obj['cat_cols']])
            X = np.hstack([num_part, cat_part])
            prediction = model.predict(X)[0]
        else:
            prediction = model_obj.predict(input_df)[0]

    else:
        base_cols = [
            'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
            'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor',
            'NSM', 'Load_Type', 'WeekStatus', 'Day_of_week',
        ]
        row = {col: features_dict[col] for col in base_cols}
        input_df = pd.DataFrame([row])[base_cols]
        prediction = model_obj.predict(input_df)[0]

    pred_val = float(max(0.0, prediction))

    if return_metadata:
        metadata = {
            'model_tag': model_tag,
            'inference_mode': inference_mode,
            'history_points_supplied': h_len,
            'is_cold_start': (inference_mode == 'cold_start_fallback'),
            'load_type': features_dict.get('Load_Type'),
            'is_advisory_estimate': True,
        }
        return pred_val, metadata

    return pred_val


# ------------------------------------------------------------------
# Sequential Inference Helper
# ------------------------------------------------------------------
def predict_sequential(records, initial_history=None):
    history = list(initial_history) if initial_history is not None else []
    predictions = []
    modes = []

    prev_q = None
    prev_pf = None
    curr_load_type = None
    load_duration = 0

    for rec in records:
        extra_telemetry = {}
        if prev_q is not None and 'Lagging_Current_Reactive.Power_kVarh' in rec:
            rec_q = float(rec['Lagging_Current_Reactive.Power_kVarh'])
            rec_pf = float(rec['Lagging_Current_Power_Factor'])
            rec_u = float(rec['Usage_kWh'])
            extra_telemetry['q_delta_1'] = rec_q - prev_q
            extra_telemetry['pf_delta_1'] = rec_pf - prev_pf
            app_power_curr = np.sqrt(rec_u**2 + rec_q**2)
            lag1 = float(history[-1]) if len(history) > 0 else rec_u
            extra_telemetry['apparent_power_delta_1'] = app_power_curr - np.sqrt(lag1**2 + prev_q**2)

        if rec.get('Load_Type') == curr_load_type:
            load_duration += 1
        else:
            curr_load_type = rec.get('Load_Type')
            load_duration = 1
        extra_telemetry['load_type_duration'] = load_duration

        pred, meta = predict_next_interval(rec, history=history, extra_telemetry=extra_telemetry, return_metadata=True)
        predictions.append(pred)
        modes.append(meta['inference_mode'])
        history.append(float(rec['Usage_kWh']))

        if 'Lagging_Current_Reactive.Power_kVarh' in rec:
            prev_q = float(rec['Lagging_Current_Reactive.Power_kVarh'])
            prev_pf = float(rec['Lagging_Current_Power_Factor'])

    return {
        'predictions': predictions,
        'final_history': history,
        'modes': modes
    }


# ------------------------------------------------------------------
# __main__: Smoke Test
# ------------------------------------------------------------------
if __name__ == '__main__':
    dummy_data = {
        'Usage_kWh': 80.96,
        'Lagging_Current_Reactive.Power_kVarh': 54.07,
        'Leading_Current_Reactive_Power_kVarh': 0.0,
        'CO2(tCO2)': 0.04,
        'Lagging_Current_Power_Factor': 83.16,
        'Leading_Current_Power_Factor': 100.0,
        'NSM': 81900,
        'Load_Type': 'Medium_Load',
        'WeekStatus': 'Weekday',
        'Day_of_week': 'Friday',
    }

    # Cold-start (no history)
    result_cold, meta_cold = predict_next_interval(dummy_data, return_metadata=True)
    print(f"Cold-start: {result_cold:.4f} kWh (Mode: {meta_cold['inference_mode']})")

    # Warm-start (10 points)
    hist = [75.0, 77.2, 78.5, 79.1, 80.0, 80.4, 80.8, 80.9, 80.96, 80.96]
    result_warm, meta_warm = predict_next_interval(dummy_data, history=hist, return_metadata=True)
    print(f"Warm-start: {result_warm:.4f} kWh (Mode: {meta_warm['inference_mode']})")

    # Sequential rollout test
    seq_records = [dummy_data, dummy_data, dummy_data]
    seq_res = predict_sequential(seq_records, initial_history=[70.0, 75.0])
    print(f"Sequential rollout (3 steps): {seq_res['predictions']}")

    # Non-finite and bounds check
    bad = dict(dummy_data, Usage_kWh=-5.0)
    try:
        predict_next_interval(bad)
    except ValueError as e:
        print(f"Negative usage blocked successfully: {e}")

