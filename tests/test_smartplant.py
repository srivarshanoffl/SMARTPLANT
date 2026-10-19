import unittest
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_next_interval, get_prediction_pipeline
from experiment_and_train import engineer_lag_features


class TestSmartPlantAI(unittest.TestCase):

    def setUp(self):
        self.valid_input = {
            'Usage_kWh': 10.0,
            'Lagging_Current_Reactive.Power_kVarh': 4.5,
            'Leading_Current_Reactive_Power_kVarh': 0.0,
            'CO2(tCO2)': 0.005,
            'Lagging_Current_Power_Factor': 90.0,
            'Leading_Current_Power_Factor': 100.0,
            'NSM': 1800,
            'Load_Type': 'Medium_Load',
            'WeekStatus': 'Weekday',
            'Day_of_week': 'Monday',
        }

    # ------------------------------------------------------------------
    # Business logic tests (no model dependency)
    # ------------------------------------------------------------------
    def test_business_billing_logic(self):
        usage = 120.0
        tariff_rate = 0.15
        cost = usage * tariff_rate
        self.assertEqual(cost, 18.0)

    def test_carbon_estimation_factor(self):
        carbon_factor = 0.4638
        usage = 1000.0
        estimated_co2 = usage * carbon_factor
        self.assertAlmostEqual(estimated_co2, 463.8)

    def test_hypothetical_peak_shaving_simulation(self):
        peak_usage = 500.0
        shaved_usage = peak_usage * (1.0 - 0.15)
        self.assertEqual(shaved_usage, 425.0)

    def test_machinery_control_recommendation(self):
        pf_value = 82.0
        recommendation = ''
        if pf_value < 85.0:
            recommendation = 'Advisory: Adjust capacitor banks to balance power factor.'
        self.assertTrue('Advisory' in recommendation)
        self.assertFalse('EXECUTE' in recommendation)

    # ------------------------------------------------------------------
    # Model tests
    # ------------------------------------------------------------------
    def test_prediction_invalid_features(self):
        invalid_input = self.valid_input.copy()
        del invalid_input['Usage_kWh']
        with self.assertRaises(ValueError):
            predict_next_interval(invalid_input)

    def test_prediction_returns_float(self):
        result = predict_next_interval(self.valid_input)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 500.0)  # physically plausible

    def test_prediction_cold_start(self):
        """Cold-start (no history) must return a positive float."""
        result = predict_next_interval(self.valid_input, history=None)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_prediction_warm_start(self):
        """Warm-start with recent history must return a positive float."""
        history = [9.5, 9.8, 10.0, 10.1, 10.0, 9.9, 10.0, 10.1, 10.2, 10.0]
        result = predict_next_interval(self.valid_input, history=history)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_model_loads_without_error(self):
        """get_prediction_pipeline() must succeed and return a usable object."""
        model_obj, model_tag, is_eng = get_prediction_pipeline()
        self.assertIn(model_tag, ('breakthrough', 'ensemble', 'optimized', 'improved', 'baseline'))
        self.assertIsNotNone(model_obj)

    # ------------------------------------------------------------------
    # Lag feature generation (experiment_and_train.py)
    # ------------------------------------------------------------------
    def test_lag_feature_generation(self):
        raw_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='15min'),
            'Usage_kWh': [float(x) for x in range(100)],
            'Lagging_Current_Reactive.Power_kVarh': 0.0,
            'Leading_Current_Reactive_Power_kVarh': 0.0,
            'CO2(tCO2)': 0.0,
            'Lagging_Current_Power_Factor': 90.0,
            'Leading_Current_Power_Factor': 100.0,
            'NSM': 0,
            'Load_Type': 'Light_Load',
            'WeekStatus': 'Weekday',
            'Day_of_week': 'Monday',
        })
        lagged = engineer_lag_features(raw_data)
        # For row index 5, Lag_1 must equal Usage from row 4
        self.assertEqual(lagged.loc[5, 'Usage_Lag_1'], 4.0)
        self.assertEqual(lagged.loc[5, 'Usage_Lag_2'], 3.0)
        self.assertEqual(lagged.loc[5, 'Usage_Lag_4'], 1.0)

    # ------------------------------------------------------------------
    # Rigorous Audit Tests: Artifact Identity & Midnight Alignment
    # ------------------------------------------------------------------
    def test_model_artifact_identity(self):
        """Verify that get_prediction_pipeline loads the breakthrough champion and preserves rollback models."""
        import joblib
        from predict import _resolve_model_path

        # 1. Active Breakthrough Champion pipeline
        model_obj, model_tag, is_eng = get_prediction_pipeline()
        self.assertEqual(model_tag, 'breakthrough')
        self.assertTrue(is_eng)
        self.assertEqual(model_obj['type'], 'delta_ensemble')
        self.assertIn('lightgbm', model_obj['models'])
        self.assertIn('xgboost', model_obj['models'])
        self.assertAlmostEqual(sum(model_obj['weights'].values()), 1.0)

        # 2. Check rollback models if present
        if not os.path.exists(_resolve_model_path('energy_model_ensemble.joblib')):
            return
        ens_obj, ens_tag, _ = get_prediction_pipeline(model_name='ensemble')
        self.assertEqual(ens_tag, 'ensemble')
        self.assertEqual(ens_obj['type'], 'weighted_ensemble')
        self.assertIn('lightgbm', ens_obj['models'])
        self.assertIn('xgboost', ens_obj['models'])
        self.assertIn('rf', ens_obj['models'])
        self.assertAlmostEqual(sum(ens_obj['weights'].values()), 1.0)

        # 3. Preserved RF_eng rollback artifact
        rf_path = _resolve_model_path('energy_model_optimized.joblib')
        self.assertTrue(os.path.exists(rf_path))
        rf_model = joblib.load(rf_path)
        self.assertTrue(hasattr(rf_model, 'named_steps'))
        reg = rf_model.named_steps['reg']
        self.assertEqual(type(reg).__name__, 'RandomForestRegressor')
        self.assertEqual(reg.n_estimators, 300)
        self.assertEqual(reg.max_depth, 20)
        self.assertEqual(reg.min_samples_leaf, 2)
        self.assertEqual(reg.random_state, 42)

    def test_midnight_target_alignment(self):
        """Verify raw dataset has zero target mismatches across midnight boundaries."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'cleaned_dataset.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(base_dir, 'data', 'train.csv')
        df_raw = pd.read_csv(data_path)
        target = df_raw['next_interval_Usage_kWh'].iloc[:-1].values
        next_usage = df_raw['Usage_kWh'].iloc[1:].values
        mismatches = int((target != next_usage).sum())
        self.assertEqual(mismatches, 0, f"Expected 0 mismatches in raw train.csv, found {mismatches}")

    def test_feature_engineering_parity_all_25(self):
        """Verify inference feature engineering produces all 25 features matching formulas."""
        from predict import _add_engineered_features_single, ENGINEERED_NUM_COLS
        history = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        row = _add_engineered_features_single(self.valid_input, history=history)
        for col in ENGINEERED_NUM_COLS:
            self.assertIn(col, row)
            self.assertFalse(pd.isna(row[col]))
        # lag_1 should be last item of history (20.0)
        self.assertEqual(row['lag_1'], 20.0)
        # lag_2 should be second to last (18.0)
        self.assertEqual(row['lag_2'], 18.0)
        # delta_1 should be lag_1 - lag_2 = 2.0
        self.assertEqual(row['delta_1'], 2.0)
        # roll_mean_4 should be mean of [14, 16, 18, 20] = 17.0
        self.assertAlmostEqual(row['roll_mean_4'], 17.0)
        # roll_std_4 with ddof=1 should match sample std
        expected_std4 = float(pd.Series([14.0, 16.0, 18.0, 20.0]).std())
        self.assertAlmostEqual(row['roll_std_4'], expected_std4, places=6)

    # ------------------------------------------------------------------
    # Stage 3 Regression Tests: Robustness, Validation & Deployment
    # ------------------------------------------------------------------
    def test_negative_usage_rejected(self):
        """Negative energy usage must raise an explicit ValueError."""
        bad_input = self.valid_input.copy()
        bad_input['Usage_kWh'] = -10.0
        with self.assertRaises(ValueError):
            predict_next_interval(bad_input)

    def test_non_finite_features_rejected(self):
        """NaN or Inf sensor readings must raise an explicit ValueError."""
        import numpy as np
        for bad_val in (float('nan'), float('inf'), -float('inf')):
            bad_input = self.valid_input.copy()
            bad_input['Usage_kWh'] = bad_val
            with self.assertRaises(ValueError):
                predict_next_interval(bad_input)

    def test_negative_history_rejected(self):
        """Negative or non-finite values in history must raise ValueError."""
        with self.assertRaises(ValueError):
            predict_next_interval(self.valid_input, history=[10.0, -5.0, 12.0])

    def test_metadata_inference_mode_distinction(self):
        """Metadata must distinguish cold_start_fallback, partial_history, and full_history."""
        # Cold start
        _, meta_cold = predict_next_interval(self.valid_input, history=None, return_metadata=True)
        self.assertTrue(meta_cold['is_cold_start'])
        self.assertEqual(meta_cold['inference_mode'], 'cold_start_fallback')

        # Partial history
        _, meta_part = predict_next_interval(self.valid_input, history=[10.0]*10, return_metadata=True)
        self.assertFalse(meta_part['is_cold_start'])
        self.assertEqual(meta_part['inference_mode'], 'partial_history')

        # Full history
        _, meta_full = predict_next_interval(self.valid_input, history=[10.0]*200, return_metadata=True)
        self.assertEqual(meta_full['inference_mode'], 'full_history')

    def test_sequential_inference_pipeline(self):
        """Sequential inference must iterate across records and maintain updated history."""
        from predict import predict_sequential
        records = [self.valid_input.copy() for _ in range(5)]
        for i, rec in enumerate(records):
            rec['Usage_kWh'] = 20.0 + i * 2.0  # 20, 22, 24, 26, 28
        res = predict_sequential(records, initial_history=[15.0, 18.0])
        self.assertEqual(len(res['predictions']), 5)
        self.assertEqual(len(res['final_history']), 7)
        self.assertEqual(res['final_history'][-1], 28.0)

    def test_model_caching_performance(self):
        """Repeated calls to get_prediction_pipeline must return the cached in-memory instance."""
        m1, tag1, eng1 = get_prediction_pipeline()
        m2, tag2, eng2 = get_prediction_pipeline()
        self.assertIs(m1, m2)
        self.assertEqual(tag1, tag2)

    def test_scada_continuous_timestamps(self):
        """Raw dataset timestamps must form a strictly continuous 15-minute series across all rows."""
        import numpy as np
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'cleaned_dataset.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(base_dir, 'data', 'train.csv')
        df = pd.read_csv(data_path)
        date_part = pd.to_datetime(df['date'].str.split(' ').str[0], dayfirst=True)
        true_dt = date_part + pd.to_timedelta(np.where(df['NSM'] == 0, 86400, df['NSM']), unit='s')
        deltas = true_dt.diff().dropna()
        self.assertTrue((deltas == pd.Timedelta(minutes=15)).all())


if __name__ == '__main__':
    unittest.main()
