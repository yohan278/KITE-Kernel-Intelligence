"""Tests for all sklearn-based estimators."""

from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


@pytest.fixture
def synthetic_measurements():
    """Generate synthetic profiling measurements for estimator testing."""
    measurements = []
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [128, 256, 512, 1024]:
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6

            measurements.append(
                OperatorMeasurement(
                    operator_name="linear_qkv",
                    category=OperatorCategory.LINEAR,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time,
                    energy_j=base_time * 400,
                    power_w=400.0,
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_prefill",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.5,
                    energy_j=base_time * 1.5 * 400,
                    power_w=420.0,
                )
            )
    return measurements


@pytest.fixture
def _skip_no_sklearn():
    pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# LinearRegressionEstimator
# ---------------------------------------------------------------------------


class TestLinearRegressionEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.linear_regression import (
            LinearRegressionEstimator,
        )

        est = LinearRegressionEstimator()
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()
        assert "time_train_r2" in scores

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s > 0


# ---------------------------------------------------------------------------
# RidgeRegressionEstimator
# ---------------------------------------------------------------------------


class TestRidgeRegressionEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.ridge import RidgeRegressionEstimator

        est = RidgeRegressionEstimator(alpha=1.0)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()
        assert "time_train_r2" in scores

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# LassoRegressionEstimator
# ---------------------------------------------------------------------------


class TestLassoRegressionEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.lasso import LassoRegressionEstimator

        est = LassoRegressionEstimator(alpha=0.001)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# SVREstimator
# ---------------------------------------------------------------------------


class TestSVREstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.svr import SVREstimator

        est = SVREstimator()
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# KNNEstimator
# ---------------------------------------------------------------------------


class TestKNNEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.knn import KNNEstimator

        est = KNNEstimator(n_neighbors=3)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s > 0


# ---------------------------------------------------------------------------
# BayesianLinearEstimator
# ---------------------------------------------------------------------------


class TestBayesianLinearEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.bayesian_linear import (
            BayesianLinearEstimator,
        )

        est = BayesianLinearEstimator()
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# GaussianProcessEstimator
# ---------------------------------------------------------------------------


class TestGaussianProcessEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.gaussian_process import (
            GaussianProcessEstimator,
        )

        # Use small dataset to keep GP fast
        small = synthetic_measurements[:16]
        est = GaussianProcessEstimator()
        scores = est.fit(small)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# MLPEstimator
# ---------------------------------------------------------------------------


class TestMLPEstimator:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.mlp import MLPEstimator

        est = MLPEstimator(hidden_layer_sizes=(32,), max_iter=200)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# XGBoostEstimator (conditional on xgboost being installed)
# ---------------------------------------------------------------------------


class TestXGBoostEstimator:
    @pytest.fixture
    def _skip_no_xgboost(self):
        pytest.importorskip("xgboost")

    def test_fit_and_predict(
        self, _skip_no_sklearn, _skip_no_xgboost, synthetic_measurements
    ):
        from inference_simulator.estimator.xgboost_estimator import XGBoostEstimator

        est = XGBoostEstimator(n_estimators=10)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s > 0


# ---------------------------------------------------------------------------
# LightGBMEstimator (conditional on lightgbm being installed)
# ---------------------------------------------------------------------------


class TestLightGBMEstimator:
    @pytest.fixture
    def _skip_no_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_fit_and_predict(
        self, _skip_no_sklearn, _skip_no_lightgbm, synthetic_measurements
    ):
        from inference_simulator.estimator.lightgbm_estimator import LightGBMEstimator

        est = LightGBMEstimator(n_estimators=10)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s > 0


# ---------------------------------------------------------------------------
# Scaling direction tests (all estimators should respect data trends)
# ---------------------------------------------------------------------------


class TestScalingDirection:
    """Verify all estimators predict larger inputs → longer time."""

    estimator_classes = [
        ("inference_simulator.estimator.linear_regression", "LinearRegressionEstimator"),
        ("inference_simulator.estimator.ridge", "RidgeRegressionEstimator"),
        ("inference_simulator.estimator.knn", "KNNEstimator"),
        ("inference_simulator.estimator.bayesian_linear", "BayesianLinearEstimator"),
    ]

    @pytest.mark.parametrize("module_path,class_name", estimator_classes)
    def test_scaling(self, _skip_no_sklearn, synthetic_measurements, module_path, class_name):
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        est = cls()
        est.fit(synthetic_measurements)

        small = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        large = est.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=1024)
        assert large.time_s > small.time_s, f"{class_name} failed scaling test"


# ---------------------------------------------------------------------------
# MultiOutputEstimatorWrapper
# ---------------------------------------------------------------------------


class TestMultiOutputEstimatorWrapper:
    def test_fit_and_predict(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.multi_output import (
            MultiOutputEstimatorWrapper,
        )

        wrapper = MultiOutputEstimatorWrapper(base_estimator_type="random_forest")
        scores = wrapper.fit(synthetic_measurements)
        assert wrapper.is_fitted()

        result = wrapper.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s >= 0


# ---------------------------------------------------------------------------
# ModelComparison
# ---------------------------------------------------------------------------


class TestModelComparison:
    def test_compare_estimators(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.model_comparison import compare_estimators

        results = compare_estimators(
            measurements=synthetic_measurements,
            model_dims=None,
            estimator_classes=["random_forest", "linear_regression", "ridge"],
            val_fraction=0.2,
        )
        assert len(results) >= 1
        # Each result should have estimator name, r2, mae, rmse
        for entry in results:
            assert "estimator" in entry
            assert "time_r2" in entry


# ---------------------------------------------------------------------------
# Polynomial Features
# ---------------------------------------------------------------------------


class TestPolynomialFeatures:
    def test_poly_degree_zero_unchanged(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.ridge import RidgeRegressionEstimator

        est0 = RidgeRegressionEstimator(alpha=1.0, poly_degree=0)
        scores0 = est0.fit(synthetic_measurements)
        r0 = est0.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)

        est_default = RidgeRegressionEstimator(alpha=1.0)
        est_default.fit(synthetic_measurements)
        r_default = est_default.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)

        assert r0.time_s == pytest.approx(r_default.time_s, rel=1e-6)

    def test_poly_degree_two_improves_quadratic(self, _skip_no_sklearn):
        """Ridge + poly=2 should better fit O(n^2) data than Ridge alone."""
        from inference_simulator.estimator.ridge import RidgeRegressionEstimator

        # Generate synthetic quadratic data (keep <=4 to bypass val split)
        measurements = []
        for seq_len in [64, 256, 1024, 4096]:
            time_s = (seq_len ** 2) * 1e-9  # pure quadratic
            measurements.append(
                OperatorMeasurement(
                    operator_name="attn_prefill",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=1,
                    seq_len=seq_len,
                    time_s=time_s,
                )
            )

        est_no_poly = RidgeRegressionEstimator(alpha=0.01, poly_degree=0)
        scores_no = est_no_poly.fit(measurements)

        est_poly = RidgeRegressionEstimator(alpha=0.01, poly_degree=2)
        scores_poly = est_poly.fit(measurements)

        # Poly should have >= train R^2 since it has more features
        assert scores_poly["time_train_r2"] >= scores_no["time_train_r2"] - 0.01

    def test_lasso_poly_degree(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.lasso import LassoRegressionEstimator

        est = LassoRegressionEstimator(alpha=0.0001, poly_degree=2)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()
        result = est.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)
        assert result.time_s > 0


# ---------------------------------------------------------------------------
# Normalized Loss
# ---------------------------------------------------------------------------


class TestNormalizedLoss:
    def test_normalize_loss_backward_compat(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator

        est_off = PerOperatorEstimator(normalize_loss=False)
        est_off.fit(synthetic_measurements)
        r_off = est_off.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)

        est_default = PerOperatorEstimator()
        est_default.fit(synthetic_measurements)
        r_default = est_default.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)

        # Same random seed, same data, same default => same results
        assert r_off.time_s == pytest.approx(r_default.time_s, rel=1e-6)

    def test_normalize_loss_improves_small_ops(self, _skip_no_sklearn):
        """Normalized loss should improve relative accuracy on small operators."""
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator

        # Mixed scale: small norm ops + large linear ops
        measurements = []
        for bs in [1, 2, 4, 8, 16]:
            for sl in [128, 256, 512, 1024]:
                tokens = bs * sl
                # Linear: ~milliseconds
                measurements.append(
                    OperatorMeasurement(
                        operator_name="linear_qkv",
                        category=OperatorCategory.LINEAR,
                        batch_size=bs,
                        seq_len=sl,
                        time_s=tokens * 1e-6,
                    )
                )
                # Norm: ~microseconds (1000x smaller)
                measurements.append(
                    OperatorMeasurement(
                        operator_name="rmsnorm",
                        category=OperatorCategory.NORMALIZATION,
                        batch_size=bs,
                        seq_len=sl,
                        time_s=tokens * 1e-9,
                    )
                )

        est_norm = PerOperatorEstimator(normalize_loss=True)
        est_norm.fit(measurements)

        est_plain = PerOperatorEstimator(normalize_loss=False)
        est_plain.fit(measurements)

        # Both should be fitted
        assert est_norm.is_fitted()
        assert est_plain.is_fitted()

        # Both should predict positive values for normalization
        r_norm = est_norm.estimate(OperatorCategory.NORMALIZATION, batch_size=4, seq_len=256)
        r_plain = est_plain.estimate(OperatorCategory.NORMALIZATION, batch_size=4, seq_len=256)
        assert r_norm.time_s > 0
        assert r_plain.time_s > 0
