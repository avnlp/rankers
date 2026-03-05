"""Tests for evaluation configuration."""

import pytest

from rankers.evaluation.evaluator_params import EvaluatorParams


class TestEvaluatorParams:
    """Test suite for EvaluatorParams class.

    Ensures correct init and validation.
    """

    def test_valid_config_initialization(self):
        """Test EvaluatorParams initializes with custom config values.

        Verifies that all config parameters are correctly set during
        initialization and can be retrieved with the exact values provided.
        """
        config = EvaluatorParams(
            cutoff_values=(1, 3, 5, 10),
            ignore_identical_ids=True,
            decimal_precision=4,
            metrics_to_compute=("ndcg", "map", "recall", "precision"),
        )

        # Verify each configuration parameter is set correctly
        assert config.cutoff_values == (1, 3, 5, 10)
        assert config.ignore_identical_ids is True
        assert config.decimal_precision == 4
        assert config.metrics_to_compute == ("ndcg", "map", "recall", "precision")

    def test_invalid_cutoff_values_raises_error(self):
        """Test that EvaluatorParams raises a ValueError for invalid cutoff values.

        Ensures that negative or zero cutoff values are not allowed,
        maintaining the integrity of metric computation.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(cutoff_values=(-1, 2))

    def test_invalid_decimal_precision_raises_error(self):
        """Test that EvaluatorParams raises a ValueError for invalid decimal precision.

        Verifies that only integer values are accepted for decimal precision,
        preventing potential rounding issues.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(decimal_precision=0.5)

    def test_invalid_metrics_raises_error(self):
        """Test that EvaluatorParams raises a ValueError for unsupported metrics.

        Ensures that only predefined, valid metrics can be specified during config,
        preventing potential errors during evaluation.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(metrics_to_compute=("invalid_metric",))

    def test_default_config_values(self):
        """Test EvaluatorParams provides sensible defaults when no params specified.

        Verifies that:
        - Cutoff values are not empty
        - Ignore identical IDs is a boolean
        - Decimal precision is an integer
        - Metrics to compute are not empty
        """
        config = EvaluatorParams()

        # Verify default configuration parameters
        assert len(config.cutoff_values) > 0
        assert isinstance(config.ignore_identical_ids, bool)
        assert isinstance(config.decimal_precision, int)
        assert len(config.metrics_to_compute) > 0

    def test_list_cutoff_values_raises_error(self):
        """Test that passing a list for cutoff_values raises a ValueError.

        EvaluatorParams requires cutoff_values to be a tuple, so a list triggers
        the type check in __post_init__ and raises ValueError.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(cutoff_values=[1, 3, 5])

    def test_list_metrics_to_compute_raises_error(self):
        """Test that passing a list for metrics_to_compute raises a ValueError.

        EvaluatorParams requires metrics_to_compute to be a tuple, so a list triggers
        the type check in __post_init__ and raises ValueError.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(metrics_to_compute=["ndcg", "map"])
