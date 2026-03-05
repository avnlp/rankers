"""Tests for listwise ranking configuration."""

import pytest
from pydantic import ValidationError

from rankers.config.listwise_ranking_config import (
    ListwiseHFConfig,
    ListwiseOpenAIConfig,
)


class TestListwiseRankingConfig:
    """Tests for listwise ranking configuration models."""

    def test_rank_gpt_without_openai_raises_value_error(self) -> None:
        """Test ListwiseHFConfig with ranker_type='rank_gpt', openai=None raises error.

        The validate_openai_config model_validator enforces that OpenAI configuration
        is required when ranker_type is 'rank_gpt'.
        """
        with pytest.raises(ValidationError):
            ListwiseHFConfig(
                model_path="test-model", ranker_type="rank_gpt", openai=None
            )

    def test_rank_gpt_with_openai_config_passes(self) -> None:
        """Test ListwiseHFConfig with ranker_type='rank_gpt' and valid OpenAI config."""
        openai_cfg = ListwiseOpenAIConfig(api_keys=["sk-test"])
        config = ListwiseHFConfig(
            model_path="test-model", ranker_type="rank_gpt", openai=openai_cfg
        )
        assert config.ranker_type == "rank_gpt"
        assert config.openai is not None

    def test_non_rank_gpt_with_openai_none_passes(self) -> None:
        """Test ListwiseHFConfig with ranker_type='zephyr' and openai=None is valid.

        The validator only enforces OpenAI config for rank_gpt, so other types
        can omit it.
        """
        config = ListwiseHFConfig(
            model_path="test-model", ranker_type="zephyr", openai=None
        )
        assert config.ranker_type == "zephyr"
        assert config.openai is None

    def test_context_size_must_be_gt_zero(self) -> None:
        """Test context_size=0 raises ValidationError due to gt=0 constraint."""
        with pytest.raises(ValidationError):
            ListwiseHFConfig(model_path="test-model", context_size=0)

    def test_top_k_must_be_gt_zero(self) -> None:
        """Test top_k=0 raises ValidationError due to gt=0 constraint."""
        with pytest.raises(ValidationError):
            ListwiseHFConfig(model_path="test-model", top_k=0)

    def test_sliding_window_size_must_be_gt_zero(self) -> None:
        """Test sliding_window_size=0 raises ValidationError due to gt=0 constraint."""
        with pytest.raises(ValidationError):
            ListwiseHFConfig(model_path="test-model", sliding_window_size=0)

    def test_sliding_window_step_must_be_gt_zero(self) -> None:
        """Test sliding_window_step=0 raises ValidationError due to gt=0 constraint."""
        with pytest.raises(ValidationError):
            ListwiseHFConfig(model_path="test-model", sliding_window_step=0)

    def test_listwise_hf_config_defaults(self) -> None:
        """Test ListwiseHFConfig initializes with expected default values."""
        config = ListwiseHFConfig(model_path="test-model")
        assert config.ranker_type == "zephyr"
        assert config.context_size == 4096
        assert config.top_k == 10
        assert config.sliding_window_size == 20
        assert config.sliding_window_step == 10
        assert config.openai is None

    def test_openai_config_all_optional_fields_none(self) -> None:
        """Test ListwiseOpenAIConfig instantiated with all optional fields as None."""
        config = ListwiseOpenAIConfig()
        assert config.api_keys is None
        assert config.key_start_id is None
        assert config.proxy is None
        assert config.api_type is None
        assert config.api_base is None
        assert config.api_version is None
