"""Tests for pairwise ranking configuration."""

import pytest
from pydantic import ValidationError

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    MilvusConfig,
    RetrievalConfig,
)
from rankers.config.pairwise_ranking_config import (
    LLMConfig,
    PairwiseMethod,
    PairwiseRankingConfig,
)


class TestPairwiseRankingConfig:
    """Tests for the pairwise ranking configuration models."""

    def test_pairwise_method_enum_values(self) -> None:
        """Test that PairwiseMethod enum contains all expected values."""
        assert PairwiseMethod.HEAPSORT == "heapsort"
        assert PairwiseMethod.BUBBLESORT == "bubblesort"
        assert PairwiseMethod.ALLPAIR == "allpair"

    def test_llm_config_device_stripped_to_none(self) -> None:
        """Test that LLMConfig.ensure_device_is_none strips 'device' from model_kwargs.

        The model_validator must set model_kwargs['device'] to None when present,
        matching the same pattern as EmbeddingConfig.
        """
        config = LLMConfig(
            model_name="test-model",
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
        assert config.model_kwargs["device"] is None
        assert config.model_kwargs["trust_remote_code"] is True

    def test_llm_config_no_device_key_unchanged(self) -> None:
        """Test LLMConfig leaves model_kwargs unchanged when 'device' key absent."""
        config = LLMConfig(
            model_name="test-model", model_kwargs={"trust_remote_code": True}
        )
        assert "device" not in config.model_kwargs

    def test_llm_config_top_k_gt_zero_constraint(self) -> None:
        """Test LLMConfig raises ValidationError when top_k=0 (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            LLMConfig(model_name="test-model", top_k=0)

    def test_llm_config_defaults(self) -> None:
        """Test LLMConfig initializes with expected default values."""
        config = LLMConfig(model_name="test-model")
        assert config.method == PairwiseMethod.HEAPSORT
        assert config.top_k == 10
        assert config.device is None
        assert config.model_kwargs == {}
        assert config.tokenizer_kwargs == {}

    def test_pairwise_ranking_config_valid_instantiation(self) -> None:
        """Test PairwiseRankingConfig instantiates correctly with required configs."""
        dataset = DatasetConfig(name="beir/fiqa/train")
        embedding = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
        milvus = MilvusConfig(
            connection_uri="http://localhost:19530", connection_token="token"
        )
        retrieval = RetrievalConfig(dataset=dataset, embedding=embedding, milvus=milvus)
        llm = LLMConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            method=PairwiseMethod.BUBBLESORT,
            top_k=5,
        )

        config = PairwiseRankingConfig(
            dataset=dataset,
            llm=llm,
            embedding=embedding,
            milvus=milvus,
            retrieval=retrieval,
        )

        assert config.llm.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.llm.method == PairwiseMethod.BUBBLESORT
        assert config.llm.top_k == 5
        assert config.dataset.name == "beir/fiqa/train"
