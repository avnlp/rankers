"""Tests for setwise ranking configuration."""

import pytest
from pydantic import ValidationError

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    MilvusConfig,
    RetrievalConfig,
)
from rankers.config.setwise_ranking_config import (
    SetwiseLLMConfig,
    SetwiseMethod,
    SetwiseRankingConfig,
)


class TestSetwiseRankingConfig:
    """Tests for setwise ranking configuration models."""

    def test_setwise_method_enum_values(self) -> None:
        """Test SetwiseMethod enum contains HEAPSORT and BUBBLESORT but not ALLPAIR."""
        assert SetwiseMethod.HEAPSORT == "heapsort"
        assert SetwiseMethod.BUBBLESORT == "bubblesort"
        assert not hasattr(SetwiseMethod, "ALLPAIR")

    def test_setwise_llm_config_device_stripped_to_none(self) -> None:
        """Test SetwiseLLMConfig.ensure_device_is_none sets model_kwargs device None."""
        config = SetwiseLLMConfig(
            model_name="test-model",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
        )
        assert config.model_kwargs["device"] is None
        assert config.model_kwargs["trust_remote_code"] is True

    def test_setwise_llm_config_top_k_gt_zero(self) -> None:
        """Test SetwiseLLMConfig raises ValidationError when top_k=0 (gt=0)."""
        with pytest.raises(ValidationError):
            SetwiseLLMConfig(model_name="test-model", top_k=0)

    def test_setwise_llm_config_num_permutation_ge_one(self) -> None:
        """Test SetwiseLLMConfig raises ValidationError when num_permutation=0."""
        with pytest.raises(ValidationError):
            SetwiseLLMConfig(model_name="test-model", num_permutation=0)

    def test_setwise_llm_config_num_child_ge_two(self) -> None:
        """Test SetwiseLLMConfig raises ValidationError when num_child=1 (ge=2)."""
        with pytest.raises(ValidationError):
            SetwiseLLMConfig(model_name="test-model", num_child=1)

    def test_setwise_llm_config_defaults(self) -> None:
        """Test SetwiseLLMConfig initializes with expected default values."""
        config = SetwiseLLMConfig(model_name="test-model")
        assert config.method == SetwiseMethod.HEAPSORT
        assert config.top_k == 10
        assert config.num_permutation == 1
        assert config.num_child == 3
        assert config.device is None
        assert config.model_kwargs == {}

    def test_setwise_ranking_config_valid_instantiation(self) -> None:
        """Test SetwiseRankingConfig instantiates correctly with required configs."""
        dataset = DatasetConfig(name="beir/fiqa/train")
        embedding = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
        milvus = MilvusConfig(
            connection_uri="http://localhost:19530", connection_token="token"
        )
        retrieval = RetrievalConfig(dataset=dataset, embedding=embedding, milvus=milvus)
        llm = SetwiseLLMConfig(
            model_name="test-model", num_permutation=2, num_child=4, top_k=5
        )

        config = SetwiseRankingConfig(
            dataset=dataset,
            llm=llm,
            embedding=embedding,
            milvus=milvus,
            retrieval=retrieval,
        )

        assert config.llm.model_name == "test-model"
        assert config.llm.num_permutation == 2
        assert config.llm.num_child == 4
        assert config.llm.top_k == 5
