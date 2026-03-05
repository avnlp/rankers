"""Tests for base configuration classes."""

import pytest
from pydantic import ValidationError

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    IndexingConfig,
    MilvusConfig,
    RetrievalConfig,
)


class TestBaseConfig:
    """Tests for the base configuration models in base_config.py."""

    def test_embedding_config_device_stripped_to_none(self) -> None:
        """Test that EmbeddingConfig strips the 'device' key from model_kwargs.

        When model_kwargs contains a 'device' key, the ensure_device_is_none
        validator must set its value to None to avoid ComponentDevice issues.
        """
        config = EmbeddingConfig(
            model="test-model",
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
        # device must be overwritten to None by the validator
        assert config.model_kwargs["device"] is None
        # other keys should be untouched
        assert config.model_kwargs["trust_remote_code"] is True

    def test_embedding_config_no_device_key_passes_through(self) -> None:
        """Test EmbeddingConfig leaves model_kwargs unchanged when no device key.

        If model_kwargs does not contain a 'device' key, the validator must not
        modify the dict in any way.
        """
        config = EmbeddingConfig(
            model="test-model", model_kwargs={"trust_remote_code": True}
        )
        assert "device" not in config.model_kwargs
        assert config.model_kwargs == {"trust_remote_code": True}

    def test_embedding_config_default_model_kwargs(self) -> None:
        """Test EmbeddingConfig initializes model_kwargs to empty dict by default."""
        config = EmbeddingConfig(model="test-model")
        assert config.model_kwargs == {}

    def test_evaluation_config_defaults(self) -> None:
        """Test that EvaluationConfig initializes with correct default values."""
        config = EvaluationConfig()
        assert config.cutoff_values == [1, 3, 5, 10]
        assert config.ignore_identical_ids is False
        assert config.decimal_precision == 4
        assert set(config.metrics_to_compute) == {"ndcg", "map", "recall", "precision"}

    def test_evaluation_config_custom_values(self) -> None:
        """Test that EvaluationConfig accepts valid custom values."""
        config = EvaluationConfig(
            cutoff_values=[5, 10],
            ignore_identical_ids=True,
            decimal_precision=2,
            metrics_to_compute=["ndcg", "recall"],
        )
        assert config.cutoff_values == [5, 10]
        assert config.ignore_identical_ids is True
        assert config.decimal_precision == 2
        assert config.metrics_to_compute == ["ndcg", "recall"]

    def test_evaluation_config_decimal_precision_out_of_bounds_low(self) -> None:
        """Test EvaluationConfig raises ValidationError when decimal_precision < 0."""
        with pytest.raises(ValidationError):
            EvaluationConfig(decimal_precision=-1)

    def test_evaluation_config_decimal_precision_out_of_bounds_high(self) -> None:
        """Test EvaluationConfig raises ValidationError when decimal_precision > 6."""
        with pytest.raises(ValidationError):
            EvaluationConfig(decimal_precision=7)

    def test_retrieval_config_documents_to_retrieve_gt_zero(self) -> None:
        """Test RetrievalConfig raises ValidationError for documents_to_retrieve < 1."""
        with pytest.raises(ValidationError):
            RetrievalConfig(
                dataset=DatasetConfig(name="test"),
                embedding=EmbeddingConfig(model="test-model"),
                milvus=MilvusConfig(
                    connection_uri="http://localhost:19530", connection_token="token"
                ),
                documents_to_retrieve=0,
            )

    def test_retrieval_config_valid_instantiation(self) -> None:
        """Test that RetrievalConfig instantiates correctly with all required fields."""
        config = RetrievalConfig(
            dataset=DatasetConfig(name="beir/fiqa/train"),
            embedding=EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2"),
            milvus=MilvusConfig(
                connection_uri="http://localhost:19530", connection_token="token"
            ),
            documents_to_retrieve=25,
        )
        assert config.documents_to_retrieve == 25
        assert config.dataset.name == "beir/fiqa/train"

    def test_milvus_config_valid_instantiation(self) -> None:
        """Test MilvusConfig instantiates with required fields and empty kwargs dict."""
        config = MilvusConfig(
            connection_uri="http://localhost:19530", connection_token="mytoken"
        )
        assert config.connection_uri == "http://localhost:19530"
        assert config.connection_token == "mytoken"
        assert config.document_store_kwargs == {}

    def test_indexing_config_missing_required_fields_raises(self) -> None:
        """Test IndexingConfig raises ValidationError when required fields missing."""
        with pytest.raises(ValidationError):
            # DatasetConfig requires 'name'; omitting it should fail
            IndexingConfig(
                dataset={},
                embedding=EmbeddingConfig(model="test-model"),
                milvus=MilvusConfig(
                    connection_uri="http://localhost:19530", connection_token="token"
                ),
            )

    def test_dataset_config_valid(self) -> None:
        """Test that DatasetConfig correctly stores the dataset name."""
        config = DatasetConfig(name="beir/fiqa/train")
        assert config.name == "beir/fiqa/train"
