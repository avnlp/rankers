from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(
        ...,
        description="Dataset identifier in ir_datasets format. Example: 'beir/fiqa/train' for BEIR FIQA dataset train split.",
    )


class EmbeddingConfig(BaseModel):
    """Configuration Schema Embedding Model."""

    model: str = Field(
        ...,
        description="Name of the embedding model to use. For example, 'sentence-transformers/all-MiniLM-L6-v2'.",
    )
    model_kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional model configuration parameters.")

    @model_validator(mode="after")
    def ensure_device_is_none(self):
        """Ensure device is always None to avoid ComponentDevice issues."""
        if "device" in self.model_kwargs:
            self.model_kwargs["device"] = None
        return self


class MilvusConfig(BaseModel):
    """Configuration for the Milvus vector database."""

    connection_uri: str = Field(..., description="Milvus server URI (e.g., 'http://localhost:19530')")

    connection_token: str = Field(..., description="Authentication token for Milvus")

    document_store_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters for MilvusDocumentStore"
    )


class IndexingConfig(BaseModel):
    """Configuration for the indexing pipeline."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics and settings."""

    cutoff_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10],
        description="Cutoff levels for evaluation metrics (e.g., NDCG@k). Multiple values supported.",
    )

    ignore_identical_ids: bool = Field(
        default=False,
        description="Exclude documents with same ID as query (prevents test set leakage)",
    )

    decimal_precision: int = Field(
        default=4,
        ge=0,
        le=6,
        description="Decimal precision for metric reporting. Range: 0-6",
    )

    metrics_to_compute: list[Literal["ndcg", "map", "recall", "precision"]] = Field(
        default_factory=lambda: ["ndcg", "map", "recall", "precision"],
        description="Evaluation metrics to compute.",
    )


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval pipeline."""

    dataset: DatasetConfig = Field(..., description="Dataset configuration")

    embedding: EmbeddingConfig = Field(..., description="Embedding model configuration")

    milvus: MilvusConfig = Field(..., description="Milvus vector database configuration")

    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration",
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata filters for retrieval",
    )

    documents_to_retrieve: int = Field(
        default=25,
        gt=0,
        description="Number of documents to retrieve per query",
    )
