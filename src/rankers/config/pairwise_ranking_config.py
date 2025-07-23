from enum import Enum

from pydantic import BaseModel, Field, model_validator

from rankers.config.base_config import DatasetConfig, EmbeddingConfig, EvaluationConfig, MilvusConfig, RetrievalConfig


class PairwiseMethod(str, Enum):
    """Available pairwise ranking methods."""

    HEAPSORT = "heapsort"
    BUBBLESORT = "bubblesort"
    ALLPAIR = "allpair"


class LLMConfig(BaseModel):
    """Configuration for the LLM ranker."""

    model_name: str = Field(
        ..., description="Hugging Face model name/path for the LLM ranker. Example: 'meta-llama/Llama-3.1-8B-Instruct'"
    )
    method: PairwiseMethod = Field(
        default=PairwiseMethod.HEAPSORT,
        description="Sorting method to be used. Allpair: brute force pairwise comparisons, "
        "Heapsort: efficient tree-based sorting, "
        "Bubblesort: sliding window pairwise comparisons.",
    )
    top_k: int = Field(
        default=10,
        gt=0,
        description="Maximum number of documents to rerank and return. Higher values increase computation time.",
    )
    device: str | None = Field(
        default=None, description="Device for model execution. Auto-detected if None. Examples: 'cuda', 'cuda:0', 'cpu'"
    )
    model_kwargs: dict = Field(default_factory=dict, description="Additional model initialization parameters.")
    tokenizer_kwargs: dict = Field(default_factory=dict, description="Tokenizer configuration.")

    @model_validator(mode="after")
    def ensure_device_is_none(self):
        """Ensure device is always None to avoid ComponentDevice issues."""
        if "device" in self.model_kwargs:
            self.model_kwargs["device"] = None
        return self


class PairwiseRankingConfig(BaseModel):
    """Configuration for the pairwise ranking pipeline.

    This configuration includes all settings needed to run the pairwise ranking pipeline,
    including dataset, embedding, retrieval, and evaluation configurations.
    """

    # Dataset configuration
    dataset: DatasetConfig = Field(..., description="Dataset configuration")

    # LLM configuration
    llm: LLMConfig = Field(..., description="LLM ranker configuration")

    # Embedding configuration
    embedding: EmbeddingConfig = Field(..., description="Embedding configuration")

    # Milvus configuration
    milvus: MilvusConfig = Field(..., description="Milvus vector database configuration")

    # Retrieval configuration
    retrieval: RetrievalConfig = Field(..., description="Retrieval configuration")

    # Evaluation configuration
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation metrics and settings"
    )
