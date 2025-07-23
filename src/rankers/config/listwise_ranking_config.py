from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rankers.config.base_config import DatasetConfig, EmbeddingConfig, EvaluationConfig, MilvusConfig, RetrievalConfig


class ListwiseOpenAIConfig(BaseModel):
    """Configuration for OpenAI API settings."""

    api_keys: list[str] | None = Field(
        default=None, description="List of OpenAI API keys to use for LLMs. Required for OpenAI-based rankers."
    )
    key_start_id: int | None = Field(
        default=None, description="Start index for OpenAI API keys. Useful for key rotation."
    )
    proxy: str | None = Field(default=None, description="Proxy for OpenAI API requests.")
    api_type: str | None = Field(default=None, description="API type for OpenAI API requests.")
    api_base: str | None = Field(default=None, description="API base URL for OpenAI API requests.")
    api_version: str | None = Field(default=None, description="API version for OpenAI API requests.")


class ListwiseHFConfig(BaseModel):
    """Configuration for the LLM ranker component."""

    model_path: str = Field(..., description="Hugging Face model name/path for the LLM ranker.")
    ranker_type: Literal["zephyr", "vicuna", "rank_gpt"] = Field(
        default="zephyr", description="Type of reranker to use."
    )
    context_size: int = Field(
        default=4096, gt=0, description="Maximum context size in tokens that the model can handle."
    )
    num_few_shot_examples: int = Field(
        default=0, ge=0, description="Number of few-shot examples to include in prompts."
    )
    top_k: int = Field(default=10, gt=0, description="Maximum number of documents to rerank and return.")
    device: str | None = Field(
        default=None, description="Device for model execution. Auto-detected if None. Examples: 'cuda', 'cuda:0', 'cpu'"
    )
    num_gpus: int = Field(default=1, gt=0, description="Number of GPUs to use.")
    variable_passages: bool = Field(
        default=False, description="Whether to allow variable number of passages per request."
    )
    sliding_window_size: int = Field(
        default=20, gt=0, description="Size of sliding window for processing long documents."
    )
    sliding_window_step: int = Field(default=10, gt=0, description="Step size for sliding window movement.")
    system_message: str = Field(
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        description="System message to configure LLM behavior.",
    )
    openai: ListwiseOpenAIConfig | None = Field(
        default=None, description="OpenAI API configuration. Required for OpenAI-based rankers."
    )

    @model_validator(mode="after")
    def validate_openai_config(self) -> "ListwiseHFConfig":
        """Validate OpenAI configuration.

        Raises:
            ValueError: If ranker_type is 'rank_gpt' and openai is None.
        """
        if self.ranker_type == "rank_gpt" and self.openai is None:
            msg = "OpenAI configuration is required for rank_gpt ranker type"
            raise ValueError(msg)
        return self


class ListwiseRankingConfig(BaseModel):
    """Complete configuration for the listwise ranking pipeline."""

    dataset: DatasetConfig = Field(..., description="Dataset configuration.")
    ranker: ListwiseHFConfig = Field(..., description="LLM ranker configuration.")
    embedding: EmbeddingConfig = Field(..., description="Embedding model configuration.")
    milvus: MilvusConfig = Field(..., description="Milvus vector database configuration.")
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig, description="Document retrieval configuration.")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration.")
