"""Tests for the listwise ranking pipeline."""

from unittest.mock import MagicMock, patch

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    MilvusConfig,
    RetrievalConfig,
)
from rankers.config.listwise_ranking_config import (
    ListwiseHFConfig,
    ListwiseOpenAIConfig,
    ListwiseRankingConfig,
)


class DummyDataset:
    """Minimal mock dataset returned by Dataloader.load()."""

    def __init__(self) -> None:
        """Initialize with synthetic queries and relevance judgments."""
        self.queries = {"q1": "What is Paris?"}
        self.relevance_judgments = {"q1": {"d1": 1}}


def _make_config(with_openai: bool = False) -> ListwiseRankingConfig:
    """Build a synthetic ListwiseRankingConfig for testing.

    Args:
        with_openai: If True, includes a ListwiseOpenAIConfig in the ranker config.

    Returns:
        A fully-populated ListwiseRankingConfig instance.
    """
    dataset = DatasetConfig(name="beir/fiqa/train")
    embedding = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    milvus = MilvusConfig(
        connection_uri="http://localhost:19530", connection_token="token"
    )
    retrieval = RetrievalConfig(
        dataset=dataset, embedding=embedding, milvus=milvus, documents_to_retrieve=5
    )

    openai_cfg = ListwiseOpenAIConfig(api_keys=["sk-test"]) if with_openai else None
    ranker = ListwiseHFConfig(
        model_path="test-model", ranker_type="zephyr", openai=openai_cfg
    )

    return ListwiseRankingConfig(
        dataset=dataset,
        ranker=ranker,
        embedding=embedding,
        milvus=milvus,
        retrieval=retrieval,
        evaluation=EvaluationConfig(),
    )


class TestListwiseRankingPipeline:
    """Tests for the listwise_ranking pipeline's main() function."""

    def _build_pipeline_output(self) -> dict:
        """Build a minimal pipeline output dict with a scored document."""
        mock_doc = MagicMock()
        mock_doc.meta = {"doc_id": "d1"}
        mock_doc.score = 0.9
        return {"ranker": {"documents": [mock_doc]}}

    def test_main_with_openai_config_passes_openai_params(self) -> None:
        """Test main() includes OpenAI params in ranker_kwargs when openai cfg present.

        Captures kwargs passed to ListwiseLLMRanker and verifies that
        'openai_api_keys' and related keys are included when config.ranker.openai
        is set.
        """
        from rankers.pipelines import listwise_ranking as listwise_module

        mock_config = _make_config(with_openai=True)
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = self._build_pipeline_output()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()
        mock_evaluator_instance = MagicMock()

        captured_ranker_kwargs = {}

        def capture_ranker(**kwargs):
            captured_ranker_kwargs.update(kwargs)
            return MagicMock()

        with (
            patch.object(listwise_module, "load_config", return_value=mock_config),
            patch.object(
                listwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(listwise_module, "MilvusDocumentStore"),
            patch.object(listwise_module, "MilvusEmbeddingRetriever"),
            patch.object(listwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                listwise_module, "ListwiseLLMRanker", side_effect=capture_ranker
            ),
            patch.object(
                listwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                listwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            listwise_module.main("fake_config.yaml")

        assert "openai_api_keys" in captured_ranker_kwargs
        assert captured_ranker_kwargs["openai_api_keys"] == ["sk-test"]

    def test_main_without_openai_config_excludes_openai_params(self) -> None:
        """Test main() skips OpenAI params in ranker_kwargs if openai cfg is None.

        When config.ranker.openai is None the conditional block is skipped, so
        'openai_api_keys' must not appear in the kwargs passed to ListwiseLLMRanker.
        """
        from rankers.pipelines import listwise_ranking as listwise_module

        mock_config = _make_config(with_openai=False)
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = self._build_pipeline_output()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()
        mock_evaluator_instance = MagicMock()

        captured_ranker_kwargs = {}

        def capture_ranker(**kwargs):
            captured_ranker_kwargs.update(kwargs)
            return MagicMock()

        with (
            patch.object(listwise_module, "load_config", return_value=mock_config),
            patch.object(
                listwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(listwise_module, "MilvusDocumentStore"),
            patch.object(listwise_module, "MilvusEmbeddingRetriever"),
            patch.object(listwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                listwise_module, "ListwiseLLMRanker", side_effect=capture_ranker
            ),
            patch.object(
                listwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                listwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            listwise_module.main("fake_config.yaml")

        assert "openai_api_keys" not in captured_ranker_kwargs

    def test_main_pipeline_executed(self) -> None:
        """Test main() calls pipeline.run() for each query and evaluator.evaluate()."""
        from rankers.pipelines import listwise_ranking as listwise_module

        mock_config = _make_config(with_openai=False)
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = self._build_pipeline_output()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()
        mock_evaluator_instance = MagicMock()

        with (
            patch.object(listwise_module, "load_config", return_value=mock_config),
            patch.object(
                listwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(listwise_module, "MilvusDocumentStore"),
            patch.object(listwise_module, "MilvusEmbeddingRetriever"),
            patch.object(listwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(listwise_module, "ListwiseLLMRanker"),
            patch.object(
                listwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                listwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            listwise_module.main("fake_config.yaml")

        # One query in DummyDataset → pipeline.run called once
        assert mock_pipeline_instance.run.call_count == 1
        mock_evaluator_instance.evaluate.assert_called_once()
