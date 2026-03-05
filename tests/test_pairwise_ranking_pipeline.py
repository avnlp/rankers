"""Tests for the pairwise ranking pipeline."""

from unittest.mock import MagicMock, patch

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    MilvusConfig,
    RetrievalConfig,
)
from rankers.config.pairwise_ranking_config import (
    LLMConfig,
    PairwiseMethod,
    PairwiseRankingConfig,
)


class DummyDataset:
    """Minimal mock dataset returned by Dataloader.load()."""

    def __init__(self) -> None:
        """Initialize with synthetic queries and relevance judgments."""
        self.queries = {"q1": "What is Paris?"}
        self.relevance_judgments = {"q1": {"d1": 1}}


def _make_config() -> PairwiseRankingConfig:
    """Build a synthetic PairwiseRankingConfig for testing.

    Returns:
        A fully-populated PairwiseRankingConfig instance.
    """
    dataset = DatasetConfig(name="beir/fiqa/train")
    embedding = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    milvus = MilvusConfig(
        connection_uri="http://localhost:19530", connection_token="token"
    )
    retrieval = RetrievalConfig(
        dataset=dataset, embedding=embedding, milvus=milvus, documents_to_retrieve=5
    )
    llm = LLMConfig(model_name="test-model", method=PairwiseMethod.HEAPSORT, top_k=3)

    return PairwiseRankingConfig(
        dataset=dataset,
        llm=llm,
        embedding=embedding,
        milvus=milvus,
        retrieval=retrieval,
        evaluation=EvaluationConfig(),
    )


class TestPairwiseRankingPipeline:
    """Tests for the pairwise_ranking pipeline's main() function."""

    def _build_pipeline_output(self) -> dict:
        """Build a minimal pipeline output dict with a scored document."""
        mock_doc = MagicMock()
        mock_doc.meta = {"doc_id": "d1"}
        mock_doc.score = 0.9
        return {"ranker": {"documents": [mock_doc]}}

    def test_main_pipeline_built_and_run(self) -> None:
        """Test main() builds pipeline, connects components, and calls pipeline.run().

        Mocks all external dependencies and verifies that pipeline.run() is invoked
        with the correct query payload for each query in the dataset.
        """
        from rankers.pipelines import pairwise_ranking as pairwise_module

        mock_config = _make_config()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = self._build_pipeline_output()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()
        mock_evaluator_instance = MagicMock()

        with (
            patch.object(pairwise_module, "load_config", return_value=mock_config),
            patch.object(
                pairwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(pairwise_module, "MilvusDocumentStore"),
            patch.object(pairwise_module, "MilvusEmbeddingRetriever"),
            patch.object(pairwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(pairwise_module, "PairwiseLLMRanker"),
            patch.object(
                pairwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                pairwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            pairwise_module.main("fake_config.yaml")

        # One query in DummyDataset → pipeline.run called once
        mock_pipeline_instance.run.assert_called_once()
        mock_evaluator_instance.evaluate.assert_called_once()

    def test_main_ranker_created_with_correct_params(self) -> None:
        """Test PairwiseLLMRanker constructed with method and top_k from config."""
        from rankers.pipelines import pairwise_ranking as pairwise_module

        mock_config = _make_config()
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
            patch.object(pairwise_module, "load_config", return_value=mock_config),
            patch.object(
                pairwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(pairwise_module, "MilvusDocumentStore"),
            patch.object(pairwise_module, "MilvusEmbeddingRetriever"),
            patch.object(pairwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                pairwise_module, "PairwiseLLMRanker", side_effect=capture_ranker
            ),
            patch.object(
                pairwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                pairwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            pairwise_module.main("fake_config.yaml")

        assert captured_ranker_kwargs["model_name"] == "test-model"
        assert captured_ranker_kwargs["method"] == PairwiseMethod.HEAPSORT
        assert captured_ranker_kwargs["top_k"] == 3
