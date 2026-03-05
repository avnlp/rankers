"""Tests for the setwise ranking pipeline."""

from unittest.mock import MagicMock, patch

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    MilvusConfig,
    RetrievalConfig,
)
from rankers.config.setwise_ranking_config import (
    SetwiseLLMConfig,
    SetwiseMethod,
    SetwiseRankingConfig,
)


class DummyDataset:
    """Minimal mock dataset returned by Dataloader.load()."""

    def __init__(self) -> None:
        """Initialize with synthetic queries and relevance judgments."""
        self.queries = {"q1": "What is Paris?"}
        self.relevance_judgments = {"q1": {"d1": 1}}


def _make_config() -> SetwiseRankingConfig:
    """Build a synthetic SetwiseRankingConfig for testing.

    Returns:
        A fully-populated SetwiseRankingConfig instance.
    """
    dataset = DatasetConfig(name="beir/fiqa/train")
    embedding = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    milvus = MilvusConfig(
        connection_uri="http://localhost:19530", connection_token="token"
    )
    retrieval = RetrievalConfig(
        dataset=dataset, embedding=embedding, milvus=milvus, documents_to_retrieve=5
    )
    llm = SetwiseLLMConfig(
        model_name="test-model",
        method=SetwiseMethod.HEAPSORT,
        top_k=5,
        num_permutation=2,
        num_child=3,
    )

    return SetwiseRankingConfig(
        dataset=dataset,
        llm=llm,
        embedding=embedding,
        milvus=milvus,
        retrieval=retrieval,
        evaluation=EvaluationConfig(),
    )


class TestSetwiseRankingPipeline:
    """Tests for the setwise_ranking pipeline's main() function."""

    def _build_pipeline_output(self) -> dict:
        """Build a minimal pipeline output dict with a scored document."""
        mock_doc = MagicMock()
        mock_doc.meta = {"doc_id": "d1"}
        mock_doc.score = 0.9
        return {"ranker": {"documents": [mock_doc]}}

    def test_main_pipeline_built_with_correct_connections(self) -> None:
        """Test that main() builds the pipeline with the correct component connections.

        Verifies that:
        - Pipeline.add_component is called for text_embedder, retriever, and ranker.
        - Pipeline.connect links text_embedder.embedding → retriever.query_embedding
          and retriever.documents → ranker.documents.
        """
        from rankers.pipelines import setwise_ranking as setwise_module

        mock_config = _make_config()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = self._build_pipeline_output()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()
        mock_evaluator_instance = MagicMock()

        with (
            patch.object(setwise_module, "load_config", return_value=mock_config),
            patch.object(
                setwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(setwise_module, "MilvusDocumentStore"),
            patch.object(setwise_module, "MilvusEmbeddingRetriever"),
            patch.object(setwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(setwise_module, "SetwiseLLMRanker"),
            patch.object(
                setwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                setwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            setwise_module.main("fake_config.yaml")

        # Three components: text_embedder, retriever, ranker
        assert mock_pipeline_instance.add_component.call_count == 3
        # Two connections: text_embedder.embedding → retriever.query_embedding,
        #                  retriever.documents → ranker.documents
        assert mock_pipeline_instance.connect.call_count == 2

    def test_main_ranker_created_with_num_permutation_and_num_child(self) -> None:
        """Test SetwiseLLMRanker constructed with num_permutation, num_child cfg."""
        from rankers.pipelines import setwise_ranking as setwise_module

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
            patch.object(setwise_module, "load_config", return_value=mock_config),
            patch.object(
                setwise_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(setwise_module, "MilvusDocumentStore"),
            patch.object(setwise_module, "MilvusEmbeddingRetriever"),
            patch.object(setwise_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                setwise_module, "SetwiseLLMRanker", side_effect=capture_ranker
            ),
            patch.object(
                setwise_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(
                setwise_module, "Evaluator", return_value=mock_evaluator_instance
            ),
        ):
            setwise_module.main("fake_config.yaml")

        assert captured_ranker_kwargs["model_name"] == "test-model"
        assert captured_ranker_kwargs["num_permutation"] == 2
        assert captured_ranker_kwargs["num_child"] == 3
        assert captured_ranker_kwargs["top_k"] == 5
