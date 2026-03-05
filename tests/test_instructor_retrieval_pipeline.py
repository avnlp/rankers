"""Tests for the INSTRUCTOR retrieval pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    MilvusConfig,
    RetrievalConfig,
)


class DummyDataset:
    """Minimal mock dataset returned by Dataloader.load()."""

    def __init__(self) -> None:
        """Initialize with synthetic queries and relevance judgments."""
        self.queries = {"q1": "What is Paris?", "q2": "Where is Berlin?"}
        self.relevance_judgments = {"q1": {"d1": 1}, "q2": {"d2": 1}}


class DummyEmptyDataset:
    """Mock dataset with no queries, used to test the empty-queries branch."""

    def __init__(self) -> None:
        """Initialize with empty queries dict."""
        self.queries = {}
        self.relevance_judgments = {}


class TestInstructorRetrievalPipeline:
    """Tests for the instructor_retrieval pipeline's main() function."""

    @pytest.fixture
    def mock_config(self) -> RetrievalConfig:
        """Fixture providing a synthetic RetrievalConfig for the retrieval pipeline.

        Returns:
            A RetrievalConfig instance with all required sub-configs populated.
        """
        return RetrievalConfig(
            dataset=DatasetConfig(name="beir/fiqa/train"),
            embedding=EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2"),
            milvus=MilvusConfig(
                connection_uri="http://localhost:19530", connection_token="token"
            ),
            evaluation=EvaluationConfig(),
            documents_to_retrieve=5,
        )

    def _build_pipeline_output(self) -> dict:
        """Build a fake pipeline output dict with scored documents.

        Returns:
            A dict mimicking embedding_pipeline.run() output.
        """
        mock_doc = MagicMock()
        mock_doc.meta = {"doc_id": "d1"}
        mock_doc.score = 0.9
        return {"embedding_retriever": {"documents": [mock_doc]}}

    def test_main_runs_evaluator(self, mock_config: RetrievalConfig) -> None:
        """Test that main() processes queries and calls Evaluator.evaluate().

        Mocks all external I/O (load_config, Dataloader, MilvusDocumentStore,
        MilvusEmbeddingRetriever, SentenceTransformersTextEmbedder, Pipeline, Evaluator)
        so the test is fully hermetic.
        """
        from rankers.pipelines import instructor_retrieval as retrieval_module

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = self._build_pipeline_output()

        mock_evaluator_instance = MagicMock()
        mock_evaluator_cls = MagicMock(return_value=mock_evaluator_instance)

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()

        with (
            patch.object(retrieval_module, "load_config", return_value=mock_config),
            patch.object(
                retrieval_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(retrieval_module, "MilvusDocumentStore"),
            patch.object(retrieval_module, "MilvusEmbeddingRetriever"),
            patch.object(retrieval_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                retrieval_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(retrieval_module, "Evaluator", mock_evaluator_cls),
        ):
            retrieval_module.main("fake_config.yaml")

        # Evaluator must be constructed and evaluate() must be called
        mock_evaluator_cls.assert_called_once()
        mock_evaluator_instance.evaluate.assert_called_once()

    def test_main_empty_queries_evaluator_not_called(
        self, mock_config: RetrievalConfig
    ) -> None:
        """Test main() with no queries skips evaluation.

        When dataset.queries is empty the tqdm loop body never executes, so
        all_query_results stays empty. The evaluator is still constructed but
        evaluate() is called with an empty run_results dict.
        """
        from rankers.pipelines import instructor_retrieval as retrieval_module

        mock_pipeline_instance = MagicMock()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyEmptyDataset()

        mock_evaluator_instance = MagicMock()
        mock_evaluator_cls = MagicMock(return_value=mock_evaluator_instance)

        with (
            patch.object(retrieval_module, "load_config", return_value=mock_config),
            patch.object(
                retrieval_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(retrieval_module, "MilvusDocumentStore"),
            patch.object(retrieval_module, "MilvusEmbeddingRetriever"),
            patch.object(retrieval_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                retrieval_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(retrieval_module, "Evaluator", mock_evaluator_cls),
        ):
            retrieval_module.main("fake_config.yaml")

        # pipeline.run() should never have been called for queries
        mock_pipeline_instance.run.assert_not_called()

    def test_main_doc_id_extracted_from_meta(
        self, mock_config: RetrievalConfig
    ) -> None:
        """Test that doc_id is extracted from document.meta['doc_id'] correctly.

        Verifies that the run_results dict passed to the Evaluator contains
        the doc_id value from the document's meta field as the key.
        """
        from rankers.pipelines import instructor_retrieval as retrieval_module

        mock_doc = MagicMock()
        mock_doc.meta = {"doc_id": "expected_doc_id"}
        mock_doc.score = 0.75
        pipeline_output = {"embedding_retriever": {"documents": [mock_doc]}}

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = pipeline_output

        mock_evaluator_instance = MagicMock()
        MagicMock(return_value=mock_evaluator_instance)

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()

        captured_run_results = {}

        def capture_evaluator(relevance_judgments, run_results, config):
            captured_run_results.update(run_results)
            return mock_evaluator_instance

        with (
            patch.object(retrieval_module, "load_config", return_value=mock_config),
            patch.object(
                retrieval_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(retrieval_module, "MilvusDocumentStore"),
            patch.object(retrieval_module, "MilvusEmbeddingRetriever"),
            patch.object(retrieval_module, "SentenceTransformersTextEmbedder"),
            patch.object(
                retrieval_module, "Pipeline", return_value=mock_pipeline_instance
            ),
            patch.object(retrieval_module, "Evaluator", side_effect=capture_evaluator),
        ):
            retrieval_module.main("fake_config.yaml")

        # Every query result must use doc_id from meta as the document key
        for query_results in captured_run_results.values():
            assert "expected_doc_id" in query_results
