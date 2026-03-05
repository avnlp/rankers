"""Tests for the indexing pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from rankers.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    IndexingConfig,
    MilvusConfig,
)


class DummyDataset:
    """Minimal mock dataset returned by Dataloader.load()."""

    def __init__(self) -> None:
        """Initialize with a small synthetic corpus."""
        self.corpus = {
            "doc1": {"text": "First document content"},
            "doc2": {"text": "Second document content"},
        }


class TestIndexingPipeline:
    """Tests for the indexing pipeline's main() function."""

    @pytest.fixture
    def mock_config(self) -> IndexingConfig:
        """Fixture providing a synthetic IndexingConfig for the indexing pipeline.

        Returns:
            An IndexingConfig instance with all required sub-configs populated.
        """
        return IndexingConfig(
            dataset=DatasetConfig(name="beir/fiqa/train"),
            embedding=EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2"),
            milvus=MilvusConfig(
                connection_uri="http://localhost:19530", connection_token="token"
            ),
        )

    def test_main_runs_pipeline(self, mock_config: IndexingConfig) -> None:
        """Test that main() loads config, builds the pipeline, and calls pipeline.run().

        Mocks all external dependencies (load_config, Dataloader, MilvusDocumentStore,
        SentenceTransformersDocumentEmbedder, Pipeline) to verify integration flow
        without touching disk, network, or real models.
        """
        from rankers.pipelines import indexing as indexing_module

        mock_pipeline = MagicMock()
        mock_embedder = MagicMock()
        mock_document_store = MagicMock()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()

        with (
            patch.object(indexing_module, "load_config", return_value=mock_config),
            patch.object(
                indexing_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(
                indexing_module, "MilvusDocumentStore", return_value=mock_document_store
            ),
            patch.object(
                indexing_module,
                "SentenceTransformersDocumentEmbedder",
                return_value=mock_embedder,
            ),
            patch.object(indexing_module, "Pipeline", return_value=mock_pipeline),
        ):
            indexing_module.main("fake_config.yaml")

        # Verify the pipeline ran with embedder documents
        mock_pipeline.run.assert_called_once()
        run_call_args = mock_pipeline.run.call_args[0][0]
        assert "embedder" in run_call_args
        assert "documents" in run_call_args["embedder"]

    def test_main_warm_up_called(self, mock_config: IndexingConfig) -> None:
        """Test that the embedder's warm_up() is called before running the pipeline."""
        from rankers.pipelines import indexing as indexing_module

        mock_pipeline = MagicMock()
        mock_embedder = MagicMock()
        mock_document_store = MagicMock()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()

        with (
            patch.object(indexing_module, "load_config", return_value=mock_config),
            patch.object(
                indexing_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(
                indexing_module, "MilvusDocumentStore", return_value=mock_document_store
            ),
            patch.object(
                indexing_module,
                "SentenceTransformersDocumentEmbedder",
                return_value=mock_embedder,
            ),
            patch.object(indexing_module, "Pipeline", return_value=mock_pipeline),
        ):
            indexing_module.main("fake_config.yaml")

        mock_embedder.warm_up.assert_called_once()

    def test_main_pipeline_components_connected(
        self, mock_config: IndexingConfig
    ) -> None:
        """Test that pipeline components are added and connected correctly.

        Verifies that add_component is called twice (embedder + writer) and
        connect is called to link them.
        """
        from rankers.pipelines import indexing as indexing_module

        mock_pipeline = MagicMock()
        mock_embedder = MagicMock()
        mock_document_store = MagicMock()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.load.return_value = DummyDataset()

        with (
            patch.object(indexing_module, "load_config", return_value=mock_config),
            patch.object(
                indexing_module, "Dataloader", return_value=mock_dataloader_instance
            ),
            patch.object(
                indexing_module, "MilvusDocumentStore", return_value=mock_document_store
            ),
            patch.object(
                indexing_module,
                "SentenceTransformersDocumentEmbedder",
                return_value=mock_embedder,
            ),
            patch.object(indexing_module, "Pipeline", return_value=mock_pipeline),
        ):
            indexing_module.main("fake_config.yaml")

        # add_component called for 'embedder' and 'writer'
        assert mock_pipeline.add_component.call_count == 2
        # connect called to link embedder → writer
        mock_pipeline.connect.assert_called_once()
