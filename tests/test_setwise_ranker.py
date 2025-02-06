import copy
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from haystack import Document

from rankers.setwise.setwise_ranker import SetwiseLLMRanker, SetwiseRankingOutput


class DummyStructuredGeneration:
    """A dummy implementation of StructuredGeneration for testing purposes."""

    def __init__(self, *args, **kwargs):
        """Initialize the dummy model without loading any real model."""
        pass

    def generate(
        self, output_format: str, user_prompt: Optional[str] = None, system_prompt: Optional[str] = None
    ) -> Mock:
        """Generate a dummy output for testing.

        Args:
            output_format: The format of the output.
            user_prompt: Optional user prompt.
            system_prompt: Optional system prompt.

        Returns:
            A mock object with a selected_passage attribute.
        """
        dummy = Mock()
        dummy.selected_passage = "Passage A"
        return dummy


@pytest.fixture(autouse=True)
def patch_structured_generation() -> None:
    """Patch the StructuredGeneration class to use the dummy version for testing.

    This fixture ensures that during tests, the StructuredGeneration class is mocked
    to prevent loading any real model.
    """
    with patch("rankers.setwise.setwise_ranker.StructuredGeneration", DummyStructuredGeneration):
        yield


class TestSetwiseLLMRanker:
    """Tests for the SetwiseLLMRanker class."""
    @pytest.fixture
    def test_documents(self) -> list[Document]:
        """Fixture providing a list of test documents.

        Returns:
            A list of Document objects with sample content.
        """
        return [
            Document(id="1", content="Paris"),
            Document(id="2", content="Berlin"),
            Document(id="3", content="Madrid"),
        ]

    @pytest.fixture
    def test_query(self) -> str:
        """Fixture providing a test query.

        Returns:
            A sample query string.
        """
        return "Which city is the capital of France?"

    @pytest.fixture
    def setwise_ranker(self) -> SetwiseLLMRanker:
        """Fixture providing an instance of SetwiseLLMRanker for testing.

        This fixture initializes the ranker with mocked dependencies to avoid
        loading real models during tests.

        Returns:
            An instance of SetwiseLLMRanker.
        """
        ranker = SetwiseLLMRanker(
            model_name="test-model",
            model_class=Mock(),
            tokenizer_class=Mock(),
            method="heapsort",
            num_child=3,
            top_k=10,
        )
        ranker._is_warmed_up = True
        ranker._structured_generation_model = Mock()
        ranker._structured_generation_model.generate.return_value = SetwiseRankingOutput(selected_passage="(A1)")
        yield ranker

    def test_init(self) -> None:
        """Test the initialization of SetwiseLLMRanker.

        Verify that all parameters are correctly set during initialization.
        """
        ranker = SetwiseLLMRanker(
            model_name="test-model",
            device="cpu",
            model_kwargs={"test": "kwarg"},
            tokenizer_kwargs={"tokenizer": "kwarg"},
            model_class=Mock(),
            tokenizer_class=Mock(),
            method="heapsort",
            num_permutation=1,
            num_child=3,
            top_k=10,
        )
        assert ranker.model_name == "test-model"
        assert ranker.device == "cpu"
        assert ranker.model_kwargs == {"test": "kwarg"}
        assert ranker.tokenizer_kwargs == {"tokenizer": "kwarg"}
        assert ranker.method == "heapsort"
        assert ranker.num_permutation == 1
        assert ranker.num_child == 3
        assert ranker.top_k == 10
        assert ranker._structured_generation_model is None
        assert not ranker._is_warmed_up

    def test_warm_up(self, setwise_ranker: SetwiseLLMRanker) -> None:
        """Test the _warm_up method.

        Ensure that the warm_up method correctly initializes the model and sets
        the _is_warmed_up flag.
        """
        setwise_ranker._structured_generation_model = None
        setwise_ranker._is_warmed_up = False
        setwise_ranker._warm_up()
        assert setwise_ranker._is_warmed_up
        assert setwise_ranker._structured_generation_model is not None

    def test_compare(self, setwise_ranker: SetwiseLLMRanker, test_query: str, test_documents: list[Document]) -> None:
        """Test the compare method.

        Verify that the compare method calls the generate method on the model and
        returns the selected passage.
        """
        setwise_ranker._structured_generation_model.generate.return_value = SetwiseRankingOutput(
            selected_passage="(A6)"
        )
        result = setwise_ranker.compare(test_query, test_documents)
        assert result == "(A6)"
        setwise_ranker._structured_generation_model.generate.assert_called_once()

    def test_heapify(self, setwise_ranker: SetwiseLLMRanker, test_query: str, test_documents: list[Document]) -> None:
        """Test the heapify method.

        Ensure that the heapify method correctly processes the documents array.
        """
        n = len(test_documents)
        arr = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect=["(A0)", "(A1)"])
        setwise_ranker.heapify(arr, n, 0, test_query)
        assert len(arr) == n

    def test_heapsort(self, setwise_ranker: SetwiseLLMRanker, test_query: str, test_documents: list[Document]) -> None:
        """Test the heapsort method.

        Verify that the heapsort method correctly sorts the documents.
        """
        n = len(test_documents)
        arr = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect="(A0)")
        setwise_ranker.heapsort(arr, test_query, 2)
        assert len(arr) == n

    def test_bubblesort(
        self, setwise_ranker: SetwiseLLMRanker, test_query: str, test_documents: list[Document]
    ) -> None:
        """Test the bubblesort method.

        Ensure that the bubblesort method correctly processes the documents array.
        """
        n = len(test_documents)
        arr = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect="(A0)")
        setwise_ranker.bubblesort(arr, test_query)
        assert len(arr) == n

    def test_rerank(self, setwise_ranker: SetwiseLLMRanker, test_query: str, test_documents: list[Document]) -> None:
        """Test the rerank method.

        Verify that the rerank method correctly reranks the documents.
        """
        n = len(test_documents)
        ranking = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect="(A0)")
        reranked = setwise_ranker.rerank(test_query, ranking)
        assert len(reranked) == n

    def test_run(self, setwise_ranker: SetwiseLLMRanker, test_query: str, test_documents: list[Document]) -> None:
        """Test the run method.

        Ensure that the run method correctly processes the documents and query.
        """
        expected_documents = test_documents[::-1]
        setwise_ranker.rerank = Mock(return_value=expected_documents)
        result = setwise_ranker.run(documents=test_documents, query=test_query)
        assert "documents" in result
        assert result["documents"] == expected_documents

    def test_run_with_parameters(self, test_query: str, test_documents: list[Document]) -> None:
        """Test the run method with overridden parameters.

        Verify that the run method correctly uses overridden parameters.
        """
        ranker = SetwiseLLMRanker(
            model_name="test-model",
            model_class=Mock(),
            tokenizer_class=Mock(),
            method="bubblesort",
            num_child=2,
            top_k=5,
        )
        ranker._is_warmed_up = True
        ranker._structured_generation_model = Mock()
        expected_documents = test_documents[::-1]
        with patch.object(ranker, "rerank", return_value=expected_documents):
            result = ranker.run(
                documents=test_documents,
                query=test_query,
                top_k=3,
                num_child=4,
                method="heapsort",
            )
        assert ranker.top_k == 3
        assert ranker.num_child == 4
        assert ranker.method == "heapsort"
        assert result["documents"] == expected_documents

    def test_error_handling_invalid_method(self, test_documents: list[Document]) -> None:
        """Test error handling for invalid method.

        Verify that using an invalid method raises a ValueError.
        """
        dummy_documents = [Document(id="dummy", content="dummy")]
        ranker = SetwiseLLMRanker(
            model_name="test-model",
            model_class=Mock(),
            tokenizer_class=Mock(),
            method="invalid",
            num_child=3,
            top_k=10,
        )
        ranker._is_warmed_up = True
        ranker._structured_generation_model = Mock()
        with pytest.raises(ValueError):
            ranker.run(documents=dummy_documents, query="test")

    def test_error_handling_num_child_zero(self, test_documents: list[Document]) -> None:
        """Test error handling for num_child <= 0.

        Verify that using num_child <= 0 raises a ValueError.
        """
        dummy_documents = [Document(id="dummy", content="dummy")]
        ranker = SetwiseLLMRanker(
            model_name="test-model",
            model_class=Mock(),
            tokenizer_class=Mock(),
            method="heapsort",
            num_child=0,
            top_k=10,
        )
        ranker._is_warmed_up = True
        ranker._structured_generation_model = Mock()
        with pytest.raises(ValueError):
            ranker.run(documents=dummy_documents, query="test")

    def test_error_handling_top_k_zero(self, test_documents: list[Document]) -> None:
        """Test error handling for top_k <= 0.

        Verify that using top_k <= 0 raises a ValueError.
        """
        dummy_documents = [Document(id="dummy", content="dummy")]
        ranker = SetwiseLLMRanker(
            model_name="test-model",
            model_class=Mock(),
            tokenizer_class=Mock(),
            method="heapsort",
            num_child=3,
            top_k=0,
        )
        ranker._is_warmed_up = True
        ranker._structured_generation_model = Mock()
        with pytest.raises(ValueError):
            ranker.run(documents=dummy_documents, query="test")
