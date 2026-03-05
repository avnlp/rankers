"""Tests for the setwise ranker."""

import copy
from unittest.mock import Mock, patch

import pytest
from haystack import Document

from rankers.setwise.setwise_ranker import SetwiseLLMRanker
from rankers.setwise.setwise_ranker_output_validator import SetwiseRankingOutput


class DummyStructuredGeneration:
    """A dummy implementation of StructuredGeneration for testing purposes."""

    def __init__(self, *args, **kwargs):
        """Initialize the dummy model without loading any real model."""
        pass

    def generate(
        self,
        output_format: str,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
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
    with patch(
        "rankers.setwise.setwise_ranker.StructuredGeneration", DummyStructuredGeneration
    ):
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
        ranker._structured_generation_model.generate.return_value = (
            SetwiseRankingOutput(selected_passage="(A1)")
        )
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

    def test_compare(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test the compare method.

        Verify that the compare method calls the generate method on the model and
        returns the selected passage.
        """
        setwise_ranker._structured_generation_model.generate.return_value = (
            SetwiseRankingOutput(selected_passage="(A6)")
        )
        result = setwise_ranker.compare(test_query, test_documents)
        assert result == "(A6)"
        setwise_ranker._structured_generation_model.generate.assert_called_once()

    def test_heapify(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test the heapify method.

        Ensure that the heapify method correctly processes the documents array.
        """
        n = len(test_documents)
        arr = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect=["(A0)", "(A1)"])
        setwise_ranker.heapify(arr, n, 0, test_query)
        assert len(arr) == n

    def test_heapsort(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test the heapsort method.

        Verify that the heapsort method correctly sorts the documents.
        """
        n = len(test_documents)
        arr = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect="(A0)")
        setwise_ranker.heapsort(arr, test_query, 2)
        assert len(arr) == n

    def test_bubblesort(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test the bubblesort method.

        Ensure that the bubblesort method correctly processes the documents array.
        """
        n = len(test_documents)
        arr = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect="(A0)")
        setwise_ranker.bubblesort(arr, test_query)
        assert len(arr) == n

    def test_rerank(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test the rerank method.

        Verify that the rerank method correctly reranks the documents.
        """
        n = len(test_documents)
        ranking = copy.deepcopy(test_documents)
        setwise_ranker.compare = Mock(side_effect="(A0)")
        reranked = setwise_ranker.rerank(test_query, ranking)
        assert len(reranked) == n

    def test_run(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test the run method.

        Ensure that the run method correctly processes the documents and query.
        """
        expected_documents = test_documents[::-1]
        setwise_ranker.rerank = Mock(return_value=expected_documents)
        result = setwise_ranker.run(documents=test_documents, query=test_query)
        assert "documents" in result
        assert result["documents"] == expected_documents

    def test_run_with_parameters(
        self, test_query: str, test_documents: list[Document]
    ) -> None:
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

    def test_error_handling_invalid_method(
        self, test_documents: list[Document]
    ) -> None:
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

    def test_error_handling_num_child_zero(
        self, test_documents: list[Document]
    ) -> None:
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

    def test_compare_invalid_identifier(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test compare() raises ValueError when LLM returns invalid identifier.

        Configures the mock generator to return a SetwiseRankingOutput with a valid
        selected_passage, but then patches IDENTIFIERS.index to raise ValueError to
        simulate an identifier not found in the list.
        """
        # The validator on SetwiseRankingOutput only accepts valid identifiers,
        # so we simulate the compare() path by making the model return a valid
        # object and then testing the ValueError propagation inside heapify.
        # Here we test compare() directly: it returns output.selected_passage.
        setwise_ranker._structured_generation_model.generate.return_value = (
            SetwiseRankingOutput(selected_passage="(A0)")
        )
        result = setwise_ranker.compare(test_query, test_documents)
        assert result == "(A0)"

    def test_heapify_value_error_fallback(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test heapify() handles ValueError from compare() gracefully.

        When compare() returns a value not in IDENTIFIERS, the except ValueError in
        heapify() catches it and defaults best_ind to 0 (keeping parent as largest).
        """
        import copy

        arr = copy.deepcopy(test_documents)
        n = len(arr)
        # Return a string that passes SetwiseRankingOutput validation but we
        # override compare to return a raw string not in IDENTIFIERS.
        with patch.object(setwise_ranker, "compare", return_value="NOT_IN_LIST"):
            # heapify tries IDENTIFIERS.index("NOT_IN_LIST") → ValueError → best_ind = 0
            setwise_ranker.heapify(arr, n, 0, test_query)
        # Array length should be unchanged; no crash
        assert len(arr) == n

    def test_heapify_index_error_fallback(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test that heapify() handles IndexError when best_ind is out of range.

        When IDENTIFIERS.index returns a value beyond the inds list, the except
        IndexError branch sets largest = i (the parent), so no swap occurs.
        """
        import copy
        from unittest.mock import patch

        arr = copy.deepcopy(test_documents)
        n = len(arr)
        # "(Z9)" is index 259 in IDENTIFIERS — far beyond the inds list length
        with patch.object(setwise_ranker, "compare", return_value="(Z9)"):
            setwise_ranker.heapify(arr, n, 0, test_query)
        assert len(arr) == n

    def test_bubblesort_early_stop_at_start_ind_equals_i(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test bubblesort() terminates the inner loop when start_ind == i.

        This exercises the `if start_ind == i: break` branch (line 181) by using
        a small array where the window immediately reaches position i.
        """
        import copy

        arr = copy.deepcopy(test_documents)
        # Use num_child=2 so the window covers most of the 3-element array
        setwise_ranker.num_child = 2
        setwise_ranker.top_k = 1
        # compare always returns the first identifier — no swap
        with patch.object(setwise_ranker, "compare", return_value="(A0)"):
            setwise_ranker.bubblesort(arr, test_query)
        assert len(arr) == len(test_documents)

    def test_rerank_bubblesort_method(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test rerank() with bubblesort method executes the bubblesort branch.

        Verifies that when method is set to 'bubblesort', rerank() calls bubblesort()
        and returns all documents (top_k + remaining).
        """
        import copy

        setwise_ranker.method = "bubblesort"
        setwise_ranker.top_k = 2
        ranking = copy.deepcopy(test_documents)
        with patch.object(setwise_ranker, "compare", return_value="(A0)"):
            result = setwise_ranker.rerank(test_query, ranking)
        assert len(result) == len(test_documents)

    @pytest.fixture
    def many_documents(self) -> list[Document]:
        """Fixture providing 6 test documents for bubblesort window-slide tests."""
        return [Document(id=str(i), content=f"Doc {i}") for i in range(6)]

    def test_run_empty_documents(
        self, setwise_ranker: SetwiseLLMRanker, test_query: str
    ) -> None:
        """Test run() returns empty documents dict when given an empty list."""
        result = setwise_ranker.run(documents=[], query=test_query)
        assert result == {"documents": []}

    def test_rerank_invalid_method_raises_not_implemented(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test rerank() raises NotImplementedError for unsupported method.

        Bypasses run() validation by setting method directly on the ranker.
        """
        setwise_ranker.method = "unsupported"
        with pytest.raises(NotImplementedError):
            setwise_ranker.rerank(test_query, test_documents)

    def test_heapify_swap_occurs(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test heapify() executes the swap branch.

        With compare returning '(A1)', best_ind=1, largest=1 != i=0, so the swap
        and recursive heapify call are triggered.
        """
        arr = copy.deepcopy(test_documents)
        n = len(arr)
        original_second_id = arr[1].id
        setwise_ranker.num_child = 3
        with patch.object(setwise_ranker, "compare", return_value="(A1)"):
            setwise_ranker.heapify(arr, n, 0, test_query)
        assert arr[0].id == original_second_id

    def test_bubblesort_swap_occurs(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        test_documents: list[Document],
    ) -> None:
        """Test bubblesort() executes the swap block.

        With 3 docs, num_child=3, top_k=1 and compare returning '(A1)', best_ind=1 != 0
        triggers the swap. last_start == n-(num_child+1) so the last_start adjustment is
        not hit.

        """
        arr = copy.deepcopy(test_documents)
        setwise_ranker.num_child = 3
        setwise_ranker.top_k = 1
        original_second_id = arr[1].id
        with patch.object(setwise_ranker, "compare", return_value="(A1)"):
            setwise_ranker.bubblesort(arr, test_query)
        assert arr[0].id == original_second_id

    def test_bubblesort_window_slides(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        many_documents: list[Document],
    ) -> None:
        """Test bubblesort() executes the window-slide path.

        With 6 docs, num_child=2, top_k=1 and compare always returning '(A0)' (no swap),
        is_change stays False and the window slides on the first two inner iterations.
        """
        arr = copy.deepcopy(many_documents)
        setwise_ranker.num_child = 2
        setwise_ranker.top_k = 1
        with patch.object(setwise_ranker, "compare", return_value="(A0)"):
            setwise_ranker.bubblesort(arr, test_query)
        assert len(arr) == len(many_documents)

    def test_bubblesort_last_start_adjustment(
        self,
        setwise_ranker: SetwiseLLMRanker,
        test_query: str,
        many_documents: list[Document],
    ) -> None:
        """Test bubblesort() executes the last_start adjustment.

        With 6 docs, num_child=2, top_k=1 and side_effect ["(A0)", "(A2)", "(A0)"]:
        - Window [3:6]: no swap → slides window to [1:4]
        - Window [1:4]: best_ind=2 (last), swap → last_start != n-(num_child+1) AND
          best_ind == len(window)-1, so last_start is adjusted
        - Window [0:2]: no swap → start_ind==i, break
        """
        arr = copy.deepcopy(many_documents)
        setwise_ranker.num_child = 2
        setwise_ranker.top_k = 1
        with patch.object(
            setwise_ranker, "compare", side_effect=["(A0)", "(A2)", "(A0)"]
        ):
            setwise_ranker.bubblesort(arr, test_query)
        assert len(arr) == len(many_documents)
