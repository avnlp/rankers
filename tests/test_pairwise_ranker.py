import copy
from unittest.mock import patch

import pytest
from haystack import Document

from rankers.pairwise.pairwise_ranker import PairwiseLLMRanker, PairwiseRankingOutput


class DummyPairwiseGeneration:
    """Mock implementation of StructuredGeneration for pairwise comparisons."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the mock generator with predefined responses."""
        self.responses = {
            "A": PairwiseRankingOutput(selected_passage="A"),
            "B": PairwiseRankingOutput(selected_passage="B"),
            "tie": PairwiseRankingOutput(selected_passage="A"),  # Mock tie handling
        }

    def generate(self, output_format: str, user_prompt: str, system_prompt: str) -> PairwiseRankingOutput:
        """Mock generate method with controlled responses based on the user prompt.

        Args:
            output_format: The format of the output.
            user_prompt: The user's prompt.
            system_prompt: The system's prompt.

        Returns:
            A PairwiseRankingOutput indicating the selected passage.
        """
        if "Passage A: A" in user_prompt and "Passage B: B" in user_prompt:
            return self.responses["A"]
        elif "Passage A: B" in user_prompt and "Passage B: A" in user_prompt:
            return self.responses["B"]
        return self.responses["tie"]


@pytest.fixture(autouse=True)
def patch_structured_generation() -> None:
    """Fixture to patch StructuredGeneration with DummyPairwiseGeneration."""
    with patch("rankers.pairwise.pairwise_ranker.StructuredGeneration", DummyPairwiseGeneration):
        yield


@pytest.fixture
def test_documents() -> list[Document]:
    """Fixture providing a list of test documents."""
    return [Document(id="A", content="A"), Document(id="B", content="B"), Document(id="C", content="C")]


@pytest.fixture
def test_query() -> str:
    """Fixture providing a test query string."""
    return "test query"


@pytest.fixture
def pairwise_ranker() -> PairwiseLLMRanker:
    """Fixture providing an initialized PairwiseLLMRanker for testing."""
    ranker = PairwiseLLMRanker(model_name="test-model", method="allpair", top_k=2)
    ranker._is_warmed_up = True
    ranker._generator = DummyPairwiseGeneration()
    return ranker


class TestPairwiseLLMRanker:
    """Test suite for the PairwiseLLMRanker class."""

    def test_initialization(self) -> None:
        """Test that the ranker initializes correctly with provided parameters."""
        ranker = PairwiseLLMRanker(
            model_name="test-model",
            device="cuda",
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"use_fast": True},
            method="bubblesort",
            top_k=5,
        )
        assert ranker.model_name == "test-model"
        assert ranker.device == "cuda"
        assert ranker.model_kwargs == {"trust_remote_code": True}
        assert ranker.tokenizer_kwargs == {"use_fast": True}
        assert ranker.method == "bubblesort"
        assert ranker.top_k == 5
        assert not ranker._is_warmed_up
        assert ranker._generator is None

    def test_warm_up(self, pairwise_ranker: PairwiseLLMRanker) -> None:
        """Test that the warm_up method initializes the generator correctly."""
        pairwise_ranker._warm_up()
        assert pairwise_ranker._is_warmed_up
        assert isinstance(pairwise_ranker._generator, DummyPairwiseGeneration)

    def test_compare_pair_conflict_resolution(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test conflict resolution in pairwise comparisons."""
        # Test clear A preference
        pairwise_ranker._generator.responses = {
            "A": PairwiseRankingOutput(selected_passage="A"),
            "B": PairwiseRankingOutput(selected_passage="B"),
        }
        result = pairwise_ranker._compare_pair(test_query, test_documents[0], test_documents[1])
        assert result == "A"

        # Test clear B preference
        pairwise_ranker._generator.responses = {
            "A": PairwiseRankingOutput(selected_passage="B"),
            "B": PairwiseRankingOutput(selected_passage="A"),
        }
        result = pairwise_ranker._compare_pair(test_query, test_documents[0], test_documents[1])
        assert result == "B"

        # Test tie handling
        pairwise_ranker._generator.responses = {
            "A": PairwiseRankingOutput(selected_passage="B"),
            "B": PairwiseRankingOutput(selected_passage="B"),
        }
        result = pairwise_ranker._compare_pair(test_query, test_documents[0], test_documents[1])
        assert result == "tie"

    def test_allpair_reranking(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test reranking using the allpair method."""
        pairwise_ranker.top_k = 2
        pairwise_ranker._generator.responses = {
            "A": PairwiseRankingOutput(selected_passage="A"),
            "B": PairwiseRankingOutput(selected_passage="B"),
            "tie": PairwiseRankingOutput(selected_passage="A"),
        }
        reranked = pairwise_ranker._allpair_rerank(test_query, test_documents)
        assert [d.id for d in reranked] == ["A", "C"]

    def test_heapsort_reranking(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test reranking using the heapsort method."""
        pairwise_ranker.method = "heapsort"
        pairwise_ranker.top_k = 2
        pairwise_ranker._generator.responses = {
            "A": PairwiseRankingOutput(selected_passage="A"),
            "B": PairwiseRankingOutput(selected_passage="B"),
            "tie": PairwiseRankingOutput(selected_passage="A"),
        }
        reranked = pairwise_ranker._heapsort_rerank(test_query, test_documents)
        assert len(reranked) == 2
        assert "A" in [d.id for d in reranked]

    def test_bubblesort_reranking(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test reranking using the bubblesort method."""
        pairwise_ranker.method = "bubblesort"
        pairwise_ranker.top_k = 2
        pairwise_ranker._generator.responses = {
            "A": PairwiseRankingOutput(selected_passage="A"),
            "B": PairwiseRankingOutput(selected_passage="B"),
            "tie": PairwiseRankingOutput(selected_passage="A"),
        }
        reranked = pairwise_ranker._bubblesort_rerank(test_query, test_documents)
        assert len(reranked) == 2
        assert "A" in [d.id for d in reranked]

    def test_run_method(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test the run method with default parameters."""
        result = pairwise_ranker.run(documents=test_documents, query=test_query, method="allpair", top_k=2)
        assert "documents" in result
        assert len(result["documents"]) == 3  # top_k + remaining
        assert result["documents"][0].id in ["A", "B"]

    def test_preserve_original_order(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test that the original order of documents is preserved for non-top_k documents."""
        pairwise_ranker.top_k = 2
        result = pairwise_ranker.run(documents=test_documents, query=test_query)
        remaining = [d for d in result["documents"] if d.id not in {"A", "B"}]
        assert remaining == [test_documents[2]]

    @pytest.mark.parametrize("invalid_method", ["invalid", "mergesort"])
    def test_invalid_method_handling(
        self, invalid_method: str, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test that an invalid method raises a ValueError."""
        with pytest.raises(ValueError):
            pairwise_ranker.run(documents=test_documents, query=test_query, method=invalid_method)

    def test_top_k_handling(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test handling when top_k is set to 0."""
        pairwise_ranker.top_k = 0
        result = pairwise_ranker.run(documents=test_documents, query=test_query)
        assert result["documents"] == []

    def test_empty_documents_handling(self, pairwise_ranker: PairwiseLLMRanker, test_query: str) -> None:
        """Test handling when an empty list of documents is provided."""
        result = pairwise_ranker.run(documents=[], query=test_query)
        assert result["documents"] == []

    def test_runtime_parameter_override(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test that runtime parameters override the initial settings."""
        result = pairwise_ranker.run(documents=test_documents, query=test_query, method="bubblesort", top_k=1)
        assert len(result["documents"]) == 3  # Original 3 docs
        assert pairwise_ranker.method == "bubblesort"
        assert pairwise_ranker.top_k == 1

    def test_document_immutability(
        self, pairwise_ranker: PairwiseLLMRanker, test_documents: list[Document], test_query: str
    ) -> None:
        """Test that the original documents are not modified during reranking."""
        original_docs = copy.deepcopy(test_documents)
        _ = pairwise_ranker.run(documents=test_documents, query=test_query)
        assert test_documents == original_docs
