from typing import Optional
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from pytest_mock import MockerFixture
from rank_llm.data import Candidate
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise.rank_gpt import SafeOpenai as RankGPT
from rank_llm.rerank.listwise.vicuna_reranker import VicunaReranker
from rank_llm.rerank.listwise.zephyr_reranker import ZephyrReranker

from rankers.listwise.listwise_ranker import ListwiseLLMRanker


@pytest.fixture
def query() -> str:
    """Provide a sample query string for testing.

    Returns:
        str: Test query string
    """
    return "test query"


@pytest.fixture
def documents() -> list[Document]:
    """Generate a list of test documents with metadata.

    Returns:
        List[Document]: List of Haystack Document objects with incremental content and order metadata
    """
    return [Document(content=f"Document {i}", id=f"doc_{i}", meta={"order": i}) for i in range(5)]


@pytest.fixture
def mock_rerank(mocker: MockerFixture) -> Mock:
    """Fixture for mocking the rerank method of ListwiseLLMRanker's underlying reranker.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock: Mock object patching the rerank method
    """

    def _mock(ranker: ListwiseLLMRanker, return_value: Optional[list[Candidate]] = None) -> Mock:
        mock = mocker.patch.object(ranker.reranker, "rerank")
        mock.return_value = return_value if return_value is not None else []
        return mock

    return _mock


class TestListwiseLLMRanker:
    """Test suite for validating the ListwiseLLMRanker component functionality."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_initialization_zephyr(self, mock_rank_zephyr_init: Mock, mock_cuda: Mock) -> None:
        """Verify correct initialization with Zephyr reranker configuration.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
        """
        rank_zephyr_model_path = "castorini/rank_zephyr_7b_v1_full"
        rank_zephyr = ListwiseLLMRanker(
            model_path=rank_zephyr_model_path,
            ranker_type="zephyr",
            context_size=4096,
            num_gpus=1,
            device="cuda",
        )

        assert isinstance(rank_zephyr.reranker, ZephyrReranker)
        assert rank_zephyr.model_path == rank_zephyr_model_path
        assert rank_zephyr.ranker_type == "zephyr"
        assert rank_zephyr.context_size == 4096
        assert rank_zephyr.prompt_mode == PromptMode.RANK_GPT
        assert rank_zephyr.num_few_shot_examples == 0
        assert rank_zephyr.device == "cuda"
        assert rank_zephyr.num_gpus == 1
        assert rank_zephyr.variable_passages is False
        assert rank_zephyr.sliding_window_size == 20
        assert rank_zephyr.sliding_window_step == 10
        assert (
            rank_zephyr.system_message
            == "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
        )
        assert rank_zephyr.openai_api_keys is None
        assert rank_zephyr.openai_key_start_id is None
        assert rank_zephyr.openai_proxy is None
        assert rank_zephyr.api_type is None
        assert rank_zephyr.api_base is None
        assert rank_zephyr.api_version is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.vicuna_reranker.VicunaReranker.__init__", return_value=None)
    def test_initialization_vicuna(self, mock_vicuna_init: Mock, mock_cuda: Mock) -> None:
        """Validate proper initialization with Vicuna reranker setup.

        Args:
            mock_vicuna_init: Mock for VicunaReranker constructor
            mock_cuda: Mock for CUDA availability check
        """
        rank_vicuna_model_path = "castorini/rank_vicuna_7b_v1"
        rank_vicuna = ListwiseLLMRanker(
            ranker_type="vicuna",
            model_path=rank_vicuna_model_path,
            context_size=4096,
            num_gpus=1,
            device="cuda",
        )

        assert isinstance(rank_vicuna.reranker, VicunaReranker)
        assert rank_vicuna.model_path == rank_vicuna_model_path
        assert rank_vicuna.ranker_type == "vicuna"
        assert rank_vicuna.context_size == 4096
        assert rank_vicuna.prompt_mode == PromptMode.RANK_GPT
        assert rank_vicuna.num_few_shot_examples == 0
        assert rank_vicuna.device == "cuda"
        assert rank_vicuna.num_gpus == 1
        assert rank_vicuna.variable_passages is False
        assert rank_vicuna.sliding_window_size == 20
        assert rank_vicuna.sliding_window_step == 10
        assert (
            rank_vicuna.system_message
            == "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
        )
        assert rank_vicuna.openai_api_keys is None
        assert rank_vicuna.openai_key_start_id is None
        assert rank_vicuna.openai_proxy is None
        assert rank_vicuna.api_type is None
        assert rank_vicuna.api_base is None
        assert rank_vicuna.api_version is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.rank_gpt.SafeOpenai.__init__", return_value=None)
    def test_initialization_rank_gpt(self, mock_rank_gpt_init: Mock, mock_cuda: Mock) -> None:
        """Test successful initialization with RankGPT configuration using API keys.

        Args:
            mock_rank_gpt_init: Mock for RankGPT constructor
            mock_cuda: Mock for CUDA availability check
        """
        test_api_keys = ["dummy_key"]
        rank_gpt = ListwiseLLMRanker(ranker_type="rank_gpt", model_path="gpt-4o-mini", openai_api_keys=test_api_keys)

        assert isinstance(rank_gpt.reranker, RankGPT)
        assert rank_gpt.model_path == "gpt-4o-mini"
        assert rank_gpt.ranker_type == "rank_gpt"
        assert rank_gpt.context_size == 4096
        assert rank_gpt.prompt_mode == PromptMode.RANK_GPT
        assert rank_gpt.num_few_shot_examples == 0
        assert rank_gpt.device == "cuda"
        assert rank_gpt.num_gpus == 1
        assert rank_gpt.variable_passages is False
        assert rank_gpt.sliding_window_size == 20
        assert rank_gpt.sliding_window_step == 10
        assert (
            rank_gpt.system_message
            == "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
        )
        assert rank_gpt.openai_api_keys == test_api_keys
        assert rank_gpt.openai_key_start_id is None
        assert rank_gpt.openai_proxy is None
        assert rank_gpt.api_type is None
        assert rank_gpt.api_base is None
        assert rank_gpt.api_version is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_run_with_no_documents_raises_error(self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, query: str) -> None:
        """Ensure ValueError is raised when no documents are provided for reranking.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            query: Test query fixture
        """
        empty_doc_ranker = ListwiseLLMRanker(
            model_path="castorini/rank_zephyr_7b_v1_full", ranker_type="zephyr", context_size=4096
        )
        with pytest.raises(ValueError, match="No documents provided for reranking"):
            empty_doc_ranker.run(query=query, documents=[])

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_run_returns_correct_order(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture, query: str, documents: list[Document]
    ) -> None:
        """Validate document ordering matches mock reranker's output sequence.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
            query: Test query fixture
            documents: Test documents fixture
        """
        rank_zephyr = ListwiseLLMRanker(ranker_type="zephyr")
        mock_rerank = mocker.patch.object(rank_zephyr.reranker, "rerank")
        mock_rerank.return_value = [
            Candidate(doc={"text": "Document 2"}, docid="doc_2", score=1),
            Candidate(doc={"text": "Document 0"}, docid="doc_0", score=1),
            Candidate(doc={"text": "Document 1"}, docid="doc_1", score=1),
            Candidate(doc={"text": "Document 3"}, docid="doc_3", score=1),
            Candidate(doc={"text": "Document 4"}, docid="doc_4", score=1),
        ]

        reranked_results = rank_zephyr.run(query=query, documents=documents)
        assert [doc.content for doc in reranked_results["documents"]] == [
            "Document 2",
            "Document 0",
            "Document 1",
            "Document 3",
            "Document 4",
        ]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_run_applies_top_k(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture, query: str, documents: list[Document]
    ) -> None:
        """Verify top_k parameter correctly limits returned results count.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
            query: Test query fixture
            documents: Test documents fixture
        """
        rank_zephyr = ListwiseLLMRanker(ranker_type="zephyr")
        mock_rerank = mocker.patch.object(rank_zephyr.reranker, "rerank")
        mock_rerank.return_value = [
            Candidate(doc={"text": "Document 2"}, docid="doc_2", score=1),
            Candidate(doc={"text": "Document 0"}, docid="doc_0", score=1),
            Candidate(doc={"text": "Document 1"}, docid="doc_1", score=1),
            Candidate(doc={"text": "Document 3"}, docid="doc_3", score=1),
            Candidate(doc={"text": "Document 4"}, docid="doc_4", score=1),
        ]

        top_k = 3
        top_k_results = rank_zephyr.run(query=query, documents=documents, top_k=top_k)
        assert len(top_k_results["documents"]) == top_k
        assert [doc.content for doc in top_k_results["documents"]] == ["Document 2", "Document 0", "Document 1"]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_run_preserves_metadata(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture, query: str, documents: list[Document]
    ) -> None:
        """Ensure document metadata remains unchanged after reranking process.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
            query: Test query fixture
            documents: Test documents fixture
        """
        rank_zephyr = ListwiseLLMRanker(ranker_type="zephyr")
        mock_rerank = mocker.patch.object(rank_zephyr.reranker, "rerank")
        mock_rerank.return_value = [Candidate(doc={"text": doc.content}, docid=doc.id, score=1) for doc in documents]

        reranked_documents = rank_zephyr.run(query=query, documents=documents)
        for original_doc, reranked_doc in zip(documents, reranked_documents["documents"]):
            assert original_doc.meta == reranked_doc.meta
            assert original_doc.content == reranked_doc.content

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_run_deepcopies_metadata(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture, query: str
    ) -> None:
        """Validate metadata deepcopy prevents original document modification.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
            query: Test query fixture
        """
        original_doc = Document(content="test", meta={"key": "original"})
        rank_zephyr = ListwiseLLMRanker(ranker_type="zephyr")
        mock_rerank = mocker.patch.object(rank_zephyr.reranker, "rerank")
        mock_rerank.return_value = [Candidate(doc={"text": "test"}, docid=original_doc.id, score=1)]

        modified_results = rank_zephyr.run(query=query, documents=[original_doc])
        modified_results["documents"][0].meta["key"] = "modified"
        assert original_doc.meta["key"] == "original"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_run_passes_top_k_retrieve_none(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture, query: str, documents: list[Document]
    ) -> None:
        """Confirm None values for top_k_retrieve are properly handled.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
            query: Test query fixture
            documents: Test documents fixture
        """
        rank_zephyr = ListwiseLLMRanker(ranker_type="zephyr")
        mock_rerank = mocker.patch.object(rank_zephyr.reranker, "rerank")
        rank_zephyr.run(query=query, documents=documents, top_k=None)

        _, called_kwargs = mock_rerank.call_args
        assert called_kwargs["top_k_retrieve"] is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_sliding_window_params_passed(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture
    ) -> None:
        """Verify sliding window parameters are correctly propagated to reranker.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
        """
        test_window_size = 30
        test_window_step = 15
        mock_rank_zephyr = mocker.patch(
            "rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None
        )

        ListwiseLLMRanker(
            ranker_type="zephyr", sliding_window_size=test_window_size, sliding_window_step=test_window_step
        )

        _, constructor_kwargs = mock_rank_zephyr.call_args
        assert constructor_kwargs["window_size"] == test_window_size

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_system_message_passed(self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture) -> None:
        """Ensure custom system messages are properly forwarded to reranker.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
        """
        custom_system_message = "Custom system message"
        mock_rank_zephyr = mocker.patch(
            "rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None
        )

        ListwiseLLMRanker(ranker_type="zephyr", system_message=custom_system_message)

        _, constructor_kwargs = mock_rank_zephyr.call_args
        assert constructor_kwargs["system_message"] == custom_system_message

    @patch("torch.cuda.is_available", return_value=True)
    @patch("rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None)
    def test_variable_passages_passed(
        self, mock_rank_zephyr_init: Mock, mock_cuda: Mock, mocker: MockerFixture
    ) -> None:
        """Validate variable passages flag is correctly passed to reranker.

        Args:
            mock_rank_zephyr_init: Mock for ZephyrReranker constructor
            mock_cuda: Mock for CUDA availability check
            mocker: Pytest mocker fixture
        """
        variable_passages_flag = True
        mock_rank_zephyr = mocker.patch(
            "rank_llm.rerank.listwise.zephyr_reranker.ZephyrReranker.__init__", return_value=None
        )

        ListwiseLLMRanker(ranker_type="zephyr", variable_passages=variable_passages_flag)

        _, constructor_kwargs = mock_rank_zephyr.call_args
        assert constructor_kwargs["variable_passages"] == variable_passages_flag
