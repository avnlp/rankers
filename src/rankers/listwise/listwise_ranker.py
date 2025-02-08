from copy import deepcopy
from typing import Literal, Optional

import weave
from haystack import Document, component
from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise.rank_gpt import SafeOpenai as RankGPT
from rank_llm.rerank.listwise.vicuna_reranker import VicunaReranker
from rank_llm.rerank.listwise.zephyr_reranker import ZephyrReranker


@component
class ListwiseLLMRanker:
    """A Haystack component for listwise reranking of documents using Large Language Models (LLMs).

    This component utilizes LLMs such as Zephyr, Vicuna, or RankGPT to rerank documents based on their relevance to a
    query. It supports different reranking strategies, sliding window processing for handling long document lists, and
    integration with various model backends including OpenAI and Azure AI.

    Example:
        ```python
        ranker = ListwiseLLMRanker(ranker_type="zephyr", model_path="castorini/rank_zephyr_7b_v1_full")
        results = ranker.run(query="What is the capital of France?", documents=documents, top_k=10)
        ```
    """

    def __init__(
        self,
        model_path: str = "castorini/rank_zephyr_7b_v1_full",
        ranker_type: Literal["zephyr", "vicuna", "rank_gpt"] = "zephyr",
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        sliding_window_size: int = 20,
        sliding_window_step: int = 10,
        system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
        openai_api_keys: Optional[list[str]] = None,
        openai_key_start_id: Optional[int] = None,
        openai_proxy: Optional[str] = None,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """Initialize the ListwiseLLMRanker component with the specified parameters.

        Args:
            model_path (str): Path or identifier of the pre-trained model to use. Defaults to
                "castorini/rank_zephyr_7b_v1_full".
            ranker_type (Literal["zephyr", "vicuna", "rank_gpt"]): Type of reranker to use. Defaults to "zephyr".
            context_size (int): Maximum context size in tokens that the model can handle. Defaults to 4096.
            prompt_mode (PromptMode): Prompt generation mode. Defaults to PromptMode.RANK_GPT.
            num_few_shot_examples (int): Number of few-shot examples to include in prompts. Defaults to 0.
            device (str): Device for inference ("cuda" or "cpu"). Defaults to "cuda".
            num_gpus (int): Number of GPUs to use. Defaults to 1.
            variable_passages (bool): Whether to allow variable number of passages per request. Defaults to False.
            sliding_window_size (int): Size of sliding window for processing long documents. Defaults to 20.
            sliding_window_step (int): Step size for sliding window movement. Defaults to 10.
            system_message (str): System message to configure LLM behavior. Defaults to standard ranking message.
            openai_api_keys (Optional[list[str]]): OpenAI API keys for authentication. Required for RankGPT.
            openai_key_start_id (Optional[int]): Starting index for OpenAI key rotation. Defaults to None.
            openai_proxy (Optional[str]): Proxy configuration for OpenAI API. Defaults to None.
            api_type (Optional[str]): API type for Azure AI services. Defaults to None.
            api_base (Optional[str]): API base URL for Azure AI services. Defaults to None.
            api_version (Optional[str]): API version for Azure AI services. Defaults to None.
        """
        self.model_path = model_path
        self.ranker_type = ranker_type
        self.context_size = context_size
        self.prompt_mode = prompt_mode
        self.num_few_shot_examples = num_few_shot_examples
        self.device = device
        self.num_gpus = num_gpus
        self.variable_passages = variable_passages
        self.sliding_window_size = sliding_window_size
        self.sliding_window_step = sliding_window_step
        self.system_message = system_message
        self.openai_api_keys = openai_api_keys
        self.openai_key_start_id = openai_key_start_id
        self.openai_proxy = openai_proxy
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version

        if ranker_type == "zephyr":
            self.reranker = ZephyrReranker(
                model_path=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                device=device,
                num_gpus=num_gpus,
                variable_passages=variable_passages,
                window_size=sliding_window_size,
                system_message=system_message,
            )
        elif ranker_type == "vicuna":
            self.reranker = VicunaReranker(
                model_path=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                device=device,
                num_gpus=num_gpus,
                variable_passages=variable_passages,
                window_size=sliding_window_size,
                system_message=system_message,
            )
        elif ranker_type == "rank_gpt":
            self.reranker = RankGPT(
                model=self.model_path,
                context_size=self.context_size,
                prompt_mode=self.prompt_mode,
                num_few_shot_examples=self.num_few_shot_examples,
                window_size=self.sliding_window_size,
                keys=self.openai_api_keys,
                key_start_id=self.openai_key_start_id,
                proxy=self.openai_proxy,
                api_type=self.api_type,
                api_base=self.api_base,
                api_version=self.api_version,
            )

    @weave.op()
    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: Optional[int] = None) -> dict:
        """Rerank documents based on query relevance using the configured LLM.

        Args:
            query (str): Search query to rank documents against
            documents (list[Document]): List of Haystack Document objects to rerank
            top_k (Optional[int]): Number of top documents to return. Returns all if None.

        Returns:
            dict: Dictionary with key "documents" containing reranked Document objects

        Raises:
            ValueError: If no documents are provided
        """
        if not documents:
            msg = "No documents provided for reranking"
            raise ValueError(msg)

        doc_id_to_doc = {doc.id: doc for doc in documents}
        request = Request(
            query=Query(query, qid=1),
            candidates=[Candidate(doc={"text": doc.content}, docid=doc.id, score=1) for doc in documents],
        )

        rerank_results = self.reranker.rerank(
            request=request, rank_end=len(documents), step=self.sliding_window_step, top_k_retrieve=top_k
        )

        results = []
        for result in rerank_results:
            if doc := doc_id_to_doc.get(result.docid):
                doc_copy = Document(content=doc.content, meta=deepcopy(doc.meta))
                results.append(doc_copy)

        if top_k:
            results = results[:top_k]

        return {"documents": results}
