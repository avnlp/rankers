import argparse
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from tqdm import tqdm

from rankers import Dataloader, Evaluator, EvaluatorParams, ListwiseLLMRanker
from rankers.config import ListwiseRankingConfig, load_config


def main(config_path: str):
    """Run a pipeline evaluating the quality of the retrieved documents after using the Listwise LLM Ranker.

    The pipeline consists of:
    1. Loading dataset with ir_datasets format
    2. Initializing Milvus document store and embedding retriever
    3. Creating text embedding pipeline with Instructor model
    4. Reranking documents using ListwiseLLMRanker
    5. Evaluating results with standard IR metrics
    """
    # Load and validate configuration
    config = load_config(Path(config_path), ListwiseRankingConfig)

    # Initialize components from config
    dataloader = Dataloader(config.dataset.name)
    dataset = dataloader.load()
    queries = dataset.queries
    relevance_judgments = dataset.relevance_judgments

    # Initialize document store and retriever
    milvus_document_store = MilvusDocumentStore(
        connection_args={
            "uri": config.milvus.connection_uri,
            "token": config.milvus.connection_token or "",
        },
        **config.milvus.document_store_kwargs,
    )

    milvus_retriever = MilvusEmbeddingRetriever(
        document_store=milvus_document_store,
        top_k=config.retrieval.documents_to_retrieve,
        filters=config.retrieval.filters,
    )

    # Initialize text embedder
    text_embedder = SentenceTransformersTextEmbedder(
        model=config.embedding.model,
        **config.embedding.model_kwargs,
    )

    # Initialize LLM ranker
    ranker_kwargs = {
        "model_path": config.ranker.model_path,
        "ranker_type": config.ranker.ranker_type,
        "context_size": config.ranker.context_size,
        "num_few_shot_examples": config.ranker.num_few_shot_examples,
        "device": config.ranker.device,
        "num_gpus": config.ranker.num_gpus,
        "variable_passages": config.ranker.variable_passages,
        "sliding_window_size": config.ranker.sliding_window_size,
        "sliding_window_step": config.ranker.sliding_window_step,
        "system_message": config.ranker.system_message,
    }

    # Add OpenAI config if provided
    if config.ranker.openai is not None:
        ranker_kwargs.update(
            {
                "openai_api_keys": config.ranker.openai.api_keys,
                "openai_key_start_id": config.ranker.openai.key_start_id,
                "openai_proxy": config.ranker.openai.proxy,
                "api_type": config.ranker.openai.api_type,
                "api_base": config.ranker.openai.api_base,
                "api_version": config.ranker.openai.api_version,
            }
        )

    llm_ranker = ListwiseLLMRanker(**ranker_kwargs)

    # Create and connect pipeline
    embedding_pipeline = Pipeline()
    embedding_pipeline.add_component(instance=text_embedder, name="text_embedder")
    embedding_pipeline.add_component(instance=milvus_retriever, name="embedding_retriever")
    embedding_pipeline.add_component(instance=llm_ranker, name="ranker")

    embedding_pipeline.connect("text_embedder", "embedding_retriever")
    embedding_pipeline.connect("embedding_retriever.documents", "ranker.documents")

    # Process each query
    all_query_results = {}
    query_items = list(queries.items())

    for query_id, query in tqdm(query_items, desc="Processing queries"):
        pipeline_output = embedding_pipeline.run(
            {
                "text_embedder": {"text": query},
                "ranker": {"query": query, "top_k": config.ranker.top_k},
            }
        )

        ranked_documents = pipeline_output["ranker"]["documents"]
        document_scores = {document.meta["doc_id"]: document.score for document in ranked_documents}
        all_query_results[query_id] = document_scores

    # Evaluate results
    evaluation_config = EvaluatorParams(
        cutoff_values=tuple(config.evaluation.cutoff_values),
        ignore_identical_ids=config.evaluation.ignore_identical_ids,
        decimal_precision=config.evaluation.decimal_precision,
        metrics_to_compute=tuple(config.evaluation.metrics_to_compute),
    )

    evaluator = Evaluator(
        relevance_judgments=relevance_judgments,
        run_results=all_query_results,
        config=evaluation_config,
    )

    evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for Listwise LLM Ranker in information retrieval tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    main(config_path=args.config_path)
