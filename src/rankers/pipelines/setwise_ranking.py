"""Setwise LLM ranking pipeline."""

import argparse
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from tqdm import tqdm

from rankers import Dataloader, Evaluator, EvaluatorParams, SetwiseLLMRanker
from rankers.config import SetwiseRankingConfig, load_config


def main(config_path: str):
    """Retrieve, rerank with a Setwise LLM ranker, and evaluate IR metrics.

    The pipeline consists of:
    1. Loading configuration from YAML
    2. Loading dataset with ir_datasets format
    3. Initializing Milvus document store and embedding retriever
    4. Creating text embedding pipeline with Instructor model
    5. Reranking documents using SetwiseLLMRanker
    6. Evaluating results with standard IR metrics
    """
    # Load and validate configuration
    config = load_config(Path(config_path), SetwiseRankingConfig)

    # Load dataset
    dataloader = Dataloader(config.dataset.name)
    dataset = dataloader.load()
    queries = dataset.queries
    relevance_judgments = dataset.relevance_judgments

    # Initialize document store and retriever
    milvus_document_store = MilvusDocumentStore(
        connection_args={
            "uri": config.milvus.connection_uri,
            "token": config.milvus.connection_token,
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
        model=config.embedding.model, **config.embedding.model_kwargs
    )

    # Initialize ranker
    llm_ranker = SetwiseLLMRanker(
        model_name=config.llm.model_name,
        method=config.llm.method,
        top_k=config.llm.top_k,
        num_permutation=config.llm.num_permutation,
        num_child=config.llm.num_child,
        device=config.llm.device,
        model_kwargs=config.llm.model_kwargs,
        tokenizer_kwargs=config.llm.tokenizer_kwargs,
    )

    # Build and run pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", milvus_retriever)
    pipeline.add_component("ranker", llm_ranker)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")

    # Process each query
    all_query_results = {}
    for q_id, query in tqdm(queries.items(), desc="Processing queries"):
        result = pipeline.run(
            {
                "text_embedder": {"text": query},
                "ranker": {"query": query},
            }
        )
        ranked_documents = result["ranker"]["documents"]
        all_query_results[q_id] = {
            doc.meta["doc_id"]: doc.score for doc in ranked_documents
        }

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
    evaluation_metrics = evaluator.evaluate()
    print(evaluation_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for Setwise LLM Ranker in information retrieval tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )

    args = parser.parse_args()

    main(config_path=args.config)
