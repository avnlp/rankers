import argparse
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from tqdm import tqdm

from rankers import Dataloader, Evaluator, EvaluatorParams, PairwiseLLMRanker
from rankers.config import PairwiseRankingConfig, load_config


def main(config_path: str):
    """Run a pipeline evaluating the quality of the retrieved documents after using the Pairwise LLM Ranker.

    The pipeline consists of:
    1. Loading dataset with ir_datasets format
    2. Initializing Milvus document store and embedding retriever
    3. Creating text embedding pipeline with Instructor model
    4. Reranking documents using PairwiseLLMRanker
    5. Evaluating results with standard IR metrics
    """
    config = load_config(Path(config_path), PairwiseRankingConfig)

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
    text_embedder = SentenceTransformersTextEmbedder(model=config.embedding.model, **config.embedding.model_kwargs)

    # Initialize ranker
    llm_ranker = PairwiseLLMRanker(
        model_name=config.llm.model_name,
        method=config.llm.method,
        top_k=config.llm.top_k,
        device=config.llm.device,
        model_kwargs=config.llm.model_kwargs,
        tokenizer_kwargs=config.llm.tokenizer_kwargs,
    )

    # Create and connect pipeline
    embedding_pipeline = Pipeline()
    embedding_pipeline.add_component(instance=text_embedder, name="text_embedder")
    embedding_pipeline.add_component(instance=milvus_retriever, name="embedding_retriever")
    embedding_pipeline.add_component(instance=llm_ranker, name="ranker")
    embedding_pipeline.connect("text_embedder", "embedding_retriever")
    embedding_pipeline.connect("embedding_retriever.documents", "ranker.documents")

    # Process each query
    all_query_results = {}
    for query_id, query in tqdm(queries.items()):
        pipeline_output = embedding_pipeline.run({"text_embedder": {"text": query}, "ranker": {"query": query}})
        ranked_documents = pipeline_output["ranker"]["documents"]
        document_scores = {}
        for document in ranked_documents:
            document_scores[document.meta["doc_id"]] = document.score
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
    evaluation_metrics = evaluator.evaluate()
    print(evaluation_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for Pairwise LLM Ranker in information retrieval tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    main(args.config_path)
