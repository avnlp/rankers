import argparse

from haystack import Pipeline
from haystack_integrations.components.embedders.instructor_embedders import InstructorTextEmbedder
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from tqdm import tqdm

from rankers import Dataloader, Evaluator, EvaluatorConfig, SetwiseLLMRanker
from rankers.utils import dict_type


def main():
    """Run a pipeline evaluating the quality of the retrieved documents after using the Setwise LLM Ranker.

    The pipeline consists of:
    1. Loading dataset with ir_datasets format
    2. Initializing Milvus document store and embedding retriever
    3. Creating text embedding pipeline with Instructor model
    4. Reranking documents using SetwiseLLMRanker
    5. Evaluating results with standard IR metrics
    """
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for Setwise LLM Ranker in information retrieval tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset identifier in ir_datasets format. Example: 'beir/fiqa/train' for BEIR FiQA dataset train split.",
    )

    # LLM Ranking configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name/path for the LLM ranker. Example: 'meta-llama/Llama-3.1-8B-Instruct'",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="heapsort",
        choices=["heapsort", "bubblesort"],
        help="Sorting method to be used. Possible values: Heapsort - efficient tree-based sorting or "
        "Bubblesort - sliding window pairwise comparisons.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Maximum number of documents to rerank and return. Higher values increase computation time.",
    )
    parser.add_argument(
        "--num_permutation",
        type=int,
        default=1,
        help="Number of permutations for reranking.",
    )
    parser.add_argument(
        "--num_child",
        type=int,
        default=3,
        help="For heapsort: children per node. For bubblesort: window size. Controls comparison complexity.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model execution. Auto-detected if None. Examples: 'cuda', 'cuda:0', 'cpu'",
    )

    # Model initialization parameters
    parser.add_argument(
        "--model_kwargs",
        type=dict_type,
        default={},
        help="Additional model initialization parameters. Example: \"{'load_in_4bit': True, 'trust_remote_code': True}\"",
    )
    parser.add_argument(
        "--tokenizer_kwargs",
        type=dict_type,
        default={},
        help="Tokenizer configuration. Example: \"{'padding_side': 'left', 'truncation': True}\"",
    )

    # Embedding configuration
    parser.add_argument(
        "--embedder_model",
        type=str,
        default="hkunlp/instructor-xl",
        help="Sentence embedding model for query/document encoding. Supports any INSTRUCTOR-compatible models.",
    )
    parser.add_argument(
        "--query_embedding_instruction",
        type=str,
        default="Represent the question for retrieving supporting documents:",
        help="Prompt template for query embedding. Modify for domain-specific tasks.",
    )
    parser.add_argument(
        "--embedder_kwargs",
        type=dict_type,
        default={},
        help="Embedding model parameters. Example: \"{'device': 'cuda', 'normalize_embeddings': True}\"",
    )

    # Milvus configuration
    parser.add_argument(
        "--milvus_connection_uri",
        type=str,
        required=True,
        help="Milvus server URI. Example: 'http://localhost:19530' for local instance",
    )
    parser.add_argument(
        "--milvus_connection_token",
        type=str,
        required=True,
        help="Authentication token for Milvus. Use empty string for local instances without auth",
    )
    parser.add_argument(
        "--milvus_document_store_kwargs",
        type=dict_type,
        default={},
        help="Milvus collection parameters. Example: \"{'collection_name': 'my_docs', 'index_params': {'metric_type': 'L2'}}\"",
    )

    # Retrieval configuration
    parser.add_argument(
        "--filters",
        type=dict_type,
        default={},
        help="Document metadata filters. Example: \"{'date': {'$gte': '2020-01-01'}, 'lang': 'en'}\"",
    )
    parser.add_argument(
        "--documents_to_retrieve",
        type=int,
        default=25,
        help="Number of documents to retrieve per query. Defaults to 25.",
    )

    # Evaluation configuration
    parser.add_argument(
        "--cutoff_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Cutoff levels for evaluation metrics (e.g., NDCG@k). Multiple values supported.",
    )
    parser.add_argument(
        "--ignore_identical_ids",
        action="store_true",
        help="Exclude documents with same ID as query (prevents test set leakage)",
    )
    parser.add_argument(
        "--decimal_precision", type=int, default=4, help="Decimal precision for metric reporting. Range: 0-6"
    )
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        nargs="+",
        default=["ndcg", "map", "recall", "precision"],
        choices=["ndcg", "map", "recall", "precision"],
        help="Evaluation metrics to compute. Available: NDCG, MAP, Recall and Precision.",
    )

    args = parser.parse_args()

    # Load dataset
    dataloader = Dataloader(args.dataset_name)
    dataset = dataloader.load()
    queries = dataset.queries
    relevance_judgments = dataset.relevance_judgments

    # Initialize document store and retriever
    milvus_document_store = MilvusDocumentStore(
        connection_args={
            "uri": args.milvus_connection_uri,
            "token": args.milvus_connection_token,
        },
        **args.milvus_document_store_kwargs,
    )
    milvus_retriever = MilvusEmbeddingRetriever(
        document_store=milvus_document_store, top_k=args.documents_to_retrieve, filters=args.filters
    )

    # Initialize text embedder
    query_instruction = args.query_embedding_instruction
    text_embedder = InstructorTextEmbedder(
        model=args.embedder_model, instruction=query_instruction, **args.embedder_kwargs
    )

    # Initialize ranker
    llm_ranker = SetwiseLLMRanker(
        model_name=args.model_name,
        method=args.method,
        top_k=args.top_k,
        num_permutation=args.num_permutation,
        num_child=args.num_child,
        device=args.device,
        model_kwargs=args.model_kwargs,
        tokenizer_kwargs=args.tokenizer_kwargs,
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
    evaluation_config = EvaluatorConfig(
        cutoff_values=tuple(args.cutoff_values),
        ignore_identical_ids=args.ignore_identical_ids,
        decimal_precision=args.decimal_precision,
        metrics_to_compute=tuple(args.metrics_to_compute),
    )
    evaluator = Evaluator(
        relevance_judgments=relevance_judgments,
        run_results=all_query_results,
        config=evaluation_config,
    )
    evaluation_metrics = evaluator.evaluate()
    print(evaluation_metrics)


if __name__ == "__main__":
    main()
