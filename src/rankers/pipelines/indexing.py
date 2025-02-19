import argparse

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.instructor_embedders import InstructorDocumentEmbedder
from milvus_haystack import MilvusDocumentStore

from rankers.dataloader import Dataloader
from rankers.utils import dict_type


def main():
    """Index documents into Milvus vector database using Instructor embeddings."""
    parser = argparse.ArgumentParser(
        description="Indexing pipeline for instructor document embedder into Milvus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="beir/nfcorpus/test",
        help="Dataset identifier in ir_datasets format. Example: 'beir/fiqa/train' for BEIR FIQA dataset train split.",
    )

    # Embedding configuration
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="hkunlp/instructor-xl",
        help="Sentence embedding model for document encoding. Supports any INSTRUCTOR-compatible models.",
    )
    parser.add_argument(
        "--document_embedding_instruction",
        type=str,
        default="Represent the document for retrieval:",
        help="Prompt template for document embedding. Modify for domain-specific tasks.",
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

    args = parser.parse_args()

    # Load dataset
    data_loader = Dataloader(args.dataset_name)
    dataset = data_loader.load()
    corpus = dataset.corpus
    documents = [
        Document(
            content=text_dict["text"],
            meta={"doc_id": str(document_id)},
        )
        for document_id, text_dict in corpus.items()
    ]

    # Initialize document store
    document_store = MilvusDocumentStore(
        connection_args={
            "uri": args.milvus_connection_uri,
            "token": args.milvus_connection_token,
        },
        **args.milvus_document_store_kwargs,
    )

    # Initialize embedder
    doc_instruction = args.document_embedding_instruction
    embedder = InstructorDocumentEmbedder(
        model=args.embedding_model, instruction=doc_instruction, **args.embedder_kwargs
    )
    embedder.warm_up()

    # Create indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", DocumentWriter(document_store))
    indexing_pipeline.connect("embedder", "writer")

    # Index documents
    indexing_pipeline.run({"embedder": {"documents": documents}})


if __name__ == "__main__":
    main()
