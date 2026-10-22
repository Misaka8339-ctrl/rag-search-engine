import argparse
import sys

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
    SemanticSearch,
    chunk_text,
    semantic_chunk_text,
    ChunkedSemanticSearch,
)

from lib.data import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    subparsers.add_parser(
        "verify",
        help="Load the embedding model and print its information",
    )

    embed_parser = subparsers.add_parser(
        "embed_text",
        help="Generate an embedding for a text input",
    )
    embed_parser.add_argument(
        "text",
        type=str,
        help="Text to embed",
    )

    subparsers.add_parser(
        "verify_embeddings",
        help="Build or load movie embeddings and verify their shape",
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate an embedding for a search query",
    )
    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies by semantic similarity",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into fixed-size word chunks",
    )
    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk",
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum number of words per chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words shared between consecutive chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into chunks at sentence boundaries",
    )
    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk",
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences shared between consecutive chunks",
    )

    subparsers.add_parser(
        "embed_chunks",
        help="Generate and cache embeddings for movie chunks",
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search movies using chunk embeddings",
    )
    search_chunked_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            try:
                embed_text(args.text)
            except ValueError as error:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search = SemanticSearch()
            documents = load_movies()
            search.load_or_create_embeddings(documents)

            results = search.search(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(
                    f"{i}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )

                description = result["description"]
                if len(description) > 100:
                    description = description[:100] + "..."

                print(f"  {description}")
                print()

        case "chunk":
            try:
                chunks = chunk_text(
                    args.text,
                    args.chunk_size,
                    args.overlap,
                )
            except ValueError as exc:
                parser.error(str(exc))

            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            try:
                chunks = semantic_chunk_text(
                    args.text,
                    args.max_chunk_size,
                    args.overlap,
                )
            except ValueError as exc:
                parser.error(str(exc))

            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            documents = load_movies()
            search = ChunkedSemanticSearch()
            embeddings = search.load_or_create_chunk_embeddings(documents)

            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            documents = load_movies()
            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(documents)

            results = search.search_chunks(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(
                    f"\n{i}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"   {result['document']}...")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()