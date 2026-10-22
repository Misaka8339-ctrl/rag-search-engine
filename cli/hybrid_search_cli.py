import argparse

from lib.data import load_movies
from lib.hybrid_search import (
    normalize_scores,
    HybridSearch,
)

from lib.query_enhancement import (
    correct_spelling,
    rewrite_query,
    expand_query,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores using min-max normalization",
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*",
        help="Scores to normalize",
    )

    weighted_parser = subparsers.add_parser(
    "weighted-search",
    help="Search movies using weighted keyword and semantic scores",
    )
    weighted_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Keyword weight between 0 and 1",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Search movies using Reciprocal Rank Fusion",
    )
    rrf_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF smoothing constant",
    )
    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")

        case "weighted-search":
            if not 0 <= args.alpha <= 1:
                parser.error("--alpha must be between 0 and 1")

            if args.limit <= 0:
                parser.error("--limit must be greater than 0")

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.weighted_search(
                args.query,
                alpha=args.alpha,
                limit=args.limit,
            )

            for i, result in enumerate(results[:args.limit], start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {result['document'][:100]}...")

        case "rrf-search":
            if args.k < 0:
                parser.error("-k must be non-negative")

            if args.limit <= 0:
                parser.error("--limit must be greater than 0")

            query = args.query

            if args.enhance is not None:
                match args.enhance:
                    case "spell":
                        enhanced_query = correct_spelling(query)
                    case "rewrite":
                        enhanced_query = rewrite_query(query)
                    case "expand":
                        enhanced_query = expand_query(query)
        
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )

                query = enhanced_query

            documents = load_movies()
            search = HybridSearch(documents)

            results = search.rrf_search(
                args.query,
                k=args.k,
                limit=args.limit,
            )

            for i, result in enumerate(results[:args.limit], start=1):
                bm25_rank = result["bm25_rank"]
                semantic_rank = result["semantic_rank"]

                if bm25_rank is None:
                    bm25_rank = "N/A"
                if semantic_rank is None:
                    semantic_rank = "N/A"

                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"  BM25 Rank: {bm25_rank}, "
                    f"Semantic Rank: {semantic_rank}"
                )
                print(f"  {result['document'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()