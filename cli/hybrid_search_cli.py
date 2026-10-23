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

from lib.reranking import (
    rerank_individual,
    rerank_batch,
    rerank_cross_encoder,
)

import logging

logger = logging.getLogger(__name__)

from lib.search_evaluation import evaluate_results


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
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method",
    )
    rrf_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug logs for each search stage",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate final search results using an LLM",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "debug", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    
    # 避免第三方库的 DEBUG 日志淹没搜索流程日志
    for name in ("httpx", "httpcore", "openai", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


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
            logger.debug("Original query: %r", query)

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

            logger.debug(
                "Query after enhancement (%s): %r",
                args.enhance or "none",
                query,
            )

            documents = load_movies()
            search = HybridSearch(documents)

            candidate_limit = (
                args.limit * 5
                if args.rerank_method is not None
                else args.limit
            )

            results = search.rrf_search(
                query,
                k=args.k,
                limit=candidate_limit,
            )

            # rrf_search 返回全部候选，先截取再交给 LLM
            results = results[:candidate_limit]

            logger.debug("RRF candidates: %d", len(results))

            for rank, result in enumerate(results, start=1):
                logger.debug(
                    "RRF #%d: id=%s title=%r score=%.6f "
                    "bm25_rank=%s semantic_rank=%s",
                    rank,
                    result["id"],
                    result["title"],
                    result["rrf_score"],
                    result["bm25_rank"],
                    result["semantic_rank"],
                )

            if args.rerank_method is not None:
                print(
                    f"Re-ranking top {len(results)} results "
                    f"using {args.rerank_method} method..."
                )

                match args.rerank_method:
                    case "individual":
                        results = rerank_individual(query, results)
                    case "batch":
                        results = rerank_batch(query, results)
                    case "cross_encoder":
                        results = rerank_cross_encoder(query, results)

            print(
                f"Reciprocal Rank Fusion Results for '{query}' "
                f"(k={args.k}):"
            )

            final_results = results[:args.limit]

            logger.debug(
                "Final results after re-ranking (%s): %d",
                args.rerank_method or "none",
                len(final_results),
            )

            for rank, result in enumerate(final_results, start=1):
                logger.debug(
                    "Final #%d: id=%s title=%r rrf_score=%.6f "
                    "rerank_score=%s rerank_rank=%s cross_encoder_score=%s",
                    rank,
                    result["id"],
                    result["title"],
                    result["rrf_score"],
                    result.get("rerank_score"),
                    result.get("rerank_rank"),
                    result.get("cross_encoder_score"),
                )

            if args.evaluate:
                scores = evaluate_results(query, final_results)

                print()
                for i, (result, score) in enumerate(
                    zip(final_results, scores),
                    start=1,
                ):
                    print(f"{i}. {result['title']}: {score}/3")

            for i, result in enumerate(final_results, start=1):
                bm25_rank = result["bm25_rank"]
                semantic_rank = result["semantic_rank"]

                if bm25_rank is None:
                    bm25_rank = "N/A"
                if semantic_rank is None:
                    semantic_rank = "N/A"

                print(f"\n{i}. {result['title']}")

                match args.rerank_method:
                    case "individual":
                        print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                    case "batch":
                        print(f"   Re-rank Rank: {result['rerank_rank']}")
                    case "cross_encoder":
                        print(
                            f"   Cross Encoder Score: "
                            f"{result['cross_encoder_score']:.3f}"
                        )

                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"   BM25 Rank: {bm25_rank}, "
                    f"Semantic Rank: {semantic_rank}"
                )
                print(f"   {result['document'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()