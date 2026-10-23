import argparse

from lib.augmented_generation import (
    rag,
    summarize, 
    citations,
    answer_question,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)",
    )
    rag_parser.add_argument(
        "query",
        type=str,
        help="Search query for RAG",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize multiple movie search results",
    )
    summarize_parser.add_argument(
        "query",
        type=str,
        help="Search query for summarization",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize",
    )

    citations_parser = subparsers.add_parser(
        "citations",
        help="Generate an answer with source citations",
    )
    citations_parser.add_argument(
        "query",
        type=str,
        help="Search query for a citation-aware answer",
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of source documents to retrieve",
    )

    question_parser = subparsers.add_parser(
    "question",
    help="Answer a question using retrieved movie information",
    )
    question_parser.add_argument(
        "question",
        type=str,
        help="Question about movies",
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of movie documents to retrieve",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            if not args.query.strip():
                parser.error("query must not be empty")

            results, answer = rag(args.query)

            print("Search Results:")
            for result in results:
                print(f"- {result['title']}")

            print("\nRAG Response:")
            print(answer)

        case "summarize":
            if not args.query.strip():
                parser.error("query must not be empty")

            if args.limit <= 0:
                parser.error("--limit must be greater than 0")

            results, summary = summarize(args.query, args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            if not args.query.strip():
                parser.error("query must not be empty")

            if args.limit <= 0:
                parser.error("--limit must be greater than 0")

            results, answer = citations(args.query, args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            print("\nLLM Answer:")
            print(answer)

        case "question":
            if not args.question.strip():
                parser.error("question must not be empty")

            if args.limit <= 0:
                parser.error("--limit must be greater than 0")

            results, answer = answer_question(
                args.question,
                args.limit,
            )

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()