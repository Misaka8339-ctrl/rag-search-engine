import argparse
from pathlib import Path

from lib.multimodal_search import (
    verify_image_embedding,
    image_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Search CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate an image embedding and print its dimensions",
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file",
    )

    image_search_parser = subparsers.add_parser(
        "image_search",
        help="Search movies using an image",
    )
    image_search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file",
    )


    args = parser.parse_args()

    if args.command is not None:
        if not Path(args.image_path).is_file():
            parser.error(f"Image file not found: {args.image_path}")

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path)

            for i, result in enumerate(results, start=1):
                print(
                    f"{i}. {result['title']} "
                    f"(similarity: {result['score']:.3f})"
                )
                print(f"   {result['description'][:100]}...")
                print()

        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()