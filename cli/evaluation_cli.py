import argparse
import json

from lib.config import PROJECT_ROOT
from lib.data import load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    if limit <= 0:
        parser.error("--limit must be greater than 0")

    dataset_path = PROJECT_ROOT / "data" / "golden_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    documents = load_movies()

    # 所有测试用例共用同一个搜索实例，避免反复加载模型
    search = HybridSearch(documents)

    print(f"k={limit}")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        relevant_set = set(relevant_titles)

        # 当前 rrf_search 返回全部排序候选，必须切片取前 K 个
        results = search.rrf_search(
            query,
            k=60,
            limit=limit,
        )[:limit]

        retrieved_titles = [result["title"] for result in results]

        retrieved_set = set(retrieved_titles)
        relevant_set = set(relevant_titles)
        relevant_count = len(retrieved_set & relevant_set)

        precision = relevant_count / limit
        recall = (
            relevant_count / len(relevant_set)
            if relevant_set
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")


if __name__ == "__main__":
    main()