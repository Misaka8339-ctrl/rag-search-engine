# Webflyx RAG Search Engine

一个基于 Python 的电影搜索与检索增强生成（RAG）项目，围绕本地电影数据集实现关键词检索、语义检索、混合排序、查询增强、重排、评估和自然语言问答。

项目基于 Boot.dev 的 RAG 课程构建，以命令行形式运行，用于学习和实践完整的检索与生成流程。

## 功能

- **关键词检索**：文本预处理、停用词过滤、Porter 词干化、倒排索引、TF-IDF 和 BM25。
- **语义检索**：生成文本向量，通过余弦相似度检索电影。
- **分块检索**：支持按单词或句子切分、重叠分块，以及按电影聚合分块得分。
- **混合检索**：支持归一化后的加权融合与 Reciprocal Rank Fusion（RRF）。
- **查询增强**：使用 LLM 进行拼写纠正、查询改写和关键词扩展。
- **结果重排**：支持逐条 LLM 评分、批量 LLM 排序和本地 CrossEncoder 重排。
- **搜索评估**：支持 Precision@K、Recall@K、F1，以及 LLM 相关性评价。
- **RAG 生成**：支持回答生成、多文档摘要、带引用的回答和单轮问答。
- **多模态查询改写**：结合图片和文字生成电影搜索查询。
- **调试日志**：观察原始查询、增强查询、RRF 候选和最终结果。

## 技术栈

| 组件 | 技术 |
|---|---|
| 语言与命令行 | Python、argparse |
| 依赖管理 | uv |
| 关键词预处理 | NLTK、PorterStemmer |
| 向量运算与存储 | NumPy |
| 文本向量模型 | sentence-transformers/all-MiniLM-L6-v2 |
| 重排模型 | cross-encoder/ms-marco-TinyBERT-L2-v2 |
| LLM 接入 | OpenRouter，通过 OpenAI Python SDK 调用 |
| 环境变量 | python-dotenv |
| 本地缓存 | Pickle、NumPy `.npy`、JSON |

## 工作流程

### 检索与重排

1. 接收用户查询，可选择纠错、改写或扩展。
2. 分别执行 BM25 检索和分块语义检索。
3. 使用加权融合或 RRF 合并两路结果。
4. 可选择对候选结果进行重排。
5. 输出最终结果，可进一步调用 LLM 评价相关性。

### RAG 生成

1. 使用 RRF 检索相关电影。
2. 根据电影 ID 取回完整简介。
3. 将查询、检索资料和任务指令发送给 LLM。
4. 输出回答、摘要或带来源引用的回答。

当前生成命令使用基础 RRF 检索；混合搜索 CLI 中的查询增强和重排选项不会自动应用到这些生成命令。

## 目录结构

```text
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py
│   ├── semantic_search_cli.py
│   ├── hybrid_search_cli.py
│   ├── evaluation_cli.py
│   ├── augmented_generation_cli.py
│   ├── describe_image_cli.py
│   └── lib/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── text_processing.py
│       ├── keyword_search.py
│       ├── semantic_search.py
│       ├── hybrid_search.py
│       ├── query_enhancement.py
│       ├── reranking.py
│       ├── search_evaluation.py
│       └── augmented_generation.py
├── data/                       # 本地数据，不提交到 Git
├── cache/                      # 生成的索引和向量，不提交到 Git
├── test_llm.py
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## 安装

需要安装 Git 和 uv，并能够在首次运行时下载模型。Python 版本与依赖要求以项目配置文件为准。

```bash
git clone https://github.com/Misaka8339-ctrl/rag-search-engine.git
cd rag-search-engine

uv sync
```

后续示例均在项目根目录运行，使用 `uv run` 执行项目环境中的脚本。

如果使用 Boot.dev CLI，可先激活虚拟环境：

```bash
source .venv/bin/activate
```

## 配置 API Key

需要调用 LLM 的功能通过 OpenRouter 提供服务。

创建 `.env.example`：

```dotenv
OPENROUTER_API_KEY=your_api_key_here
```

复制为本地配置：

```bash
cp .env.example .env
```

在 `.env` 中填写自己的有效密钥：

```dotenv
OPENROUTER_API_KEY="your_actual_api_key"
```

确认 `.env` 已加入 `.gitignore`，不要提交真实密钥。

验证 LLM 连接：

```bash
uv run test_llm.py
```

当前代码使用 `openrouter/free`。模型可用性、响应时间和限流情况由服务提供方决定。

BM25、本地语义检索和 CrossEncoder 重排本身不需要 OpenRouter API Key；查询增强、LLM 重排、LLM 评价、RAG 生成和图片查询改写需要。

## 准备数据

从 Boot.dev 对应课程获取数据，并放到：

```text
data/
├── movies.json
├── stopwords.txt
├── golden_dataset.json
└── paddington.jpeg
```

- `movies.json`：检索所需的电影数据。
- `stopwords.txt`：关键词预处理使用的停用词，每行一个。
- `golden_dataset.json`：离线评估使用的查询及相关电影标注。
- `paddington.jpeg`：多模态查询改写示例图片。

`movies.json` 的结构：

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Example Movie",
      "description": "A description of the movie."
    }
  ]
}
```

当前评估脚本使用的黄金数据集结构：

```json
{
  "test_cases": [
    {
      "query": "cute british bear marmalade",
      "relevant_docs": ["Paddington"]
    }
  ]
}
```

数据文件不随仓库分发。运行检索前，请先准备电影数据和停用词。

## 构建索引与向量

构建 BM25 所需的倒排索引、词频和文档长度缓存：

```bash
uv run cli/keyword_search_cli.py build
```

生成分块向量，供混合检索使用：

```bash
uv run cli/semantic_search_cli.py embed_chunks
```

如需使用整篇电影语义检索，生成对应向量：

```bash
uv run cli/semantic_search_cli.py verify_embeddings
```

第一次生成向量可能需要较长时间，后续运行会加载本地缓存。

文本向量模型与 CrossEncoder 是不同模型，需要分别下载。模型文件缓存与项目 `cache/` 中的电影向量缓存也不同。

## 使用示例

### BM25 检索

```bash
uv run cli/keyword_search_cli.py bm25search \
  "animated family" --limit 5
```

### 整篇电影语义检索

```bash
uv run cli/semantic_search_cli.py search \
  "adventures in space" --limit 5
```

### 分块语义检索

```bash
uv run cli/semantic_search_cli.py search_chunked \
  "a bear exploring a magical forest" --limit 5
```

### 加权混合检索

```bash
uv run cli/hybrid_search_cli.py weighted-search \
  "a bear in London" --alpha 0.5 --limit 5
```

两路分数分别归一化后，按以下公式融合：

```text
hybrid_score = alpha × bm25_score + (1 - alpha) × semantic_score
```

`alpha` 越大，关键词检索的权重越高。

### RRF 混合检索

```bash
uv run cli/hybrid_search_cli.py rrf-search \
  "family movie about bears" -k 60 --limit 5
```

RRF 使用各路排名计算贡献，并按电影 ID 合并：

```text
rrf_score = Σ 1 / (k + rank)
```

排名从 1 开始。`-k` 是 RRF 平滑参数，`--limit` 是最终结果数量。

### 查询增强

```bash
# 拼写纠正
uv run cli/hybrid_search_cli.py rrf-search \
  "family moovie about bears" --enhance spell

# 将模糊描述改写成具体查询
uv run cli/hybrid_search_cli.py rrf-search \
  "that bear movie where leo gets attacked" --enhance rewrite

# 在原查询后追加相关词
uv run cli/hybrid_search_cli.py rrf-search \
  "scary bear movie" --enhance expand
```

### 结果重排

```bash
# LLM 逐条评分
uv run cli/hybrid_search_cli.py rrf-search \
  "family movie about bears" --rerank-method individual --limit 5

# LLM 批量排序
uv run cli/hybrid_search_cli.py rrf-search \
  "family movie about bears" --rerank-method batch --limit 5

# 本地 CrossEncoder 重排
uv run cli/hybrid_search_cli.py rrf-search \
  "dinosaur" --rerank-method cross_encoder --limit 5
```

启用重排后，系统先获取最多 `5 × limit` 个 RRF 候选，重排后返回前 `limit` 条。

CrossEncoder 分数可能为负，按原值从高到低排序；它不是百分比，也不是 `0–10` 的评分。

### 调试与 LLM 评价

```bash
uv run cli/hybrid_search_cli.py rrf-search \
  "dinosaur" \
  --rerank-method cross_encoder \
  --limit 5 \
  --debug \
  --evaluate
```

`--debug` 记录搜索各阶段信息。

`--evaluate` 对最终展示的结果进行 LLM 评价：

| 分数 | 含义 |
|---|---|
| 3 | 高度相关 |
| 2 | 相关 |
| 1 | 边缘相关 |
| 0 | 不相关 |

### RAG 回答

```bash
uv run cli/augmented_generation_cli.py rag \
  "Recommend movies about dinosaurs and explain why they fit."
```

### 多文档摘要

```bash
uv run cli/augmented_generation_cli.py summarize \
  "movies about dinosaurs" --limit 5
```

### 带引用的回答

```bash
uv run cli/augmented_generation_cli.py citations \
  "Recommend science fiction movies involving robots." --limit 5
```

引用编号对应本次检索提供的文档顺序，不是电影 ID。

### 单轮问答

```bash
uv run cli/augmented_generation_cli.py question \
  "What is Jurassic Park about?" --limit 5
```

该命令采用自然的对话风格，但不保存多轮聊天历史。

### 多模态查询改写

```bash
uv run cli/describe_image_cli.py \
  --image data/paddington.jpeg \
  --query "What movie is this bear from?"
```

图片以 Base64 Data URL 形式发送给模型，与文本共同用于改写查询。该命令输出改写结果，不会自动执行后续检索。

## 离线评估

准备 `data/golden_dataset.json` 后运行：

```bash
uv run cli/evaluation_cli.py --limit 10
```

每个测试用例输出：

- **Precision@K**：前 K 条结果中，相关结果所占比例。
- **Recall@K**：全部已标注相关电影中，被前 K 条结果找出的比例。
- **F1**：Precision 与 Recall 的调和平均值。

当前评估使用基础 RRF，平滑参数为 `60`，按电影标题匹配相关性标注。它不会自动启用查询增强或重排。

## 缓存与数据更新

项目缓存包括：

```text
cache/
├── index.pkl
├── docmap.pkl
├── term_frequencies.pkl
├── doc_lengths.pkl
├── movie_embeddings.npy
├── chunk_embeddings.npy
└── chunk_metadata.json
```

修改电影数据、文档顺序、模型或预处理规则后，应重新构建对应缓存。

重新构建关键词索引：

```bash
uv run cli/keyword_search_cli.py build
```

重新生成分块向量：

```bash
rm -f cache/chunk_embeddings.npy cache/chunk_metadata.json
uv run cli/semantic_search_cli.py embed_chunks
```

重新生成整篇电影向量：

```bash
rm -f cache/movie_embeddings.npy
uv run cli/semantic_search_cli.py verify_embeddings
```

## 当前限制

- 检索使用本地数据和向量遍历，尚未接入向量数据库或近似最近邻索引。
- 当前按正则识别句子边界，缩写、复杂标点和长句可能影响分块效果。
- 缓存尚未根据数据内容、模型及参数自动进行版本校验。
- 当前重排和 LLM 搜索评价使用的 `document` 通常是简介前 100 个字符；RAG 生成命令会取回完整简介。
- LLM 查询增强、排序、评价和生成可能出现波动或错误。
- 引用由提示词约束，尚未自动验证引用是否真正支持回答。
- 黄金数据集按标题标注，同名电影可能带来歧义。
- 项目目前是课程学习与实验用 CLI，尚未提供 Web API、访问控制或生产部署方案。

## 课程测试

在对应课程步骤中，使用 Boot.dev CLI 运行并提交测试：

```bash
bootdev run -s
```

课程测试通过不等于所有查询都具有良好效果。搜索质量需要结合黄金数据集、调试日志和实际查询持续评估。