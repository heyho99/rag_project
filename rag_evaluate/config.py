
from pathlib import Path


# プロジェクトルート（for_blog ディレクトリ）
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# pdf2md_per_pages 用設定
# =============================================================================

# 入力PDFのパターン（プロジェクトルートからの相対パス）
PDF2MD_INPUT_PATTERN = "docs/*.pdf"

# 使用するGeminiモデル
PDF2MD_MODEL_NAME = "gemini-2.5-pro"

# LLMパラメータ
PDF2MD_TEMPERATURE = 1.0
PDF2MD_THINKING_BUDGET = 128
PDF2MD_MAX_OUTPUT_TOKENS = 10000

# 出力Markdownのディレクトリ（プロジェクトルートからの相対パス）
# 例: for_blog/rag_evaluate/pdf2md_per_pages/*.md
PDF2MD_OUTPUT_DIR = "rag_evaluate/pdf2md_per_pages"

# pdf2md_per_pagesで、マークダウン変換するページの分割単位
PDF2MD_PAGES_PER_CHUNK = 3


# PDF → Markdown 変換用のデフォルトプロンプト
PDF2MD_PROMPT = """次のPDFの内容を、日本語で読みやすいMarkdownに変換してください。

- 見出し・箇条書き・表などは可能な範囲で構造を保ってください
- 単なる画像だけのページは、わかる範囲で簡単にテキストで説明してください
- 元PDFの改行やページ区切りは、必要な範囲で整形して構いません
"""


# =============================================================================
# テストセット生成（create_testset.py）用設定
# =============================================================================

# 生成したテストセットCSVの出力ディレクトリ（PROJECT_ROOT からの相対パス）
CREATE_TESTSET_OUTPUT_DIR = "rag_evaluate/testsets"

# テストセット生成に使う LLM / Embedding モデル
CREATE_TESTSET_LLM_MODEL_NAME = "gpt-5-mini"
CREATE_TESTSET_LLM_TEMPERATURE = 0.2
CREATE_TESTSET_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# 生成するテストセットのサイズ
CREATE_TESTSET_SIZE = 5

# クエリ分布の重み
CREATE_TESTSET_HEADLINE_WEIGHT = 0.5
CREATE_TESTSET_KEYPHRASE_WEIGHT = 0.5

# トランスフォーム用パラメータ
CREATE_TESTSET_HEADLINE_MAX = 20
CREATE_TESTSET_SPLITTER_MAX_TOKENS = 1500

# チャンク済みMarkdownのglobパターン（PROJECT_ROOT からの相対パス）
CREATE_TESTSET_CHUNK_MDS_GLOB = f"{PDF2MD_OUTPUT_DIR}/*.md"


# =============================================================================
# 評価用データセット生成 
# =============================================================================

# RAG 評価で利用する OpenSearch インデックス名
# 例: rag_opensearch/config.py の RAG_INDEX_NAME と揃える
EVAL_INDEX_NAME = "tesseract-txt"

# create_dataset.py で使用するテストセット CSV のデフォルトパス
# PROJECT_ROOT からの相対パスで指定
EVAL_INPUT_CSV = "rag_evaluate/testsets/testset_20251201101757.csv"
EVAL_TESTSETS_DIR = "outputs/testdatas/testsets"

EVAL_DATASET_OUTPUT_DIR = "rag_evaluate/datasets"
EVAL_LLM_MODEL_NAME = "gemini-2.5-pro"
EVAL_LLM_TEMPERATURE = 0.7
EVAL_LLM_MAX_OUTPUT_TOKENS = 10000
EVAL_LLM_THINKING_BUDGET = 128
EVAL_RAG_METHOD = "rrf"
EVAL_RAG_TOP_K = 4
EVAL_RAG_RRF_RANK_CONSTANT = 60


# =============================================================================
# 評価設定
# =============================================================================

EVAL_EVALUATOR_LLM_MODEL_NAME = "gpt-5-mini"
EVAL_EVALUATOR_LLM_TEMPERATURE = 0.2
EVAL_DATASET_CSV_PATH = "rag_evaluate/datasets/dataset_20251201_111112.csv"
EVAL_RESULT_OUTPUT_DIR = "rag_evaluate/eval_results"

# 評価に用いるメトリクス（エイリアス名のリスト）
# 現状は Legacy API の3つを利用
EVAL_METRICS = [
    "llm_context_recall",
    "context_entity_recall",
    # "context_relevance",
]