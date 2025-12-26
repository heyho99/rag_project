"""
RAG Project 共通設定ファイル

全モジュール（pdf2md, rag_opensearch, rag_evaluate）がこのファイルを参照します。
"""

from pathlib import Path


# =============================================================================
# プロジェクト共通設定
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# 入力PDFのパターン（プロジェクトルートからの相対パス）
PDF_INPUT_GLOB = "docs/*.pdf"


# =============================================================================
# PDF変換設定（pdf2md）
# =============================================================================

# --- 出力ディレクトリ ---
GEMINI_MD_OUTPUT_DIR = "pdf2md/coverted_texts/gemini_md"
PDFPLUMBER_TXT_OUTPUT_DIR = "pdf2md/coverted_texts/pdfplumber_txt"
TESSERACT_TXT_OUTPUT_DIR = "pdf2md/coverted_texts/tesseract_txt"

# --- Gemini PDF→MD 変換 ---
GEMINI_PDF2MD_MODEL_NAME = "gemini-3-flash-preview"
GEMINI_PDF2MD_TEMPERATURE = 1.0
GEMINI_PDF2MD_THINKING_LEVEL = "HIGH"
GEMINI_PDF2MD_MAX_OUTPUT_TOKENS = 100000

GEMINI_PDF2MD_PROMPT = """あなたはPDFのテキストを正確なマークダウン形式に変換する専門家です。
以下のルールに従って変換してください：

1. **構造の保持**: 元の文書の階層構造を正確に反映
2. **見出しの変換**: 適切なマークダウンヘッダー（#, ##, ###等）を使用
3. **リストの変換**: 箇条書きや番号付きリストを正確に変換
4. **表の変換**: 表は適切なマークダウン表形式に変換
5. **強調の変換**: 太字や斜体を適切なマークダウン記法に変換
6. **コードブロック**: コードや特殊な書式は適切にマークダウン化
7. **改行とスペース**: 読みやすさを保つために適切な改行とスペースを使用
8. **日本語対応**: 日本語の文書構造を理解して適切に変換

変換後は、元の情報を失わず、かつマークダウンとして正しく表示されるようにしてください。"""

# --- pdfplumber ---
PDFPLUMBER_DEBUG_IMAGE_RESOLUTION = 300

# --- Tesseract OCR ---
TESSERACT_OCR_LANG = "jpn"
TESSERACT_OCR_DPI = 300

# =============================================================================
# OpenSearch / RAG設定（rag_opensearch）
# =============================================================================

# --- OpenSearch接続 ---
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200

# --- Embedding ---
EMBEDDING_DIM = 1536
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-001"

# --- チャンク分割 ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_OUTPUT_DIR = "outputs/chunks"

# --- RAG検索 ---
RAG_TOP_K = 2
RRF_RANK_CONSTANT = 60

# --- RAG用LLM ---
RAG_LLM_MODEL_NAME = "gemini-3-pro-preview"
RAG_LLM_THINKING_LEVEL = "LOW"

# 使用するインデックス名（変換方式に合わせて変更）
# INDEX_NAME = "tesseract-txt"
# INDEX_NAME = "pdfplumber-txt"
INDEX_NAME = "gemini-md"

# インデックス対象ファイルパターン（INDEX_NAMEに合わせて変更）
# INDEX_FILE_PATTERNS = ["rag_opensearch/ocr_tesseract/*.txt"]
# INDEX_FILE_PATTERNS = [f"{TESSERACT_TXT_OUTPUT_DIR}/*.txt"]
# INDEX_FILE_PATTERNS = [f"{PDFPLUMBER_TXT_OUTPUT_DIR}/*.txt"]
INDEX_FILE_PATTERNS = [f"{GEMINI_MD_OUTPUT_DIR}/*.md"]


# =============================================================================
# テストセット生成設定（rag_evaluate/create_testset.py）
# =============================================================================

# --- pdf2md_per_pages用（テストセット作成のページ分割変換） ---
PDF2MD_PER_PAGES_OUTPUT_DIR = "rag_evaluate/pdf2md_per_pages"
PDF2MD_PAGES_PER_CHUNK = 3
PDF2MD_PER_PAGES_PROMPT = """次のPDFの内容を、日本語で読みやすいMarkdownに変換してください。

- 見出し・箇条書き・表などは可能な範囲で構造を保ってください
- 単なる画像だけのページは、わかる範囲で簡単にテキストで説明してください
- 元PDFの改行やページ区切りは、必要な範囲で整形して構いません
"""

# --- テストセット生成LLM/Embedding ---
CREATE_TESTSET_OUTPUT_DIR = "rag_evaluate/testsets"
CREATE_TESTSET_LLM_MODEL_NAME = "gpt-5-mini"
CREATE_TESTSET_LLM_TEMPERATURE = 0.2
CREATE_TESTSET_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# --- テストセット生成パラメータ ---
CREATE_TESTSET_SIZE = 20  # 生成するテストセットの数
CREATE_TESTSET_HEADLINE_WEIGHT = 0.5
CREATE_TESTSET_KEYPHRASE_WEIGHT = 0.5
CREATE_TESTSET_HEADLINE_MAX = 20
CREATE_TESTSET_SPLITTER_MAX_TOKENS = 1500
CREATE_TESTSET_CHUNK_MDS_GLOB = f"{PDF2MD_PER_PAGES_OUTPUT_DIR}/*.md"


# --- テストから生成するデータセット設定 ---
EVAL_INPUT_CSV = "rag_evaluate/testsets/testset_20251226101134.csv" # 生成したテストセット
EVAL_DATASET_OUTPUT_DIR = "rag_evaluate/datasets"
EVAL_LLM_MODEL_NAME = "gemini-3-pro-preview" # 回答生成LLMモデル
EVAL_LLM_TEMPERATURE = 0.7
EVAL_LLM_MAX_OUTPUT_TOKENS = 10000
EVAL_LLM_THINKING_LEVEL = "LOW"
EVAL_RAG_METHOD = "rrf"

# --- 評価実行 ---
EVAL_EVALUATOR_LLM_MODEL_NAME = "gpt-5-mini"
EVAL_EVALUATOR_LLM_TEMPERATURE = 0.2
EVAL_DATASET_CSV_PATH = "rag_evaluate/datasets/dataset_gemini.csv"
EVAL_RESULT_OUTPUT_DIR = "rag_evaluate/eval_results"

# --- 評価メトリクス ---
EVAL_METRICS = [
    "llm_context_recall",
    "context_entity_recall",
    "context_relevance",
    # "llm_context_precision_without_reference",
    # "llm_context_precision_with_reference",
    # "nonllm_context_precision_with_reference",
    # "nonllm_context_recall",
    # "legacy.noise_sensitivity",
    # "legacy.answer_accuracy",
    # "response_groundedness",
]
