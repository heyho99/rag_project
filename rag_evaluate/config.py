
from pathlib import Path


# プロジェクトルート（for_blog ディレクトリ）
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# pdf2md_per_pages 用設定
# =============================================================================

# 入力PDFのパターン（プロジェクトルートからの相対パス）
PDF2MD_INPUT_PATTERN = "docs/*.pdf"

# 出力Markdownのディレクトリ（プロジェクトルートからの相対パス）
# 例: for_blog/rag_evaluate/pdf2md_per_pages/*.md
PDF2MD_OUTPUT_DIR = "rag_evaluate/pdf2md_per_pages"

# 使用するGeminiモデル
PDF2MD_MODEL_NAME = "gemini-2.5-pro"

# LLMパラメータ
PDF2MD_TEMPERATURE = 1.0
PDF2MD_THINKING_BUDGET = 128
PDF2MD_MAX_OUTPUT_TOKENS = 10000

# 1チャンクあたりのページ数
PDF2MD_PAGES_PER_CHUNK = 2


# PDF → Markdown 変換用のデフォルトプロンプト
PDF2MD_PROMPT = """次のPDFの内容を、日本語で読みやすいMarkdownに変換してください。

- 見出し・箇条書き・表などは可能な範囲で構造を保ってください
- 単なる画像だけのページは、わかる範囲で簡単にテキストで説明してください
- 元PDFの改行やページ区切りは、必要な範囲で整形して構いません
"""

