PDF_INPUT_GLOB = "docs/*.pdf"

GEMINI_MD_OUTPUT_DIR = "pdf2md/coverted_texts/gemini_md"
PDFPLUMBER_TXT_OUTPUT_DIR = "pdf2md/coverted_texts/pdfplumber_txt"
TESSERACT_TXT_OUTPUT_DIR = "pdf2md/coverted_texts/tesseract_txt"

GEMINI_PDF2MD_MODEL_NAME = "gemini-3-flash-preview"
GEMINI_PDF2MD_TEMPERATURE = 1.0
GEMINI_PDF2MD_THINKING_LEVEL = "HIGH"
GEMINI_PDF2MD_MAX_OUTPUT_TOKENS = 10000

GEMINI_PDF2MD_PROMPT = """次のPDFの内容を、日本語で読みやすいMarkdownに変換してください。

- 見出し・箇条書き・表などは可能な範囲で構造を保ってください
- 単なる画像だけのページは、わかる範囲で簡単にテキストで説明してください
- 元PDFの改行やページ区切りは、必要な範囲で整形して構いません
"""

TESSERACT_OCR_LANG = "jpn"
TESSERACT_OCR_DPI = 300

PDFPLUMBER_DEBUG_IMAGE_RESOLUTION = 150
