PDF_INPUT_GLOB = "docs/*.pdf"

GEMINI_MD_OUTPUT_DIR = "pdf2md/coverted_texts/gemini_md"
PDFPLUMBER_TXT_OUTPUT_DIR = "pdf2md/coverted_texts/pdfplumber_txt"
TESSERACT_TXT_OUTPUT_DIR = "pdf2md/coverted_texts/tesseract_txt"

GEMINI_PDF2MD_MODEL_NAME = "gemini-3-flash-preview"
GEMINI_PDF2MD_TEMPERATURE = 1.0
GEMINI_PDF2MD_THINKING_LEVEL = "HIGH"
GEMINI_PDF2MD_MAX_OUTPUT_TOKENS = 10000

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


TESSERACT_OCR_LANG = "jpn"
TESSERACT_OCR_DPI = 300

PDFPLUMBER_DEBUG_IMAGE_RESOLUTION = 150
