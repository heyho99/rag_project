# pdf2md

`pdf2md/` は、PDF を以下の3方式でテキスト化（Markdown/プレーンテキスト）して `pdf2md/coverted_texts/` 配下へ保存するためのユーティリティです。

- Gemini（File API）で **PDF→Markdown**
- pdfplumber で **PDF→テキスト抽出（非OCR）**
- tesseract で **PDF→OCRテキスト抽出**

## 前提

- Python 環境
  - リポジトリルートの `venv` を使う想定（例: `. venv/bin/activate`）
- Gemini を使う場合
  - 環境変数 `GEMINI_API_KEY` が必要
- tesseract を使う場合
  - OS に tesseract がインストールされている必要があります（例: `tesseract-ocr`）
  - 日本語OCRを使う場合、`jpn` の言語データが必要です

## 設定（プロジェクトルート `config.py`）

パラメータはプロジェクトルートの `config.py` で一元管理します。

- 入力PDF
  - `PDF_INPUT_GLOB`（例: `docs/*.pdf`）
- 出力先（※ディレクトリ名は `coverted_texts` です）
  - `GEMINI_MD_OUTPUT_DIR`
  - `PDFPLUMBER_TXT_OUTPUT_DIR`
  - `TESSERACT_TXT_OUTPUT_DIR`
- Gemini
  - `GEMINI_PDF2MD_MODEL_NAME`
  - `GEMINI_PDF2MD_TEMPERATURE`
  - `GEMINI_PDF2MD_THINKING_LEVEL`
  - `GEMINI_PDF2MD_MAX_OUTPUT_TOKENS`
  - `GEMINI_PDF2MD_PROMPT`
- tesseract
  - `TESSERACT_OCR_LANG`
  - `TESSERACT_OCR_DPI`
- pdfplumber
  - `PDFPLUMBER_DEBUG_IMAGE_RESOLUTION`

## 実行方法

以下はすべて **リポジトリルート** で実行します。

### 1) Geminiで PDF→Markdown

```bash
python3 -m pdf2md.pdf2md_llm
```

- 出力先: `pdf2md/coverted_texts/gemini_md/`
- PDF 1ファイルにつき Markdown 1ファイルを生成します

### 2) pdfplumberで PDF→テキスト抽出

```bash
python3 -m pdf2md.simpleext_pdfplumber
```

- 出力先: `pdf2md/coverted_texts/pdfplumber_txt/`
- `--debug` を付けると、デバッグ画像を `.../debug/` に出力します

```bash
python3 -m pdf2md.simpleext_pdfplumber --debug
```

### 3) tesseractで PDF→OCRテキスト抽出

```bash
python3 -m pdf2md.ocr_tesseract
```

- 出力先: `pdf2md/coverted_texts/tesseract_txt/`

## 出力ディレクトリ

```text
pdf2md/
└── coverted_texts/
    ├── gemini_md/        # Gemini変換結果（*.md）
    ├── pdfplumber_txt/   # pdfplumber抽出結果（*.txt）
    └── tesseract_txt/    # tesseract OCR結果（*.txt）
```

## 次のステップ（RAG登録・評価）

- 生成した `*.md` / `*.txt` を OpenSearch へ登録する処理は `rag_opensearch/` を使用します。
- ragas評価の流れを整備する場合も、まずは `pdf2md/coverted_texts/` に成果物が揃っている状態を作るのが前提になります。
