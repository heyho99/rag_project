# rag_project

PDFを各種手法でテキスト化し、OpenSearch にインデックス登録して RAG（ハイブリッド検索 + Gemini）で回答生成し、ragas で検索・回答品質を評価するための実験用プロジェクトです。

「PDF変換方式（Gemini / pdfplumber / tesseract）の違いが RAG の精度にどう影響するか」を比較検証できる構成になっています。

## 全体の流れ

```text
docs/*.pdf
   │
   ├─(1) pdf2md/            PDF → Markdown / テキスト
   │        ├── pdf2md_llm.py            Gemini File API → *.md
   │        ├── simpleext_pdfplumber.py  pdfplumber   → *.txt
   │        └── ocr_tesseract.py         tesseract OCR → *.txt
   │                    ↓
   │        pdf2md/coverted_texts/{gemini_md, pdfplumber_txt, tesseract_txt}/
   │                    ↓
   ├─(2) rag_opensearch/    チャンク分割 → Embedding → OpenSearch 登録 → RAG検索/回答
   │        ├── index_documents.py       インデックス作成・登録
   │        └── rag_opensearch.py        RRFハイブリッド検索 + Gemini 回答生成
   │
   └─(3) rag_evaluate/      ragas による評価
            ├── pdf2md_per_pages.py      評価用にページ分割でMD化
            ├── create_testset.py        テストセット（質問+正解）生成
            ├── create_dataset.py        テストセットに RAG を実行して評価用データ作成
            └── evaluate_rag.py          ragas メトリクスで評価
```

## ディレクトリ構成

```text
rag_project/
├── config.py               全モジュール共通の設定ファイル（★ここを編集して挙動を変える）
├── docker-compose.yml      OpenSearch / OpenSearch Dashboards
├── docs/                   入力PDF置き場
├── pdf2md/                 PDF → テキスト変換（詳細は pdf2md/README.md）
├── rag_opensearch/         インデックス登録・RAG検索
├── rag_evaluate/           ragas 評価
└── official_docs/          参照用の公式ドキュメントメモ
```

## 前提

- Python（プロジェクトルートの `venv` を使用する想定）
- Docker / Docker Compose（OpenSearch 用）
- tesseract OCR を使う場合は OS 側に `tesseract-ocr` と日本語データ `jpn` が必要

### 環境変数（`.env` をプロジェクトルートに配置）

```dotenv
GEMINI_API_KEY=...   # PDF変換・Embedding・RAG回答生成に使用
OPENAI_API_KEY=...   # ragas のテストセット生成・評価LLMに使用
```

## セットアップ

```bash
python3 -m venv venv
. venv/bin/activate

pip install -r rag_opensearch/requirements.txt
pip install -r rag_evaluate/requirements.txt
```

### OpenSearch の起動

```bash
docker-compose up -d
```

- OpenSearch: http://localhost:9200
- Dashboards: http://localhost:5601

停止・削除:

```bash
docker-compose stop      # 停止（コンテナは残す）
docker-compose down      # 停止＋コンテナ削除
docker-compose down -v   # ボリュームも含めて完全削除
```

## 使い方

以下のコマンドはすべて **リポジトリルート** で実行します。

### 1. PDF → テキスト変換

`docs/*.pdf`（`config.PDF_INPUT_GLOB`）が対象です。

```bash
python3 -m pdf2md.pdf2md_llm            # Gemini      → pdf2md/coverted_texts/gemini_md/*.md
python3 -m pdf2md.simpleext_pdfplumber  # pdfplumber  → pdf2md/coverted_texts/pdfplumber_txt/*.txt
python3 -m pdf2md.ocr_tesseract         # tesseract   → pdf2md/coverted_texts/tesseract_txt/*.txt
```

詳細は [`pdf2md/README.md`](pdf2md/README.md) を参照してください。

### 2. OpenSearch へインデックス登録

`config.py` の `INDEX_NAME` と `INDEX_FILE_PATTERNS` を、使用する変換方式に合わせて切り替えてから実行します。

```bash
python3 -m rag_opensearch.index_documents
```

- `RecursiveCharacterTextSplitter`（日本語の句点等を区切り文字に設定）でチャンク分割
- Gemini Embedding（`gemini-embedding-001` / 1536次元）でベクトル化
- OpenSearch に本文 + ベクトルを登録

### 3. RAG で質問する

```bash
python3 -m rag_opensearch.rag_opensearch
```

- kNN（ベクトル）と BM25（キーワード）を **RRF** で統合したハイブリッド検索
- 上位 `RAG_TOP_K` 件をコンテキストとして Gemini が回答を生成
- サンプル質問は `rag_opensearch/rag_opensearch.py` の `main()` 内に定義されています

### 4. ragas による評価

```bash
# 4-1. 評価用にPDFをページ分割してMD化（config.PDF2MD_PAGES_PER_CHUNK ページ単位）
python3 -m rag_evaluate.pdf2md_per_pages

# 4-2. テストセット（質問 + reference）を生成 → rag_evaluate/testsets/testset_*.csv
python3 -m rag_evaluate.create_testset

# 4-3. テストセットに RAG を実行して評価用データセットを作成 → rag_evaluate/datasets/
#      事前に config.EVAL_INPUT_CSV を 4-2 で生成したCSVに更新すること
python3 -m rag_evaluate.create_dataset

# 4-4. ragas メトリクスで評価 → rag_evaluate/eval_results/rag_result_*.csv
#      事前に config.EVAL_DATASET_CSV_PATH を 4-3 の出力に更新すること
python3 -m rag_evaluate.evaluate_rag
```

評価メトリクスは `config.EVAL_METRICS` で切り替えます（デフォルトは `llm_context_recall` / `context_entity_recall` / `context_relevance`。コメントアウトされた項目を有効化すると precision 系なども計測できます）。

## 設定（`config.py`）

すべてのパラメータはプロジェクトルートの `config.py` に集約されています。主なものは以下の通りです。

| 区分 | 主な設定 |
| --- | --- |
| 共通 | `PROJECT_ROOT`, `PDF_INPUT_GLOB` |
| PDF変換 | `GEMINI_PDF2MD_MODEL_NAME`, `GEMINI_PDF2MD_PROMPT`, `TESSERACT_OCR_LANG`, `TESSERACT_OCR_DPI`, 各出力ディレクトリ |
| OpenSearch | `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `INDEX_NAME`, `INDEX_FILE_PATTERNS` |
| Embedding/チャンク | `GEMINI_EMBEDDING_MODEL_NAME`, `EMBEDDING_DIM`, `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| RAG | `RAG_TOP_K`, `RRF_RANK_CONSTANT`, `RAG_LLM_MODEL_NAME`, `RAG_LLM_THINKING_LEVEL` |
| 評価 | `CREATE_TESTSET_*`, `EVAL_INPUT_CSV`, `EVAL_DATASET_CSV_PATH`, `EVAL_METRICS`, `EVAL_RESULT_OUTPUT_DIR` |

### 変換方式を切り替えて比較する

`config.py` の以下2つをセットで変更します（コメントアウトされた候補が用意されています）。

```python
INDEX_NAME = "gemini-md"
INDEX_FILE_PATTERNS = [f"{GEMINI_MD_OUTPUT_DIR}/*.md"]
```

方式ごとに別インデックスとして登録できるため、同じテストセットで検索精度を比較できます。
