## はじめに

RAG（Retrieval-Augmented Generation）システムを作ったあとに、
「**どのくらいちゃんと動いているのか？**」「**どこを改善すべきか？**」を定量的に知りたいと思い、
Ragas を使って評価パイプラインを組んでみました。

この記事では、前回のハイブリッドRAG（OpenSearch + Gemini）の実装を前提に、

- **何を使ったか**（Ragas / OpenAI / Gemini / 既存の RAG 実装）
- **どう動かすか**（コマンドと実行の順番）
- **どんな結果が出るか**（メトリクス例）

だけをシンプルにまとめます。

この記事は **前回記事（qiita_rag.md）を読んだあと**に見る想定です。
先に `rag_opensearch/` を動かしてインデックス作成まで済んでいる状態を前提にしています。

- リポジトリ: https://github.com/heyho99/rag_project

まだ clone していない場合は、まずここからです：

```bash
git clone https://github.com/heyho99/rag_project
cd rag_project
```

特に、次の 4 ステップを **この順番で必ず実行する必要がある** ので、その点を明示しておきます。

1. [pdf2md_per_pages.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages.py:0:0-0:0) で PDF → Markdown チャンクを作る  
2. [create_testset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_testset.py:0:0-0:0) でテストセット（質問＋理想回答）を作る  
3. [create_dataset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_dataset.py:0:0-0:0) で RAG を実行した結果付きデータセットを作る  
4. [evaluate_rag.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/evaluate_rag.py:0:0-0:0) で Ragas による評価を行う  


## 目次

- [やりたいこと](#やりたいこと)
- [Ragas とは](#ragas-とは)
- [ディレクトリ構成](#ディレクトリ構成)
- [全体の流れと実行順序](#全体の流れと実行順序)
- [セットアップ](#セットアップ)
- [Step 1: PDF → Markdown チャンク生成 (pdf2md_per_pages)](#step-1-pdf--markdown-チャンク生成-pdf2md_per_pages)
- [Step 2: テストセット生成 (create_testset.py)](#step-2-テストセット生成-create_testsetpy)
- [Step 3: 評価用データセット作成 (create_dataset.py)](#step-3-評価用データセット作成-create_datasetpy)
- [Step 4: RAG を Ragas で評価 (evaluate_rag.py)](#step-4-rag-を-ragas-で評価-evaluate_ragpy)
- [実行結果の例](#実行結果の例)
- [重要なポイント](#重要なポイント)
- [以下実装内容をもう少し解説](#以下実装内容をもう少し解説)
- [まとめ](#まとめ)


## やりたいこと

- 既存の RAG（OpenSearch + Gemini）に対して、**Ragas で自動評価を回せる状態**を作る
- 評価のためのデータを、できるだけ自動で生成したい  
  - PDF からチャンク化した Markdown を作る  
  - そこから「質問＋理想回答＋参照コンテキスト」を自動生成  
  - さらに実際の RAG の回答を自動で付けて、評価入力を整形
- 最終的に、Ragas のメトリクス（Context Recall / Context Relevance など）で  
  「この RAG はどれくらい良さそうか？」の目安を取れるようにする


## Ragas とは

https://docs.ragas.io/en/stable/

- RAG システム向けの **評価ライブラリ**
- 単なる BLEU/ROUGE ではなく、RAG 特有の観点を評価できるメトリクスを持っている  
  - コンテキストがちゃんと取れているか（Context Recall）  
  - 取ってきたコンテキストが質問に関係あるか（Context Relevance）  
  - 回答がコンテキストにちゃんと根拠付けられているか（Groundedness）
- LLM を評価器として利用するタイプのメトリクスが多いので、  
  評価用の LLM（この記事では `gpt-5-mini`）の API キーが必要になります。


## ディレクトリ構成

RAG 本体（前回記事の `rag_opensearch/`）に加えて、評価用のコードは [rag_evaluate/](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate:0:0-0:0) にまとめています。

```text
rag_project/
├─ docs/                       # 入力PDFを置くディレクトリ
├─ rag_opensearch/             # 前回記事のハイブリッドRAG実装
├─ rag_evaluate/
│  ├─ config.py                # 評価・テストセット・PDF→MD変換の設定値
│  ├─ pdf2md_per_pages.py      # PDFをページ単位で分割し、LLMでMarkdown化
│  ├─ pdf2md_per_pages/        # 生成されたMarkdownチャンク(.md)が入る
│  ├─ create_testset.py        # MarkdownチャンクからRagas用テストセットCSVを生成
│  ├─ testsets/                # 生成されたテストセットCSVの一時置き場
│  ├─ create_dataset.py        # テストセットに対して実際にRAGを実行し、評価データセットを生成
│  ├─ datasets/                # 評価用データセットCSVの出力先
│  ├─ evaluate_rag.py          # Ragasでメトリクスを計算
│  ├─ eval_results/            # 評価結果CSVの出力先
│  └─ llm_models.py            # Gemini / OpenAI を使うためのLLMラッパー
└─ （その他のディレクトリ）
```


## 全体の流れと実行順序

この評価パイプラインは、**必ず次の順番で実行する前提**で設計しています。

1. **Step 1: PDF → Markdown チャンク生成（[pdf2md_per_pages.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages.py:0:0-0:0)）**  
   - 入力: `docs/*.pdf`  
   - 出力: `rag_evaluate/pdf2md_per_pages/*.md`
2. **Step 2: テストセット生成（[create_testset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_testset.py:0:0-0:0)）**  
   - 入力: Step 1 の Markdown チャンク（`PDF2MD_OUTPUT_DIR`）  
   - 出力: `rag_evaluate/testsets/testset_YYYYMMDDHHMMSS.csv`
3. **Step 3: 評価用データセット作成（[create_dataset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_dataset.py:0:0-0:0)）**  
   - 入力: Step 2 のテストセット CSV（`EVAL_INPUT_CSV`）  
   - RAG 実行: `rag_opensearch` の RAG 実装に問い合わせ  
   - 出力: `rag_evaluate/datasets/dataset_YYYYMMDD_HHMMSS.csv`
4. **Step 4: RAG を Ragas で評価（[evaluate_rag.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/evaluate_rag.py:0:0-0:0)）**  
   - 入力: Step 3 のデータセット CSV（`EVAL_DATASET_CSV_PATH`）  
   - 出力: メトリクス平均＋詳細 CSV（`rag_evaluate/eval_results/*.csv`）

途中の CSV パスはすべて [rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) に集約しているため、  
**Step 2 の出力 → Step 3 の入力**、**Step 3 の出力 → Step 4 の入力** を  
そこに書き換えるだけでパイプラインを繋げられるようにしています。


## セットアップ

### 前提

- Docker / Python 3 系環境
- OpenSearch + Gemini を使った RAG 環境（前回記事の `rag_opensearch/`）が動く状態
- Google AI Studio で取得した **Gemini API キー**（PDF→MD と RAG 用）
- OpenAI の **API キー**（Ragas 評価用 LLM / Embedding 用）

### 1. 仮想環境と依存パッケージ

前回記事と同様、リポジトリ直下で仮想環境を作り、依存を入れます。
（前回記事で clone 済みなら `git clone` / `cd` はスキップでOKです）

```bash
git clone https://github.com/heyho99/rag_project
cd rag_project

python -m venv venv
source venv/bin/activate  # Windowsなら: venv\Scripts\activate

pip install -r rag_evaluate/requirements.txt
```

### 2. OpenSearch と RAG 本体を起動

ハイブリッドRAG本体（`rag_opensearch/`）の方で、OpenSearch などを起動しておきます。

```bash
docker compose -f docker-compose.yml up -d

python -m rag_opensearch.index_documents      # まだならインデックスを作成
python -m rag_opensearch.rag_opensearch       # RAG本体の動作確認（任意）
```

### 3. API キーの設定

プロジェクトルートに `.env` を置きます。

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. [rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) を確認

特に次のあたりを自分の環境に合わせておきます。

```python
PDF2MD_MODEL_NAME = "gemini-3-flash-preview"
PDF2MD_THINKING_LEVEL = "HIGH"

PDF2MD_INPUT_PATTERN = "docs/*.pdf"                 # 評価対象PDF
PDF2MD_OUTPUT_DIR = "rag_evaluate/pdf2md_per_pages" # Markdownチャンク出力先

CREATE_TESTSET_OUTPUT_DIR = "rag_evaluate/testsets"
CREATE_TESTSET_CHUNK_MDS_GLOB = f"{PDF2MD_OUTPUT_DIR}/*.md"

EVAL_INDEX_NAME = "tesseract-txt"                   # RAGで使うOpenSearchインデックス
EVAL_INPUT_CSV = "rag_evaluate/testsets/testset_....csv"

EVAL_DATASET_OUTPUT_DIR = "rag_evaluate/datasets"
EVAL_LLM_MODEL_NAME = "gemini-3-flash-preview"
EVAL_LLM_THINKING_LEVEL = "HIGH"
EVAL_DATASET_CSV_PATH = "rag_evaluate/datasets/dataset_....csv"

EVAL_RESULT_OUTPUT_DIR = "rag_evaluate/eval_results"
```

Step 2 / Step 3 を実行するたびに CSV のファイル名が変わるので、  
**最新の CSV パスに合わせて `EVAL_INPUT_CSV` / `EVAL_DATASET_CSV_PATH` を更新する**運用を想定しています。


## Step 1: Ragasでテストセットを作成するためのMDファイルを作成 (pdf2md_per_pages)

Ragasではテストセットを自動で作成できますが、**複数のある程度の文字数のマークダウンファイル**があると、きれいに作成されます。
よって、最初にソースPDFを適当なページ数で区切って、マークダウンファイルに分割します。

まず、評価対象となる PDF（例: 行政レポートや観光動向調査など）を `docs/` 配下に置きます。

そしてこれをnページごとに取り出して、nページごとのマークダウンを生成します。

実行コマンド（プロジェクトルートで実行）:

```bash
python -m rag_evaluate.pdf2md_per_pages
```

内部では、

- `pypdf` で PDF を数ページごとのチャンクに分割
- 各チャンクを Gemini の File API + LLM に投げて Markdown に変換
- タイムスタンプ付きの `.md` ファイルとして `PDF2MD_OUTPUT_DIR` に保存

という流れになっています。


## Step 2: テストセット生成 (create_testset.py)

次に、Step 1 で作った Markdown チャンクから、  
Ragas のテストセット（**質問＋理想回答＋参照コンテキスト**）を自動生成します。

- 入力: `CREATE_TESTSET_CHUNK_MDS_GLOB` に一致する Markdown  
- 出力: `rag_evaluate/testsets/testset_YYYYMMDDHHMMSS.csv`

実行コマンド:

```bash
python -m rag_evaluate.create_testset
```

[create_testset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_testset.py:0:0-0:0) の中では、ざっくり次のような処理をしています。

- [pdf2md_per_pages/](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages:0:0-0:0) 以下の Markdown を読み込み、Knowledge Graph を構築
- LLM（`gpt-5-mini`）と Embedding（`text-embedding-3-small`）を使って  
  - 見出し抽出  
  - 見出しベースの分割  
  - キーフレーズ抽出
- それをもとに、ペルソナごとの具体的な質問と、理想回答・参照コンテキストを生成

生成された CSV は、次のようなカラムを持ちます（一例）：

- `user_input`: ユーザーの質問
- `reference`: 理想的な回答（ゴールドアンサー）
- `reference_contexts`: 回答の根拠になるコンテキスト群

この CSV のパスを、[rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) の `EVAL_INPUT_CSV` に設定しておきます。


## Step 3: 評価用データセット作成 (create_dataset.py)

テストセットができたら、実際の RAG システム（`rag_opensearch/`）に対して  
各 `user_input` を投げ、**「RAG が返した回答」と「取得コンテキスト」** を付け足したデータセットを作ります。

- 入力: `EVAL_INPUT_CSV` （Step 2 で生成したテストセット）  
- RAG 実行先: `EVAL_INDEX_NAME`（例: `tesseract-txt`）  
- 出力: `rag_evaluate/datasets/dataset_YYYYMMDD_HHMMSS.csv`

RAG の回答生成に使う Gemini のモデル名や思考レベルは、[rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) の
`EVAL_LLM_MODEL_NAME` / `EVAL_LLM_THINKING_LEVEL` で切り替えます。

実行コマンド:

```bash
python -m rag_evaluate.create_dataset
```

[create_dataset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_dataset.py:0:0-0:0) では、次のようなことをしています。

- テストセット CSV を読み込み（BOM 付き UTF-8 に対応）
- `GeminiRAGModel` と `get_opensearch_rag()` を使って、各 `user_input` に対して RAG を実行  
  - `EVAL_RAG_METHOD` で RAG の検索方式（`knn` / `normalize` / `rrf`）を選択  
  - `EVAL_RAG_TOP_K` で取得チャンク数を指定  
  - `EVAL_RAG_RRF_RANK_CONSTANT` で RRF のパラメータを設定
- 実行結果として  
  - `response`: 実際の RAG の回答  
  - `retrieved_contexts`: 取得されたコンテキスト（テキストだけを JSON 文字列として保存）

最終的に、Ragas の `EvaluationDataset` にそのまま渡せる形の CSV を [datasets/](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/datasets:0:0-0:0) に保存します。

この CSV のパスを、[rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) の `EVAL_DATASET_CSV_PATH` に設定します。


## Step 4: RAG を Ragas で評価 (evaluate_rag.py)

最後に、Step 3 で作ったデータセットを Ragas に渡して、  
コンテキスト関連のメトリクスをまとめて計算します。

- 入力: `EVAL_DATASET_CSV_PATH`（Step 3 の出力 CSV）  
- 出力:  
  - コンソール: 各メトリクスの平均値  
  - CSV: 行ごとのスコアを含む詳細結果（`rag_evaluate/eval_results/*.csv`）

実行コマンド:

```bash
python -m rag_evaluate.evaluate_rag
```

[evaluate_rag.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/evaluate_rag.py:0:0-0:0) では、

- `.env` から `OPENAI_API_KEY` を読み込み
- `LangchainLLMWrapper(ChatOpenAI(...))` で評価用 LLM を構築
- `EVAL_METRICS` に列挙したメトリクスを `METRIC_REGISTRY` 経由でインスタンス化
- `EvaluationDataset.from_list(...)` で CSV からデータセットを構築
- `ragas.evaluate()` を呼び出してメトリクスを一括計算

という流れになっています。

デフォルトでは、次の 3 つのメトリクスを使っています。

- `llm_context_recall`
- `context_entity_recall`
- `context_relevance`

他のメトリクスも `EVAL_METRICS` に名前を追加するだけで有効化できるようにしてあります。


## 実行結果の例

実際のログやメトリクスの出力は環境によって変わりますが、イメージとしては次のような形になります。

### 1. create_dataset 実行ログの例

```text
📄 入力CSV: /home/ouchi/rag_project/rag_evaluate/testsets/testset_20251201101757.csv
✓ テストセット読み込み完了: 5件

=== RAG実行開始 ===
テストセット数: 5
RAG方法: rrf
Top-K: 4

[1/5] 質問: 保険会社は持続可能なビジネスモデルを構築するために、どのような取組みを求...
  ✓ 完了 (検索ドキュメント数: 4)

[2/5] 質問: 令和5年度神戸市観光動向調査において、女性比率が高い地区はどこですか...
  ✓ 完了 (検索ドキュメント数: 4)

...

=== RAG実行完了 ===
✅ データセットを保存しました: /home/ouchi/rag_project/rag_evaluate/datasets/dataset_20251201_111112.csv
   レコード数: 5
```

### 2. evaluate_rag 実行結果の例

```text
EvaluationResult(
  metrics=['llm_context_recall', 'context_entity_recall', 'context_relevance'],
  scores={
    'llm_context_recall': 0.82,
    'context_entity_recall': 0.76,
    'context_relevance': 0.88,
  }
)

結果CSV: rag_evaluate/eval_results/rag_result_20251201XXXXXX.csv に詳細を保存しました
```

CSV 側では、各質問ごとに個別スコアが出ているので、  
「特定の質問タイプだけスコアが悪い」といった傾向も追いやすくなります。


## 重要なポイント

- **4 ステップの実行順序が前提になっている**  
  - [pdf2md_per_pages](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages:0:0-0:0) → `create_testset` → `create_dataset` → `evaluate_rag`  
  - どこかをスキップすると、後続ステップの入力 CSV が存在せずにエラーになります
- PDF → Markdown チャンク → テストセット → 評価データセット → 評価結果、という  
  一連の成果物がすべて **CSV/MD としてファイルに残る** ので、途中で中身を確認しやすい
- 設定値（インデックス名・モデル名・ファイルパス）は **すべて [rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) に寄せている**  
  - 実装側は「config から読むだけ」にし、実験時は [rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) を書き換える運用
- 評価用 LLM（`gpt-5-mini`）と RAG 用 LLM（`gemini-3-flash-preview`）を役割分担させている  
  - 回答を生成する LLM と、それを評価する LLM を分けることで、バイアスをある程度避ける狙い


## 以下実装内容をもう少し解説

ここからは、記事中で触れたスクリプトの中身を、コード抜粋ベースで少しだけ補足します。  
（詳細は GitHub の該当ファイルを参照してください。）

### 1. PDF → Markdown 変換（[pdf2md_per_pages.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages.py:0:0-0:0)）

- [split_pdf_into_chunks()](cci:1://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages.py:39:0-85:26) で PDF を `PDF2MD_PAGES_PER_CHUNK` ページごとの一時 PDF に分割
- 各チャンクを `GeminiPDFConverterModel` に渡して Markdown を生成
- タイムスタンプ付きファイル名＋ページ範囲付きで保存

ページ数が多い PDF でも、ページ単位で LLM に投げることで、  
トークン上限に引っかかりにくくしています。

### 2. テストセット生成（[create_testset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_testset.py:0:0-0:0)）

- `KnowledgeGraph` + 各種 Transform（見出し抽出・キーフレーズ抽出）で  
  ドキュメント構造を Ragas にとって扱いやすいグラフに変換
- 複数のペルソナ（行政アナリスト、保険会社経営企画担当、観光政策担当など）を定義し、  
  人物像に応じた質問を生成

「どんなユーザーが、どんな視点でこの資料を読むのか？」を  
ペルソナとして明示することで、より現実に近い質問セットを作る狙いがあります。

### 3. データセット作成（[create_dataset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_dataset.py:0:0-0:0)）

- `get_opensearch_rag()` で前回記事の RAG 実装を再利用
- [run_rag_on_testset()](cci:1://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_dataset.py:55:0-126:18) でテストセットの各行に対して RAG を実行し、  
  - `response`（RAG の回答）  
  - `retrieved_contexts`（取得されたコンテキスト）  
  を追記

エラー時には `ERROR: ...` を詰めつつ処理を継続するようになっているので、  
一部のクエリで失敗しても全体のバッチが止まらないようにしています。

### 4. RAG 評価（[evaluate_rag.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/evaluate_rag.py:0:0-0:0)）

- `METRIC_REGISTRY` で、文字列のメトリクス名 → 実際のクラス  
  のマッピングを管理
- `EVAL_METRICS` に指定した名前だけを [build_metrics()](cci:1://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/evaluate_rag.py:78:0-85:18) でインスタンス化
- 新しいメトリクスを試したいときは、  
  - `METRIC_REGISTRY` にエイリアスを追加  
  - `EVAL_METRICS` にその名前を入れる  
  だけで評価対象に含められるようにしてあります。


## まとめ

- RAG システムを作ったあとに、「どれくらい良く動いているか？」を  
  **Ragas で定量的に評価するためのパイプライン**を作りました。
- パイプラインは  
  1. PDF → Markdown チャンク ([pdf2md_per_pages.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/pdf2md_per_pages.py:0:0-0:0))  
  2. テストセット生成 ([create_testset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_testset.py:0:0-0:0))  
  3. 評価用データセット作成 ([create_dataset.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/create_dataset.py:0:0-0:0))  
  4. RAG 評価 ([evaluate_rag.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/evaluate_rag.py:0:0-0:0))  
  の 4 ステップで構成されており、**この順番で実行する前提**です。
- 途中の成果物はすべてファイルとして残しているので、  
  「どの質問で RAG が弱いのか」「どのメトリクスが特に低いのか」などを、  
  ログや CSV を見ながら後から振り返りやすい構成にしています。

あとは、使いたいメトリクスや評価用 LLM を [rag_evaluate/config.py](cci:7://file://wsl.localhost/Ubuntu/home/ouchi/rag_project/rag_evaluate/config.py:0:0-0:0) や `EVAL_METRICS` で入れ替えつつ、  
自分のデータセット・自分の RAG 実装に合わせてカスタマイズしていく想定です。
