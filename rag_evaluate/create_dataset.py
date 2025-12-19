import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import sys

from rag_evaluate.config import (
    PROJECT_ROOT,
    EVAL_INDEX_NAME,
    EVAL_INPUT_CSV,
    EVAL_TESTSETS_DIR,
    EVAL_DATASET_OUTPUT_DIR,
    EVAL_LLM_MODEL_NAME,
    EVAL_LLM_THINKING_LEVEL,
    EVAL_RAG_METHOD,
    EVAL_RAG_TOP_K,
    EVAL_RAG_RRF_RANK_CONSTANT,
)

# プロジェクトルートをパスに追加
project_root = PROJECT_ROOT
sys.path.insert(0, str(project_root))

from rag_opensearch.rag_opensearch import get_opensearch_rag
from rag_opensearch.llm_models import GeminiRAGModel


def load_testset_csv(csv_path: str) -> List[Dict]:
    """
    CSVファイルからテストセットを読み込む
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        テストデータのリスト
    """
    testset = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # BOMを自動的に処理
        reader = csv.DictReader(f)
        for row in reader:
            testset.append(row)
    
    # デバッグ情報: CSVのカラムと最初の行の内容を表示
    if testset:
        print(f"📋 CSVカラム: {list(testset[0].keys())}")
        print(f"📋 最初の行のuser_input: '{testset[0].get('user_input', 'N/A')}'")
    
    return testset


def run_rag_on_testset(
    index_name: str,
    testset: List[Dict],
    rag_method: str = 'knn',
    top_k: int = 5,
    llm_model = None,
    **rag_kwargs
) -> List[Dict]:
    """
    テストセットの各質問に対してRAGを実行し、回答と検索ドキュメントを追加
    
    Args:
        testset: テストデータのリスト
        rag_method: RAG検索方法 ('knn', 'normalize', 'rrf')
        top_k: 取得するドキュメント数
        **rag_kwargs: RAGの追加パラメータ
        
    Returns:
        回答と検索ドキュメントが追加されたテストデータのリスト
    """
    # RAGシステムの初期化
    rag = get_opensearch_rag(
        index_name=index_name, 
        top_k=top_k,
        llm_model=llm_model,
        **rag_kwargs
    )
    
    results = []
    total = len(testset)
    
    print(f"\n=== RAG実行開始 ===")
    print(f"テストセット数: {total}")
    print(f"RAG方法: {rag_method}")
    print(f"Top-K: {top_k}\n")
    
    for idx, test_item in enumerate(testset, 1):
        question = test_item.get('user_input', '').strip()
        
        # 質問が空の場合はスキップ
        if not question:
            print(f"[{idx}/{total}] ⚠️ 警告: 質問が空です。スキップします。")
            test_item['response'] = "ERROR: 質問が空です"
            test_item['retrieved_contexts'] = json.dumps([], ensure_ascii=False)
            results.append(test_item)
            continue
        
        print(f"[{idx}/{total}] 質問: {question[:50]}...")
        
        try:
            # RAGで回答を取得
            result = rag.answer(question, k=top_k, verbose=False)
            
            # 回答を追加
            test_item['response'] = result['answer']
            
            # 検索されたドキュメントのcontentをリストとして追加
            retrieved_contexts = [doc['content'] for doc in result['sources']]
            # JSON形式で保存（CSVに保存するため）
            test_item['retrieved_contexts'] = json.dumps(retrieved_contexts, ensure_ascii=False)
            
            print(f"  ✓ 完了 (検索ドキュメント数: {len(retrieved_contexts)})")
            
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            test_item['response'] = f"ERROR: {str(e)}"
            test_item['retrieved_contexts'] = json.dumps([], ensure_ascii=False)
        
        results.append(test_item)
    
    print(f"\n=== RAG実行完了 ===\n")
    return results


def save_dataset_csv(dataset: List[Dict], output_dir: str = None):
    """
    データセットをCSVファイルとして保存
    
    Args:
        dataset: 保存するデータセット
        output_dir: 出力ディレクトリ
    """
    if output_dir is None:
        output_dir = project_root / EVAL_DATASET_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    # ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイル名に現在時刻を含める
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"dataset_{current_time}.csv"
    
    # CSVに書き込み
    if dataset:
        fieldnames = list(dataset[0].keys())
        
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f: # BOMつきで出力し、Excelで文字化けしない
            writer = csv.DictWriter(
                f, 
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                doublequote=True
            )
            writer.writeheader()
            writer.writerows(dataset)
        
        print(f"✅ データセットを保存しました: {output_path}")
        print(f"   レコード数: {len(dataset)}")
    else:
        print("⚠️ データセットが空のため保存をスキップしました")


def main():
    """
    メイン処理
    使用例:
        python -m src.ragas.create_dataset
    """
    # 入力CSVファイルのパスを指定
    input_csv_path = project_root / EVAL_INPUT_CSV
    
    # CSVファイルが存在するか確認
    if not input_csv_path.exists():
        print(f"❌ エラー: CSVファイルが見つかりません: {input_csv_path}")
        print("\n利用可能なテストセット:")
        testsets_dir = project_root / EVAL_TESTSETS_DIR
        if testsets_dir.exists():
            for csv_file in testsets_dir.glob("*.csv"):
                print(f"  - {csv_file.name}")
        return
    
    print(f"📄 入力CSV: {input_csv_path}")
    
    # テストセットを読み込み
    testset = load_testset_csv(str(input_csv_path))
    print(f"✓ テストセット読み込み完了: {len(testset)}件\n")

    llm_model = GeminiRAGModel(
        model_name=EVAL_LLM_MODEL_NAME,
        thinking_level=EVAL_LLM_THINKING_LEVEL,
    )

    # RAGを実行
    # rag_method: 'knn', 'normalize', 'rrf' から選択
    dataset = run_rag_on_testset(
        index_name=EVAL_INDEX_NAME,
        testset=testset,
        rag_method=EVAL_RAG_METHOD,
        top_k=EVAL_RAG_TOP_K,
        llm_model=llm_model,
        ## normalize の場合の追加パラメータ例:
        # knn_weight=0.7,
        # bm25_weight=0.3,
        # normalization_technique='min_max',
        # combination_technique='arithmetic_mean'
        # rrf の場合の追加パラメータ例:
        rank_constant=EVAL_RAG_RRF_RANK_CONSTANT,
    )
    
    # データセットを保存
    save_dataset_csv(dataset)


if __name__ == "__main__":
    main()
