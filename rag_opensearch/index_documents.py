import os
import glob
import json
from pathlib import Path
from typing import List, Optional, Dict
from uuid import uuid4
from dotenv import load_dotenv
from opensearchpy import OpenSearch

from .embedding_models import get_gemini_embedding
from .config import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    EMBEDDING_DIM,
    RAG_INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INDEX_FILE_PATTERNS,
)

# .envファイルを読み込み
load_dotenv()


class DocumentIndexer:
    """ドキュメントインデクサー（OpenSearch + Gemini Embedding）"""
    
    def __init__(
        self,
        index_name: str,
        host: str = OPENSEARCH_HOST,
        port: int = OPENSEARCH_PORT,
        embedding_dim: int = EMBEDDING_DIM,
        splitter = None
    ):
        if not index_name:
            raise ValueError("index_nameは必須です")

        # OpenSearchクライアント初期化
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        
        # Embeddingモデル初期化
        self.embedding_model = get_gemini_embedding(
            output_dimensionality=embedding_dim,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.splitter = splitter
        
        # インデックス作成
        self._create_index()
    
    def _create_index(self):
        """インデックスが存在しない場合は作成"""
        if not self.client.indices.exists(index=self.index_name):
            index_body = {
                'settings': {
                    'index': {
                        'knn': True,
                        'number_of_shards': 2
                    }
                },
                'mappings': {
                    'properties': {
                        'content': {'type': 'text'},
                        'source': {'type': 'keyword'},
                        'filename': {'type': 'keyword'},
                        'chunk_index': {'type': 'integer'},
                        'embedding': {
                            'type': 'knn_vector',
                            'dimension': self.embedding_dim
                        }
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=index_body)
            print(f"インデックス '{self.index_name}' を作成しました")
    
    def load_file_and_register(self, file_path: str, metadata: Optional[dict] = None) -> List[str]:
        """ファイルを読み込んでOpenSearchに登録"""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        # ファイル読み込み
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            with open(file_path_obj, 'r', encoding='shift_jis') as f:
                text = f.read().strip()
        
        if not text:
            print(f"警告: {file_path} からテキストが抽出できませんでした")
            return []
        
        # メタデータ準備
        base_metadata = metadata or {}
        base_metadata.update({
            'source': str(file_path_obj),
            'filename': file_path_obj.name
        })
        
        # テキスト分割（LangChainのsplit_text()を使用してList[str]を取得）
        text_chunks = self.splitter.split_text(text)
        print(f"  分割完了: {len(text_chunks)}個のチャンク")
        
        # 各チャンクを辞書形式に変換
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['uuid'] = str(uuid4())
            chunks.append({
                'content': chunk_text,
                'metadata': chunk_metadata
            })
        
        # OpenSearchに登録
        print(f"  OpenSearchに登録中... ({len(chunks)}個のチャンク)")
        uuids = self._index_documents(chunks)
        
        print(f"登録完了: {file_path_obj.name} から {len(chunks)}個のチャンクを登録")
        return uuids
    
    def _index_documents(self, chunks: List[Dict]) -> List[str]:
        """チャンクをOpenSearchにインデックス"""
        uuids = []
        
        # テキストを抽出してバッチでEmbedding生成
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts) # List[List[float]]
        
        # バルク登録用データ準備
        bulk_data = []
        for chunk, embedding in zip(chunks, embeddings):
            doc_id = chunk['metadata']['uuid']
            uuids.append(doc_id)
            
            bulk_data.append(json.dumps({"index": {"_index": self.index_name, "_id": doc_id}}))
            bulk_data.append(json.dumps({
                'content': chunk['content'],
                'source': chunk['metadata'].get('source', ''),
                'filename': chunk['metadata'].get('filename', ''),
                'chunk_index': chunk['metadata'].get('chunk_index', 0),
                'embedding': embedding
            }))
        
        # バルクインデックス実行
        if bulk_data:
            bulk_string = '\n'.join(bulk_data) + '\n'
            response = self.client.bulk(body=bulk_string)
            if response.get('errors'):
                print(f"  警告: インデックス中にエラーが発生しました")
        
        return uuids


def main():
    """
    python -m index_documents
    """
    # 環境変数チェック
    if not os.getenv("GEMINI_API_KEY"):
        print("エラー: GEMINI_API_KEY環境変数が設定されていません")
        return

    index_name = RAG_INDEX_NAME
    
    # 日本語に合わせたチャンク分割器
    separators=[
        "\n\n",  # 段落区切り（最優先）
        "\n",    # 改行
        "。",    # 日本語の句点
        "！",    # 感嘆符
        "？",    # 疑問符
        " ",     # 空白
        "",      # 文字単位（最後の手段）
    ]
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    # DocumentIndexer初期化（接続情報や次元数は config.py のデフォルトを使用）
    indexer = DocumentIndexer(  # embddingはgemini_embeddingで固定
        index_name=index_name,
        splitter=splitter
    )
    
    # 登録するファイルパターン
    file_patterns = INDEX_FILE_PATTERNS

    print("=== ドキュメント登録システム ===\n")
    
    # ファイル検索
    all_files = []
    for pattern in file_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    all_files = list(set(all_files))
    
    if not all_files:
        print("警告: 処理対象のファイルが見つかりません")
        return
    
    print(f"処理対象ファイル: {len(all_files)}個")
    for f in sorted(all_files):
        print(f"  - {f}")
    
    # ファイルを1つずつ処理
    print("\nファイル処理開始...\n")
    doc_ids = []
    for i, file_path in enumerate(sorted(all_files), 1):
        print(f"[{i}/{len(all_files)}] 処理中: {Path(file_path).name}")
        try:
            file_doc_ids = indexer.load_file_and_register(file_path)
            doc_ids.extend(file_doc_ids)
            print(f"  ✓ 完了\n")
        except Exception as e:
            print(f"  ✗ エラー: {e}\n")
            continue
    
    print(f"全処理完了: 合計 {len(doc_ids)}個のチャンクを登録")


if __name__ == "__main__":
    main()
