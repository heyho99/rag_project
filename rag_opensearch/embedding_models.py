import os
import time
import numpy as np
from tqdm import tqdm
from typing import List, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .config import GEMINI_EMBEDDING_MODEL_NAME, EMBEDDING_DIM

# .envファイルを読み込み
load_dotenv()


class GeminiEmbedding:
    """Gemini Embedding モデル（LangChain不使用）"""
    
    def __init__(
        self, 
        model: str = GEMINI_EMBEDDING_MODEL_NAME,
        output_dimensionality: int = EMBEDDING_DIM,
        task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """
        Args:
            model: 使用するモデル名
            output_dimensionality: 出力次元数（推奨: 768, 1536, 3072。デフォルト: 3072）
            task_type: タスクタイプ（RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITYなど）
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY環境変数が設定されていません")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type
    
    # 検索で使う
    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        """単一テキストのEmbeddingを生成"""
        if not text or not text.strip():
            raise ValueError("テキストが空です。埋め込みを生成できません。")
        
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type or self.task_type,
                output_dimensionality=self.output_dimensionality
            )
        )
        
        embedding = result.embeddings[0].values
        
        # 3072次元以外は正規化が必要
        if self.output_dimensionality < 3072:
            embedding = self._normalize(embedding)
            return embedding.tolist()  # numpy配列をリストに変換
        
        return list(embedding)
    
    # インデックス作成で使う
    def embed_batch(self, texts: List[str], task_type: str = None) -> List[List[float]]:
        """複数テキストのEmbeddingを一度に生成（クオータ制限時は1件ずつ処理）"""
        try:
            # 一括処理を試行
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type or self.task_type,
                    output_dimensionality=self.output_dimensionality
                )
            )
            embeddings = [list(e.values) for e in result.embeddings]
            
        except Exception as e:
            # クオータ制限エラーなら1件ずつ処理
            if '429' not in str(e) and 'RESOURCE_EXHAUSTED' not in str(e):
                raise
            
            print(f"    クオータ制限検出。1件ずつ処理に切り替え ({len(texts)}件)")
            embeddings = []
            
            for text in tqdm(texts):
                while True:
                    try:
                        result = self.client.models.embed_content(
                            model=self.model,
                            contents=text,
                            config=types.EmbedContentConfig(
                                task_type=task_type or self.task_type,
                                output_dimensionality=self.output_dimensionality
                            )
                        )
                        embeddings.append(list(result.embeddings[0].values))
                        break
                    except Exception as retry_e:
                        if '429' in str(retry_e) or 'RESOURCE_EXHAUSTED' in str(retry_e):
                            for _ in tqdm(range(60), desc="      クオータ制限: 60秒待機", leave=False):
                                time.sleep(1)
                        else:
                            raise
        
        # 3072次元以外は正規化が必要
        if self.output_dimensionality < 3072:
            embeddings = [self._normalize(emb).tolist() for emb in embeddings]
        
        return embeddings
    
    def _normalize(self, embedding: Union[List[float], np.ndarray]) -> np.ndarray:
        """エンベディングを正規化（ノルムを1にする）"""
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        return embedding_np / norm if norm > 0 else embedding_np
    
    def encode(self, text: Union[str, List[str]], task_type: str = None) -> Union[List[float], List[List[float]]]:
        """テキストをエンコード（sentence-transformers互換）"""
        return self.embed_text(text, task_type) if isinstance(text, str) else self.embed_batch(text, task_type)


def get_gemini_embedding(
    model: str = GEMINI_EMBEDDING_MODEL_NAME,
    output_dimensionality: int = EMBEDDING_DIM,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> GeminiEmbedding:
    """Gemini埋め込みモデルを取得
    
    Args:
        model: モデル名
        output_dimensionality: 出力次元数（推奨: 768, 1536, 3072。デフォルト: 3072だがストレージ節約のため768を使用）
        task_type: タスクタイプ
            - RETRIEVAL_DOCUMENT: ドキュメント検索用（インデックス時）
            - RETRIEVAL_QUERY: クエリ検索用（検索時）
            - SEMANTIC_SIMILARITY: 意味的類似性評価用
    
    Returns:
        GeminiEmbedding: Gemini埋め込みモデルのインスタンス
    """
    return GeminiEmbedding(
        model=model,
        output_dimensionality=output_dimensionality,
        task_type=task_type
    )

