import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from .llm_models import GeminiRAGModel
from .embedding_models import get_gemini_embedding
from .config import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    EMBEDDING_DIM,
    RAG_INDEX_NAME,
    RAG_TOP_K,
    RRF_RANK_CONSTANT,
    RAG_LLM_MODEL_NAME,
    RAG_LLM_TEMPERATURE,
    RAG_LLM_MAX_OUTPUT_TOKENS,
    RAG_LLM_THINKING_BUDGET,
)

load_dotenv()


class BaseOpenSearchRAG(ABC):
    """OpenSearch RAGの基底クラス"""
    
    def __init__(
        self,
        host: str = OPENSEARCH_HOST,
        port: int = OPENSEARCH_PORT,
        index_name: str = RAG_INDEX_NAME,
        embedding_dim: int = EMBEDDING_DIM,
        top_k: int = RAG_TOP_K,
        llm_model = None
    ):
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
            task_type="RETRIEVAL_QUERY"
        )
        
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.llm_model = llm_model
        
        # インデックス存在確認
        if not self.client.indices.exists(index=self.index_name):
            raise ValueError(f"インデックス '{self.index_name}' が存在しません")
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """クエリのEmbeddingを生成"""
        return self.embedding_model.embed_text(query, task_type="RETRIEVAL_QUERY")
    
    @abstractmethod
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """検索を実行（サブクラスで実装）"""
        pass
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """クエリに基づいて関連ドキュメントを取得（simple_rag.pyと互換）"""
        return self.search(query, top_k=k)
    
    def _format_results(self, response: dict) -> List[Dict]:
        """検索結果を整形"""
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'content': hit['_source']['content'],
                'score': hit['_score'],
                'source': hit['_source'].get('source', ''),
                'filename': hit['_source'].get('filename', ''),
                'chunk_index': hit['_source'].get('chunk_index', 0),
                'page': hit['_source'].get('page', 'null'),
                'id': hit['_id']
            })
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """取得したドキュメントを基に回答を生成"""
        context = "\n\n".join([
            f"[ソース: {doc['filename']}, ページ: {doc['page']}]\n{doc['content']}"
            for doc in context_docs
        ])
        
        prompt = f"""以下のコンテキストに基づいて、質問に答えてください。
コンテキストに情報がない場合は、「提供された情報からは回答できません」と答えてください。

コンテキスト:
{context}

質問: {query}

回答:"""
        
        if self.llm_model:
            answer = self.llm_model.generate_response(prompt)
        else:
            answer = "LLMモデルが設定されていません"
        
        return answer
    
    def query(self, query: str, k: Optional[int] = None) -> str:
        """質問に対する回答を生成（検索と生成を統合）"""
        k = k or self.top_k
        retrieved_docs = self.retrieve(query, k=k)
        answer = self.generate_answer(query, retrieved_docs)
        return answer
    
    def answer(self, query: str, k: Optional[int] = None, verbose: bool = True) -> Dict:
        """
        Args:
            query: 検索クエリ
            k: 取得する上位件数
            verbose: チャンク情報を表示するか

        Returns:
            Dict: {query, answer, sources}
        """
        k = k or self.top_k
        retrieved_docs = self.retrieve(query, k=k)
        
        # チャンク情報を表示
        if verbose:
            self._display_chunks(retrieved_docs)
        
        answer = self.generate_answer(query, retrieved_docs)
        
        if verbose:
            print(f"\n【回答】")
            print(answer)
        
        return {
            'query': query,
            'answer': answer,
            'sources': retrieved_docs
        }
    
    def _display_chunks(self, sources: List[Dict]):
        """取得したチャンク情報を表示（基本版：サブクラスでオーバーライド可能）"""
        print("\n【取得チャンク】")
        for i, doc in enumerate(sources, 1):
            print(f"{i}. file: {doc['filename']} | chunk: {doc['chunk_index']} | page: {doc['page']}")
            print(f"   score: {doc['score']:.4f}")
            print(f"   内容: {doc['content'][:100]}...")
            print()
 
 
class RRFOpenSearchRAG(BaseOpenSearchRAG):
    """RRF (Reciprocal Rank Fusion)"""
    
    def __init__(
        self,
        host: str = OPENSEARCH_HOST,
        port: int = OPENSEARCH_PORT,
        index_name: str = RAG_INDEX_NAME,
        embedding_dim: int = EMBEDDING_DIM,
        top_k: int = RAG_TOP_K,
        llm_model = None,
        rank_constant: int = RRF_RANK_CONSTANT
    ):
        super().__init__(host, port, index_name, embedding_dim, top_k, llm_model)
        self.rank_constant = rank_constant
        self.search_pipeline_name = f"{index_name}-rrf-pipeline"
        
        self._create_search_pipeline()
    
    def _create_search_pipeline(self):
        """RRF検索パイプラインを作成"""
        pipeline_body = {
            "description": "Post processor for hybrid RRF search",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": self.rank_constant
                        }
                    }
                }
            ]
        }
        
        try:
            self.client.transport.perform_request(
                'DELETE',
                f'/_search/pipeline/{self.search_pipeline_name}'
            )
        except:
            pass
        
        self.client.transport.perform_request(
            'PUT',
            f'/_search/pipeline/{self.search_pipeline_name}',
            body=pipeline_body
        )
        print(f"✅ RRF検索パイプライン '{self.search_pipeline_name}' を作成しました")
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """クエリに基づいて関連ドキュメントを取得（ランク情報付き）"""
        k = k or self.top_k
        query_embedding = self._generate_query_embedding(query)

        # メインのハイブリッド検索
        return self.search(query, top_k=k)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """RRFによるハイブリッド検索"""
        k = top_k or self.top_k
        query_embedding = self._generate_query_embedding(query)
        
        search_body = {
            'size': k,
            '_source': {
                'exclude': ['embedding']
            },
            'query': {
                'hybrid': {
                    'queries': [
                        {
                            'match': {
                                'content': query
                            }
                        },
                        {
                            'knn': {
                                'embedding': {
                                    'vector': query_embedding,
                                    'k': k
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        response = self.client.search(
            index=self.index_name,
            body=search_body,
            params={'search_pipeline': self.search_pipeline_name}
        )
        
        return self._format_results(response)
    
    def _display_chunks(self, sources: List[Dict]):
        """RRF検索のチャンク情報を表示"""
        print(f"\n設定: rank_constant={self.rank_constant}")
        print("\n【取得チャンク】")
        
        for i, doc in enumerate(sources, 1):
            print(f"{i}. file: {doc['filename']} | chunk: {doc['chunk_index']} | page: {doc['page']}")
            print(f"   RRF_rank: {i} | score: {doc['score']:.6f}")
            print(f"   内容: {doc['content'][:100]}...")
            print()





def get_opensearch_rag(
    index_name: str,
    host: str = OPENSEARCH_HOST,
    port: int = OPENSEARCH_PORT,
    embedding_dim: int = EMBEDDING_DIM,
    top_k: int = RAG_TOP_K,
    llm_model = None,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> BaseOpenSearchRAG:
    """OpenSearch RAGインスタンスを取得するファクトリ関数"""
    return RRFOpenSearchRAG(
        host, port, index_name, embedding_dim, top_k, llm_model,
        rank_constant=rank_constant
    )



def main():
    """
    python -m rag_opensearch
    """
    
    llm_model = GeminiRAGModel(
        model_name=RAG_LLM_MODEL_NAME,
        temperature=RAG_LLM_TEMPERATURE,
        max_output_tokens=RAG_LLM_MAX_OUTPUT_TOKENS,
        thinking_budget=RAG_LLM_THINKING_BUDGET,
    )

    # サンプル質問
    questions = [
        "「持続可能なビジネスモデル」って、保険会社はどんな取組みをすればいいとされてるの？",
        "来神客の属性 って、どっち多い 女か男か、年齢どの層多いか、居住地どこ多いか 教えて?",
    ]
    
    rag = get_opensearch_rag(  # embddingはgemini_embeddingで固定
        index_name=RAG_INDEX_NAME,
        top_k=RAG_TOP_K,
        llm_model=llm_model,
    )

    for question in questions:
        print(f"\n\n=== 質問: {question} ===")

        # Dict{query:str, answer:str, sources:[{id:str, filename:str, chunk_index:int, page:int, score:float, content:str}]}
        result = rag.answer(question, k=5, verbose=True) 
        print(f"\n\n{result}\n\n")
    


if __name__ == "__main__":
    main()