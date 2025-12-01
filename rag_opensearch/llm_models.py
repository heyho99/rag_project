import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .envファイルを読み込み
load_dotenv()

class BaseGeminiModel():
    """Gemini系モデルの基底クラス"""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.4,
        max_output_tokens: int = 65536,
        thinking_budget: int = 128
    ):
        """
        Geminiモデルの基本初期化
        
        Args:
            model_name (str): 使用するGeminiモデル名
            temperature (float): 温度パラメータ
            thinking_budget (int): 思考予算（トークン数）
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEYが設定されていません")
        
        load_dotenv()
        
        # Gemini APIクライアントの設定
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens

    def _handle_api_error(self, e: Exception, model_class_name: str) -> str:
        """API呼び出しエラーのハンドリング"""
        print(f"{model_class_name} API呼び出しエラー: {e}")
        print(f"エラーの詳細: {type(e).__name__}: {str(e)}")
        return "レスポンス生成に失敗"


# =============================================================================
# RAGシステム用 - Gemini Models
# =============================================================================

class GeminiRAGModel(BaseGeminiModel):
    """RAG用Geminiモデル: contextを受け取りレスポンス生成"""

    def generate_response(self, prompt: str, **kwargs) -> str:
        """RAG用レスポンス生成"""
        quota_retried = False
        overload_retry_count = 0
        max_overload_retries = 3
        while True:
            try:
                if not self.api_key:
                    return "レスポンス生成に失敗（APIキーエラー）"

                # Gemini APIを使用してレスポンス生成
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget) # 思考予算を設定（proは128以上）
                        # thinking_config=types.ThinkingConfig(thinking_budget=-1) # 動的思考
                    )
                )

                return response.text.strip() if response.text else "レスポンスが空です"

            except Exception as e:
                if (not quota_retried) and self._is_quota_error(e):
                    quota_retried = True
                    wait_seconds = 60
                    print(f"GeminiRAGChatModel: クォータ制限を検出。{wait_seconds}秒待機してリトライします。")
                    time.sleep(wait_seconds)
                    continue

                if overload_retry_count < max_overload_retries and self._is_overloaded_error(e):
                    overload_retry_count += 1
                    wait_seconds = 20
                    print(
                        f"GeminiRAGChatModel: モデル過負荷エラーを検出。"
                        f"{wait_seconds}秒待機してリトライします。"
                    )
                    time.sleep(wait_seconds)
                    continue

                return self._handle_api_error(e, "GeminiRAGChatModel")

    def _is_quota_error(self, error: Exception) -> bool:
        message = str(error)
        lowered = message.lower()
        return (
            "\"code\": 429" in lowered
            or " resource_exhausted" in lowered
            or "status': 'resource_exhausted" in lowered
            or "\"status\": \"resource_exhausted\"" in lowered
        )

    def _is_overloaded_error(self, error: Exception) -> bool:
        message = str(error)
        lowered = message.lower()
        return (
            "\"code\": 503" in lowered
            or " 503" in message
            or "unavailable" in lowered
            or "model is overloaded" in lowered
        )

