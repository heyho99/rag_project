import os
from typing import Optional
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
        thinking_level: str = "HIGH",
        api_key: Optional[str] = None,
    ):
        """
        Geminiモデルの基本初期化
        
        Args:
            model_name (str): 使用するGeminiモデル名
            thinking_level (str): 思考レベル
            api_key (Optional[str]): Gemini API Key（未指定なら環境変数 GEMINI_API_KEY を参照）
        """
        # Gemini APIクライアントの設定
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        )
        self.model_name = model_name
        self.thinking_level = thinking_level


# =============================================================================
# RAGシステム用 - Gemini Models
# =============================================================================

class GeminiRAGModel(BaseGeminiModel):
    """RAG用Geminiモデル: contextを受け取りレスポンス生成"""

    def generate_response(self, prompt: str, **kwargs) -> str:
        """RAG用レスポンス生成"""
        model = self.model_name
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level,
            ),
        )

        chunks = []
        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            chunks.append(chunk.text or "")

        return "".join(chunks)

