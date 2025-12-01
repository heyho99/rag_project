#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm_models.py - LLMモデル定義

責務の再整理:
    * 各モデルは「プロンプト(と追加メタ)→LLM呼び出し→テキスト出力」だけを担当
    * PDFバイナリ読込/URL取得/Base64化/ストリーミング結合等の具体的処理は pipelines モジュール側へ移動

本ファイルでは最小限のモデルクラスとファクトリのみを提供する。
"""

# llm呼び出しも基本同じ形式なので、クラスに含めて呼び出せるようにする

import os
import pathlib
import time
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
# プロンプトは外部（呼び出し側）からDIで渡す方針に変更。

# .envファイルを読み込み
load_dotenv()


class LLMModel(ABC):
    """LLMモデルの抽象基底クラス"""
    
    # インターフェースはあくまでもDIのためで、中身は薄くても良いか
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        プロンプトからレスポンスを生成する抽象メソッド
        
        Args:
            prompt (str): LLMに送信するプロンプト
            **kwargs: 追加のパラメータ
            
        Returns:
            str: LLMからのレスポンス
        """
        pass


class BaseGeminiModel(LLMModel):
    """Gemini系モデルの基底クラス"""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.4,
        max_output_tokens: int = 65536,
        thinking_budget: int = 0
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
# PDF変換用 - Gemini Models
# =============================================================================

class GeminiPDFConverterModel(BaseGeminiModel):
    """Gemini APIのFile APIを使用したPDF処理専用モデル"""
    
    def generate_response(self, prompt: str, pdf_file_path: str, **kwargs) -> str:
        """
        Gemini File APIを使用してPDFファイルを処理してレスポンス生成
        
        Args:
            prompt (str): LLMに送信するプロンプト
            pdf_file_path (str): PDFファイルのパス（必須）
            **kwargs: 追加のパラメータ
            
        Returns:
            str: LLMからのレスポンス
        """
        try:
            if not self.api_key:
                return "レスポンス生成に失敗（APIキーエラー）"
            
            return self._process_pdf_with_file_api(prompt, pdf_file_path)
                        
        except Exception as e:
            return self._handle_api_error(e, "GeminiPDFConverterModel")
    
    # PDFサイズに関わらず、Google File APIでPDFを処理
    def _process_pdf_with_file_api(self, prompt: str, pdf_file_path: str) -> str:
        """File APIを使用してPDFファイルを処理"""
        try:
            # PDFファイルパスをPathlibオブジェクトに変換
            file_path = pathlib.Path(pdf_file_path)
            
            if not file_path.exists():
                return f"エラー: PDFファイルが見つかりません: {pdf_file_path}"
            
            # PDFファイルをGoogle File APIにアップロード
            sample_file = self.client.files.upload(file=file_path)
            
            # File APIからのレスポンス生成
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[sample_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget) # 思考予算を設定（proは128以上）
                    # thinking_config=types.ThinkingConfig(thinking_budget=-1) # 動的思考
                )
            )
            
            return response.text.strip() if response.text else "レスポンスが空です"
            
        except Exception as e:
            print(f"File API処理エラー: {e}")
            return f"File API処理に失敗: {str(e)}"