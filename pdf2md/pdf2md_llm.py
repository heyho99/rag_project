#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m src.pdf2text.pdf2md_llm
※ 13000文字程度までしかgeminiは生成してくれない
"""

import os
import glob
import re
from datetime import datetime
from pathlib import Path
from ..llm_components.llm_models import LLMModel



def extract_text_from_pdf(pdf_path: str, llm_model: LLMModel, prompt: str) -> str:
    """
    PDFファイルからLLMを使用してマークダウン形式のテキストを生成
    
    Args:
        pdf_path (str): PDFファイルのパス
        llm_model (LLMModel): 使用するLLMモデル
        
    Returns:
        str: 生成されたマークダウンテキスト
    """
    try:
        if not os.path.exists(pdf_path):
            print(f"エラー: PDFファイルが見つかりません: {pdf_path}")
            return ""
        
        print(f"PDF処理中: {pdf_path}")
        print(f"使用モデル: {llm_model.__class__.__name__}")
        
        # LLMモデルにPDFとプロンプトを渡し、応答を取得
        response = llm_model.generate_response(
            prompt=prompt,
            pdf_file_path=pdf_path
        )
        
        if response and response != "レスポンスが空です":
            print(f"変換成功: {len(response)} 文字のコンテンツを生成")
            return response
        else:
            print("エラー: LLMからの応答が空です")
            return ""
            
    except Exception as e:
        print(f"PDF処理エラー: {e}")
        return ""


def generate_output_filename(output_dir: str, pdf_path: str, model_name: str) -> str:
    """
    タイムスタンプ付きの出力ファイル名を生成
    
    Args:
        pdf_path (str): 元のPDFファイルパス
        model_name (str): 使用するLLMモデル名
    Returns:
        str: 出力ファイル名（フルパス）
    """
    # 出力ディレクトリの作成
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = re.sub(r"\.", "", model_name)
    base_name = Path(pdf_path).stem
    output_filename = f"{base_name}_{model_name}_{timestamp}.md"
    return os.path.join(output_dir, output_filename)


if __name__ == "__main__":
    # 必要設定 ====================================================================

    # 出力ディレクトリパス
    output_dir = "outputs/pdf2md_llm"
    # output_dir = "outputs/pdf2text_llm"
    os.makedirs(output_dir, exist_ok=True) # ディレクトリの存在確認と作成

    # モデル名の設定
    model_name = "gemini-2.5-flash"
    
    # LLMモデルのインスタンス作成
    from ..llm_components.llm_models import GeminiPDFConverterModel
    llm_model = GeminiPDFConverterModel(
        model_name=model_name,
        temperature=1.0,
        thinking_budget=0,
        max_output_tokens=65536
    )

    # 使用するプロンプトの選択
    from ..llm_components.prompts import *
    prompt = PDF_CONVERSION_SYSTEM_PROMPT
    # prompt = PDF_TEXT_EXTRACTION_SYSTEM_PROMPT

    # =============================================================================
    

    # 処理対象PDFパターン
    # pdf_glob_pattern = "docs/*.pdf"
    pdf_glob_pattern = "docs/01.pdf"
    pdf_files = glob.glob(pdf_glob_pattern)

    if pdf_files:
        print(f"パターン '{pdf_glob_pattern}' に一致したPDFファイル {len(pdf_files)} 件を検出:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
    
    if not pdf_files:
        print("処理対象のPDFファイルがありません")
        exit(1)
    
    print(f"使用モデル: {llm_model.__class__.__name__}")
    print()
    
    # 各PDFファイルを処理
    for pdf_path in pdf_files:
        print(f"処理中: {pdf_path}")
        
        # マークダウン変換
        content = extract_text_from_pdf(
            pdf_path=pdf_path, 
            llm_model=llm_model, 
            prompt=prompt
            )
        
        if content:
            # 出力ファイル名を生成
            output_path = generate_output_filename(
                output_dir=output_dir, 
                pdf_path=pdf_path, 
                model_name=model_name
                )
            
            # ファイルに保存
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  - 出力: {output_path}")
                
                # ファイルサイズを表示
                file_size = os.path.getsize(output_path)
                print(f"  - サイズ: {file_size:,} bytes")
                
            except Exception as e:
                print(f"  - 保存エラー: {e}")
        else:
            print(f"  - エラー: 変換に失敗しました")
        
        print()
