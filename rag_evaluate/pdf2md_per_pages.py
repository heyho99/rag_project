#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m src.pdf2text.pdf2md_per_pages

PDFを任意のページ数ごとに分割してLLMへ送信し、チャンク単位でマークダウンへ変換します。
`pdf2md_llm.py` と同様に Gemini の File API を利用しますが、ページ分割のために
pypdf (PdfReader / PdfWriter) を使用して一時的にページごとのPDFを生成します。
"""

import os
import glob
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader, PdfWriter

from rag_evaluate.llm_models import LLMModel, GeminiPDFConverterModel
from rag_evaluate.config import (
    PROJECT_ROOT,
    PDF2MD_INPUT_PATTERN,
    PDF2MD_OUTPUT_DIR,
    PDF2MD_MODEL_NAME,
    PDF2MD_TEMPERATURE,
    PDF2MD_THINKING_BUDGET,
    PDF2MD_MAX_OUTPUT_TOKENS,
    PDF2MD_PAGES_PER_CHUNK,
    PDF2MD_PROMPT,
)


# =============================================================================
# 変換ロジック
# =============================================================================

def split_pdf_into_chunks(
    pdf_path: str,
    pages_per_chunk: int,
    temp_dir: str
) -> List[Tuple[str, int, int]]:
    """PDFをページ数ごとのチャンクに分割し、一時PDFを作成する。

    Args:
        pdf_path (str): 入力PDFのパス
        pages_per_chunk (int): 1チャンクあたりのページ数 (1以上)
        temp_dir (str): 一時ファイルを配置するディレクトリ

    Returns:
        List[Tuple[str, int, int]]: (チャンクPDFパス, 開始ページ(1-index), 終了ページ(1-index)) のリスト
    """
    pages_per_chunk = max(1, pages_per_chunk)

    # PDFファイルの読み込み
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    if total_pages == 0:
        return []

    chunk_info_list: List[Tuple[str, int, int]] = [] # (チャンク名, 開始ページ, 終了ページ)のリスト
    pdf_stem = Path(pdf_path).stem # sample.pdf -> sample という文字列を取得する

    # n=3の場合、0,3,6... 分割した単位をチャンクとする
    for start_idx in range(0, total_pages, pages_per_chunk):
        end_idx = min(start_idx + pages_per_chunk, total_pages) # 1ループの終了ページを取得(6+3=9)
        writer = PdfWriter() # 1ループ分のPDFを生成して入れる箱(P6~P8のセット)

        for page_index in range(start_idx, end_idx): # P6~P8のセットをwriterに入れる
            writer.add_page(reader.pages[page_index])

        chunk_start_page = start_idx + 1  # 0Pを1Pとして人間が読みやすいように
        chunk_end_page = end_idx          
        chunk_filename = ( # 1チャンク分のPDFを保存するファイル名
            Path(temp_dir)
            / f"{pdf_stem}_pages_{chunk_start_page:04d}-{chunk_end_page:04d}.pdf"
        )

        with open(chunk_filename, "wb") as chunk_file: # 1チャンク分のPDFを保存する
            writer.write(chunk_file)

        chunk_info_list.append((str(chunk_filename), chunk_start_page, chunk_end_page))

    return chunk_info_list


def extract_markdown_from_pdf(
    pdf_path: str,
    llm_model: LLMModel,
    prompt: str,
    start_page: int,
    end_page: int
) -> str:
    """PDFチャンクをLLMでマークダウンへ変換する。

    Args:
        pdf_path (str): チャンクPDFファイルのパス
        llm_model (LLMModel): 使用するLLMモデル
        prompt (str): LLMに渡すプロンプト
        start_page (int): チャンク開始ページ (1-indexed)
        end_page (int): チャンク終了ページ (1-indexed)

    Returns:
        str: 生成されたマークダウンテキスト
    """
    try:
        if not os.path.exists(pdf_path):
            print(f"エラー: PDFファイルが見つかりません: {pdf_path}")
            return ""

        pages_label = f"ページ {start_page} - {end_page}"
        print(f"PDFチャンク処理中 ({pages_label}): {pdf_path}")
        print(f"使用モデル: {llm_model.__class__.__name__}")

        response = llm_model.generate_response(
            prompt=prompt,
            pdf_file_path=pdf_path
        )

        if response and response != "レスポンスが空です":
            print(f"変換成功 ({pages_label}): {len(response)} 文字")
            return response
        else:
            print(f"エラー: LLMからの応答が空です ({pages_label})")
            return ""

    except Exception as e:
        print(f"PDFチャンク処理エラー ({start_page}-{end_page}): {e}")
        return ""


def generate_chunk_output_filename(
    output_dir: str,
    pdf_path: str,
    model_name: str,
    start_page: int,
    end_page: int
) -> str:
    """チャンク出力用のファイル名を生成する。"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = re.sub(r"\.", "", model_name)
    base_name = Path(pdf_path).stem
    output_filename = (
        f"{base_name}_{model_name}_pages_{start_page:04d}-{end_page:04d}_{timestamp}.md"
    )
    return os.path.join(output_dir, output_filename)


# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    # 必要設定 ====================================================================

    output_dir = str(PROJECT_ROOT / PDF2MD_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    model_name = PDF2MD_MODEL_NAME
    pages_per_chunk = PDF2MD_PAGES_PER_CHUNK  # 任意のページ数に変更してください

    # LLMモデルのインスタンス作成
    llm_model = GeminiPDFConverterModel(
        model_name=model_name,
        temperature=PDF2MD_TEMPERATURE,
        thinking_budget=PDF2MD_THINKING_BUDGET,
        max_output_tokens=PDF2MD_MAX_OUTPUT_TOKENS,
    )

    # 使用するプロンプト
    prompt = PDF2MD_PROMPT

    # =============================================================================

    # pdf_glob_pattern = "docs/01.pdf"
    pdf_glob_pattern = str(PROJECT_ROOT / PDF2MD_INPUT_PATTERN)
    pdf_files = glob.glob(pdf_glob_pattern)

    if pdf_files:
        print(f"パターン '{pdf_glob_pattern}' に一致したPDFファイル {len(pdf_files)} 件を検出:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
    else:
        print("処理対象のPDFファイルがありません")
        exit(1)

    print(f"使用モデル: {llm_model.__class__.__name__}")
    print(f"チャンクページ数: {pages_per_chunk}\n")

    for pdf_path in pdf_files:
        print(f"処理中: {pdf_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_infos = split_pdf_into_chunks(pdf_path, pages_per_chunk, temp_dir)

            if not chunk_infos:
                print("  - エラー: チャンクを作成できませんでした")
                continue

            for chunk_index, (chunk_path, start_page, end_page) in enumerate(chunk_infos, 1):
                content = extract_markdown_from_pdf(
                    pdf_path=chunk_path,
                    llm_model=llm_model,
                    prompt=prompt,
                    start_page=start_page,
                    end_page=end_page
                )

                if not content:
                    print(f"  - エラー: ページ {start_page}-{end_page} の変換に失敗しました")
                    continue

                output_path = generate_chunk_output_filename(
                    output_dir=output_dir,
                    pdf_path=pdf_path,
                    model_name=model_name,
                    start_page=start_page,
                    end_page=end_page
                )

                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    print(f"  - 出力({chunk_index}/{len(chunk_infos)}): {output_path}")
                    file_size = os.path.getsize(output_path)
                    print(f"    サイズ: {file_size:,} bytes")
                except Exception as e:
                    print(f"  - 保存エラー ({start_page}-{end_page}): {e}")

        print()
