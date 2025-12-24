"""Tesseract OCR を用いた日本語PDF -> テキスト変換モジュール"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import pdfplumber
import pytesseract

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def pdf_to_text(
    pdf_path: str,
    lang: str = None,
    resolution: int = None,
) -> str:
    """PDF を 1 ページずつ OCR しテキストとして連結する.

    Args:
        pdf_path: 入力 PDF のパス。
        lang: pytesseract に渡す言語コード。
        resolution: OCR 用にレンダリングする際の DPI。

    Returns:
        連結済みテキスト。ページ毎に区切りを入れる。
    """
    if lang is None:
        lang = config.TESSERACT_OCR_LANG
    if resolution is None:
        resolution = config.TESSERACT_OCR_DPI
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            pil_image = page.to_image(resolution=resolution).original
            text = pytesseract.image_to_string(pil_image, lang=lang).strip()
            texts.append(f"--- ページ {page_number} ---\n{text}")
    return "\n\n".join(texts)


def pdf_to_text_file(
    pdf_path: str,
    output_path: Optional[str] = None,
    lang: str = None,
    resolution: int = None,
) -> Path:
    """PDF からテキストを抽出しファイルへ保存する."""
    pdf_path_obj = Path(pdf_path)
    if output_path is None:
        output_path_obj = pdf_path_obj.with_suffix(".txt")
    else:
        output_path_obj = Path(output_path)

    if lang is None:
        lang = config.TESSERACT_OCR_LANG
    if resolution is None:
        resolution = config.TESSERACT_OCR_DPI
    text = pdf_to_text(str(pdf_path_obj), lang=lang, resolution=resolution)
    output_path_obj.write_text(text, encoding="utf-8")
    return output_path_obj


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pdf_pattern = str(project_root / config.PDF_INPUT_GLOB)
    output_dir = project_root / config.TESSERACT_TXT_OUTPUT_DIR
    lang = config.TESSERACT_OCR_LANG
    dpi = config.TESSERACT_OCR_DPI

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(glob.glob(pdf_pattern))
    if not pdf_paths:
        print(f"対象PDFが見つかりませんでした: {pdf_pattern}")
        return

    for pdf_path in pdf_paths:
        pdf_path_obj = Path(pdf_path)
        output_path = output_dir / f"{pdf_path_obj.stem}.txt"
        saved_path = pdf_to_text_file(
            pdf_path=str(pdf_path_obj),
            output_path=str(output_path),
            lang=lang,
            resolution=dpi,
        )
        print(f"テキストを保存しました: {saved_path}")


if __name__ == "__main__":
    main()
