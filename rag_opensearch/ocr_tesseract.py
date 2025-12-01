"""Tesseract OCR を用いた日本語PDF -> テキスト変換モジュール"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import pdfplumber
import pytesseract
from .config import OCR_INPUT_PATTERN, OCR_OUTPUT_DIR, OCR_LANG, OCR_DPI


def pdf_to_text(
    pdf_path: str,
    lang: str,
    resolution: int,
) -> str:
    """PDF を 1 ページずつ OCR しテキストとして連結する.

    Args:
        pdf_path: 入力 PDF のパス。
        lang: pytesseract に渡す言語コード。
        resolution: OCR 用にレンダリングする際の DPI。

    Returns:
        連結済みテキスト。ページ毎に区切りを入れる。
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            pil_image = page.to_image(resolution=resolution).original
            text = pytesseract.image_to_string(pil_image, lang=lang).strip()
            texts.append(f"--- ページ {page_number} ---\n{text}")
    return "\n\n".join(texts)


def pdf_to_text_file(
    pdf_path: str,
    output_path: Optional[str],
    lang: str,
    resolution: int,
) -> Path:
    """PDF からテキストを抽出しファイルへ保存する."""
    pdf_path_obj = Path(pdf_path)
    if output_path is None:
        output_path_obj = pdf_path_obj.with_suffix(".txt")
    else:
        output_path_obj = Path(output_path)

    text = pdf_to_text(str(pdf_path_obj), lang=lang, resolution=resolution)
    output_path_obj.write_text(text, encoding="utf-8")
    return output_path_obj


def main() -> None:
    pdf_pattern = OCR_INPUT_PATTERN
    output_dir = Path(OCR_OUTPUT_DIR)
    lang = OCR_LANG
    dpi = OCR_DPI

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
