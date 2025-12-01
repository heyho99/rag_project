"""
pdfplumberを使用したPDF→テキスト変換モジュール
# python -m src.pdf2text.simpleext_pdfplumber
# python -m src.pdf2text.simpleext_pdfplumber --debug
"""

import os
import pdfplumber
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional


class PDFToTextConverter:
    """pdfplumberを使用したPDF→テキスト変換クラス"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初期化
        Args:
            debug_mode (bool): デバッグモード（Trueの場合デバッグ画像を生成）
        """
        self.output_dir = Path("outputs/simpleext_pdfplumber")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.debug_mode = debug_mode
        # デバッグ用ディレクトリの作成（デバッグモード時のみ）
        if self.debug_mode:
            self.debug_dir = self.output_dir / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDFファイルから全ページのテキストとテーブルを抽出して結合
        同時にデバッグ画像も生成する
        Args:
            pdf_path (str): PDFファイルのパス
        Returns:
            str: 抽出された全テキストとテーブル
        """
        try:
            # PDFファイル名（拡張子なし）を取得
            pdf_filename = Path(pdf_path).stem
            
            with pdfplumber.open(pdf_path) as pdf:
                all_content = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    all_content.append(f"--- ページ {page_num} ---")
                    
                    # ページからテキストを抽出
                    text = page.extract_text()
                    if text:
                        all_content.append(text)
                    
                    # ページからテーブルを抽出
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            all_content.append(f"\n--- テーブル {table_num} ---")
                            for row in table:
                                if row:  # 空の行をスキップ
                                    row_text = " | ".join(str(cell) if cell else "" for cell in row)
                                    all_content.append(row_text)
                            all_content.append("")  # テーブル後の区切り
                    
                    all_content.append("")  # ページ間の区切り
                
                # デバッグモード時のみデバッグ画像を生成
                if self.debug_mode:
                    print(f"デバッグ画像を生成中...")
                    self.create_debug_images(pdf_path, pdf_filename)
                
                return '\n'.join(all_content)
                        
        except Exception as e:
            print(f"PDFの読み込みエラー: {e}")
            return ""

    def create_debug_images(self, pdf_path: str, pdf_filename: str):
        """
        PDFの各ページに対してデバッグ画像を作成
        Args:
            pdf_path (str): PDFファイルのパス
            pdf_filename (str): PDFファイル名（拡張子なし）
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # ページ全体の画像を作成
                    im = page.to_image(resolution=150)
                    
                    # 文字のバウンディングボックスを描画（赤色）
                    if page.chars:
                        im.draw_rects(page.chars, stroke=(255, 0, 0), stroke_width=1)
                    
                    # 検出された線を描画（青色）
                    if page.lines:
                        im.draw_lines(page.lines, stroke=(0, 0, 255), stroke_width=2)
                    
                    # カスタム描画画像を保存
                    page_debug_path = self.debug_dir / f"{pdf_filename}_page{page_num:02d}_debug.png"
                    im.save(page_debug_path, format="PNG")
                    print(f"  - デバッグ画像: {page_debug_path}")
                    
                    # テーブル検出のデバッグ画像を作成
                    try:
                        im_table_debug = page.debug_tablefinder()
                        table_debug_path = self.debug_dir / f"{pdf_filename}_page{page_num:02d}_table_debug.png"
                        im_table_debug.save(table_debug_path, format="PNG")
                        print(f"  - テーブルデバッグ画像: {table_debug_path}")
                    except Exception as table_debug_error:
                        print(f"  - テーブルデバッグ画像作成エラー（ページ{page_num}）: {table_debug_error}")
                        
        except Exception as e:
            print(f"デバッグ画像作成エラー: {e}")

    # メインメソッド
    def convert_pdf_file(self, pdf_path: str, output_filename: Optional[str] = None) -> str:
        """
        PDFファイルを変換してテキストファイルとして保存
        
        Args:
            pdf_path (str): PDFファイルのパス
            output_filename (str, optional): 出力ファイル名（省略時は元ファイル名を使用）
            
        Returns:
            str: 保存されたファイルのパス
        """
        if not os.path.exists(pdf_path):
            print(f"エラー: PDFファイルが見つかりません: {pdf_path}")
            return ""
        
        # テキストを抽出
        text_content = self.extract_text_from_pdf(pdf_path)
        
        if not text_content:
            print(f"エラー: PDFからテキストを抽出できませんでした: {pdf_path}")
            return ""
        
        # 出力ファイル名を決定
        if output_filename is None:
            pdf_name = Path(pdf_path).stem
            converter_name = self.__class__.__name__.lower().replace('converter', '')
            # 現在の日時を取得（YYYYMMDD_HHMMSS形式）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{pdf_name}_{converter_name}_{timestamp}.txt"
        
        # ファイルに保存
        output_path = self.output_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            print(f"ファイルが保存されました: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"ファイル保存エラー: {e}")
            return ""




def test_converter(paths_list, debug_mode: bool = False):
    """テスト関数
    Args:
        paths_list: テスト対象のPDFファイルパスリスト
        debug_mode (bool): デバッグモード（Trueの場合デバッグ画像を生成）
    """
    # コンバーターの初期化
    converter = PDFToTextConverter(debug_mode=debug_mode)
    
    test_pdf_paths = paths_list
    
    print(f"=== {converter.__class__.__name__} ===")
    if debug_mode:
        print("デバッグモード: ON")
    else:
        print("デバッグモード: OFF")
    print()
    
    for pdf_path in test_pdf_paths:
        if os.path.exists(pdf_path):
            print(f"処理中: {pdf_path}")
            
            # 変換実行
            output_path = converter.convert_pdf_file(pdf_path)
            
            if output_path:
                print(f"  - 出力: {output_path}")
                
                # ファイルサイズを表示
                file_size = os.path.getsize(output_path)
                print(f"  - サイズ: {file_size:,} bytes")
            else:
                print(f"  - エラー: 変換に失敗しました")
            
            print()
        else:
            print(f"ファイルが見つかりません: {pdf_path}\n")



if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="PDFからテキストを抽出するツール")
    parser.add_argument("--debug", action="store_true", help="デバッグ画像を生成する")
    args = parser.parse_args()
    
    # テスト用のPDFファイルパス
    test_pdf_paths = [
        # "docs/01.pdf",
        "docs/r5_doukou.pdf",
        # "docs/000108043.pdf"
    ]
    test_converter(test_pdf_paths, debug_mode=args.debug)
