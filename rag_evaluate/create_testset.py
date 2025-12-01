"""
Ragas用テストセット生成スクリプト
python src/ragas/create_testset.py
"""

from typing import List, Sequence, Iterable, Tuple
import csv
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    apply_transforms,
)


load_dotenv()

project_root = Path(__file__).resolve().parents[1]
def build_knowledge_graph_from_documents(docs: Sequence[Document]) -> KnowledgeGraph:
    """ドキュメント群からKnowledge Graphを構築する。"""
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )
    return kg


def apply_default_transforms(
    kg: KnowledgeGraph,
    llm: LangchainLLMWrapper,
    headline_max: int = 20,
    splitter_max_tokens: int = 1500,
) -> None:
    """見出し抽出・分割・キーフレーズ抽出の変換を適用する。"""
    transforms = [
        HeadlinesExtractor(llm=llm, max_num=headline_max),
        HeadlineSplitter(max_tokens=splitter_max_tokens),
        KeyphrasesExtractor(llm=llm),
    ]
    apply_transforms(kg, transforms=transforms)


def build_query_distribution(
    llm: LangchainLLMWrapper,
    headline_weight: float,
    keyphrase_weight: float,
) -> List[Tuple[SingleHopSpecificQuerySynthesizer, float]]:
    """シングルホップクエリの重み付き分布を生成する。"""
    return [
        (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="headlines"), headline_weight),
        (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="keyphrases"), keyphrase_weight),
    ]


def generate_testset(
    kg: KnowledgeGraph,
    llm: LangchainLLMWrapper,
    embeddings: OpenAIEmbeddings,
    personas: Sequence[Persona],
    query_distribution: Iterable[Tuple[SingleHopSpecificQuerySynthesizer, float]],
    testset_size: int,
):
    """指定設定でテストセットを生成する。"""
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=list(personas),
    )
    return generator.generate(testset_size=testset_size, query_distribution=list(query_distribution))


def save_testset_to_csv(testset, output_path: Path) -> None:
    """生成されたテストセットをCSVで保存する。"""
    df = testset.to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig", # BOMつきで出力し、Excelで文字化けしない
        quoting=csv.QUOTE_ALL,
        doublequote=True,
    )
    print(f"\n💾 テストセットを保存しました: {output_path}")


def main() -> None:
    # ===== ハードコード設定 ====================================================
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。") 
    
    # llm_model_name = "gpt-5" # 2個作成で1$ 20個作成で2$?
    llm_model_name = "gpt-5-mini" # 20個作成で0.38$
    embedding_model_name = "text-embedding-3-small"

    # gpt-5-miniはtemperature=1のみサポート（デフォルト値）
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
        model=llm_model_name,
        temperature=0.2 # デフォルトが0.01になっておりgpt-5は対応していない
        ),
        bypass_temperature=True  # temperatureをユーザで変更できるようにする設定
    )

    openai_client = openai.OpenAI(api_key=openai_api_key)
    generator_embeddings = OpenAIEmbeddings(client=openai_client, model=embedding_model_name)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_csv_path = project_root / "outputs" / "testdatas" / "testsets" / f"testset_{timestamp}.csv"

    testset_size = 20
    headline_weight = 0.5
    keyphrase_weight = 0.5

    chunk_mds_glob_pattern = "outputs/pdf2md_per_pages/*.md" # testset作成用マークダウンファイル
    print("\n取得したファイルリスト:")
    for file in glob.glob(chunk_mds_glob_pattern):
        print(file)

    personas = [
        Persona(
            name="金融行政アナリスト",
            role_description="金融庁のモニタリング担当として、保険会社の社会的役割や諸課題への対応状況を洞察し、政策判断に活かすための情報を必要とする。",
        ),
        Persona(
            name="保険会社経営企画担当",
            role_description="少子高齢化やデジタル化を踏まえた持続可能なビジネスモデル構築を検討し、営業チャネルや商品開発の方向性を探っている。",
        ),
        Persona(
            name="リスクマネジメント責任者",
            role_description="自然災害リスクや再保険料率の上昇に対応するため、異常危険準備金の活用や水災料率の細分化などの実務的な施策を比較検討している。",
        ),
        Persona(
            name="神戸市観光政策担当",
            role_description="令和5年度神戸市観光動向調査を踏まえ、女性比率や60歳以上来訪者が多い地区への施策検討のため、地区別属性データを精査して観光行政の改善案を導きたい。",
        ),
        Persona(
            name="ハーバーランド集客マーケター",
            role_description="神戸港エリアの情報収集チャネルの差分（旅行前後のインターネット利用割合など）を把握し、再来訪意向とイベント施策を最適化するための示唆を求めている。",
        ),
        Persona(
            name="神戸市交通政策プランナー",
            role_description="地区ごとの主な交通手段（西北神での車利用や北野での新幹線割合など）を分析し、観光客の移動動線に合わせた交通インフラ・周遊施策を設計したい。",
        ),
    ]

    # ===========================================================================

    chunk_glob_absolute = project_root / Path(chunk_mds_glob_pattern)
    matched_markdown = sorted(Path(p) for p in glob.glob(str(chunk_glob_absolute)))
    if not matched_markdown:
        raise FileNotFoundError(
            f"パターン '{chunk_glob_absolute}' に一致するMarkdownが見つかりません。"
            " `chunk_mds_glob_pattern` を確認し、必要に応じて `create_mds_from_chunks.py` を実行してください。"
        )

    loader = DirectoryLoader(str(project_root), glob=chunk_mds_glob_pattern)
    docs = loader.load()
    kg = build_knowledge_graph_from_documents(docs)
    apply_default_transforms(kg, generator_llm)

    # query_distribution生成
    query_distribution = build_query_distribution(
        llm=generator_llm,
        headline_weight=headline_weight,
        keyphrase_weight=keyphrase_weight,
    )

    # testset生成
    testset = generate_testset(
        kg=kg,
        llm=generator_llm,
        embeddings=generator_embeddings,
        personas=personas,
        query_distribution=query_distribution,
        testset_size=testset_size,
    )

    # testset保存
    save_testset_to_csv(testset=testset, output_path=output_csv_path)


if __name__ == "__main__":
    main()
