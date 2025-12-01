import os
import csv
from datetime import datetime
import ast
import asyncio
from pathlib import Path
from dotenv import load_dotenv
# from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
# from langchain_google_genai import GoogleGenerativeAI
from ragas.metrics import (
    LLMContextRecall,
    ContextEntityRecall as LegacyContextEntityRecall,
    ContextRelevance as LegacyContextRelevance,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    NoiseSensitivity as LegacyNoiseSensitivity,
    AnswerAccuracy as LegacyAnswerAccuracy,
    ResponseGroundedness,
)
from ragas.metrics.collections import (
    ContextPrecision,
    ContextUtilization,
    ContextEntityRecall as CollectionsContextEntityRecall,
    NoiseSensitivity as CollectionsNoiseSensitivity,
    Faithfulness as CollectionsFaithfulness,
    AnswerAccuracy as CollectionsAnswerAccuracy,
    ContextRelevance as CollectionsContextRelevance,
)
from ragas import evaluate
from ragas import EvaluationDataset

from rag_evaluate.config import (
    PROJECT_ROOT,
    EVAL_EVALUATOR_LLM_MODEL_NAME,
    EVAL_EVALUATOR_LLM_TEMPERATURE,
    EVAL_DATASET_CSV_PATH,
    EVAL_RESULT_OUTPUT_DIR,
    EVAL_METRICS,
)


def _resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


METRIC_REGISTRY: dict[str, callable] = {
    # Legacy metrics (クラスベース)
    "llm_context_recall": lambda llm: LLMContextRecall(llm=llm),
    "context_entity_recall": lambda llm: LegacyContextEntityRecall(llm=llm),
    "context_relevance": lambda llm: LegacyContextRelevance(llm=llm),

    "llm_context_precision_without_reference": lambda llm: LLMContextPrecisionWithoutReference(llm=llm),
    "llm_context_precision_with_reference": lambda llm: LLMContextPrecisionWithReference(llm=llm),
    "nonllm_context_precision_with_reference": lambda llm: NonLLMContextPrecisionWithReference(),
    "nonllm_context_recall": lambda llm: NonLLMContextRecall(),
    "legacy.noise_sensitivity": lambda llm: LegacyNoiseSensitivity(llm=llm),
    "legacy.answer_accuracy": lambda llm: LegacyAnswerAccuracy(llm=llm),
    "response_groundedness": lambda llm: ResponseGroundedness(llm=llm),

    # Collections API metrics
    "collections.context_precision": lambda llm: ContextPrecision(llm=llm),
    "collections.context_utilization": lambda llm: ContextUtilization(llm=llm),
    "collections.context_entity_recall": lambda llm: CollectionsContextEntityRecall(llm=llm),
    "collections.noise_sensitivity": lambda llm: CollectionsNoiseSensitivity(llm=llm),
    "collections.noise_sensitivity_irrelevant": lambda llm: CollectionsNoiseSensitivity(llm=llm, mode="irrelevant"),
    "collections.faithfulness": lambda llm: CollectionsFaithfulness(llm=llm),
    "collections.answer_accuracy": lambda llm: CollectionsAnswerAccuracy(llm=llm),
    "collections.context_relevance": lambda llm: CollectionsContextRelevance(llm=llm),
}


def build_metrics(evaluator_llm) -> list:
    metrics = []
    for name in EVAL_METRICS:
        factory = METRIC_REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"未知のメトリクス名です: {name}")
        metrics.append(factory(evaluator_llm))
    return metrics


def _parse_list_field(raw_value: str, field_name: str, row_number: int) -> list[str]:
    if raw_value is None:
        return []

    stripped = raw_value.strip()
    if not stripped:
        return []

    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"CSVの列 '{field_name}' の{row_number}行目をリストとして解釈できません。"
        ) from exc

    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, str):
        return [parsed]

    raise ValueError(
        f"CSVの列 '{field_name}' の{row_number}行目は想定外の形式です: {type(parsed)}"
    )


def load_dataset_from_csv(csv_path: str) -> list[dict]:
    resolved_path = _resolve_csv_path(csv_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"評価用CSVファイルが見つかりません: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = {
            "user_input",
            "response",
            "reference",
            "reference_contexts",
            "retrieved_contexts",
        }

        if reader.fieldnames is None:
            raise ValueError("CSVにヘッダー行が存在しません。")

        missing_columns = expected_columns.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSVに必要な列が不足しています: {missing}")

        dataset: list[dict] = []
        for row_index, row in enumerate(reader, start=2):
            dataset.append(
                {
                    "user_input": row.get("user_input", ""),
                    "response": row.get("response", ""),
                    "reference": row.get("reference", ""),
                    "reference_contexts": _parse_list_field(
                        row.get("reference_contexts", ""),
                        "reference_contexts",
                        row_index,
                    ),
                    "retrieved_contexts": _parse_list_field(
                        row.get("retrieved_contexts", ""),
                        "retrieved_contexts",
                        row_index,
                    ),
                }
            )

    return dataset


# 非同期処理のメイン部分を定義
async def main():
    """
    python -m src.ragas.evaluate_rag
    """
    load_dotenv()

    model_name = EVAL_EVALUATOR_LLM_MODEL_NAME # gpt-5-miniは20個の評価で0.41$

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=model_name,
            temperature=EVAL_EVALUATOR_LLM_TEMPERATURE
        ),
        bypass_temperature=True
    )

    # gemini_api_key = os.getenv("GEMINI_API_KEY")
    # if not gemini_api_key:
    #     raise ValueError("GEMINI_API_KEYが設定されていません。.envファイルを確認してください。")

    # evaluator_llm = LangchainLLMWrapper(GoogleGenerativeAI(
    #     google_api_key=gemini_api_key,
    #     model="gemini-2.5-pro",
    #     temperature=0.2,
    #     max_tokens=10000,
    #     thinking_budget=128
    # ))

    # metrics
    metrics = build_metrics(evaluator_llm)

    dataset_csv_path = EVAL_DATASET_CSV_PATH

    dataset = load_dataset_from_csv(dataset_csv_path)
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # evaluate
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm
    )

    print(result) # evaluate()はprintすると平均のみ

    result_df = result.to_pandas()

    project_root = PROJECT_ROOT
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_csv_path = project_root / EVAL_RESULT_OUTPUT_DIR / f"rag_result_{timestamp}.csv"

    # 出力ディレクトリが存在しない場合は作成
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig", # BOMつきで出力し、Excelで文字化けしない
        quoting=csv.QUOTE_ALL,
        doublequote=True,
    )


if __name__ == "__main__":
    asyncio.run(main())