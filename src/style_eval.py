import os
import json
import re
import statistics
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .brand_chain import ask
from app_lc import STYLE, BASE
from schemas import Grade

load_dotenv(BASE / ".env", override=True)
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)


# Простые проверки до LLM
def rule_checks(text: str) -> int:
    """Оценивает текст по набору правил стиля бренда на основе эвристик.

    Начальный балл — 100. Снимаются баллы за:
    - наличие эмодзи (–20),
    - использование трёх и более восклицательных знаков подряд (–10),
    - превышение длины 600 символов (–10).

    Итоговый балл не может быть ниже 0.

    Args:
        text (str): Проверяемый текст ответа.

    Returns:
        int: Оценка от 0 до 100.
    """
    score = 100
    # 1) Без эмодзи
    if re.search(r"[\U0001F300-\U0001FAFF]", text):
        score -= 20
    # 2) Без крика!!!
    if "!!!" in text:
        score -= 10
    # 3) Длина
    if len(text) > 600:
        score -= 10
    return max(score, 0)


LLM = ChatOpenAI(model=os.getenv("EVAL_MODEL_NAME","gpt-4o-mini"), temperature=0)
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"Ты — строгий ревьюер соответствия голосу бренда {STYLE['brand']}"),
    ("system", f"Тон: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
               f"Обязательно: {', '.join(STYLE['tone']['must_include'])}."),
    ("human", "Ответ ассистента:\n{answer}\n\nДай целочисленный score 0..100 и краткие заметки почему.")
])


def llm_grade(text: str) -> Grade:
    """Оценивает качество стиля текста с помощью LLM в соответствии с стиль-гайдом.

    Модель анализирует ответ ассистента и возвращает структурированную оценку:
    целочисленный балл (0–100) и пояснительные заметки.

    Args:
        text (str): Текст ответа для оценки.

    Returns:
        Grade: Pydantic-модель с полями `score` (int) и `notes` (str).
    """
    parser = LLM.with_structured_output(Grade)
    return (GRADE_PROMPT | parser).invoke({"answer": text})


def eval_batch(prompts: List[str]) -> dict:
    """Проводит комплексную оценку серии ответов модели на заданные промпты.

    Для каждого промпта:
    1. Получает ответ от основной системы (через функцию `ask`),
    2. Оценивает его по правилам (`rule_checks`),
    3. Получает LLM-оценку (`llm_grade`),
    4. Вычисляет финальную оценку как взвешенную сумму: 40% — правила, 60% — LLM.

    Результаты сохраняются в файл `reports/style_eval.json` в читаемом JSON-формате.

    Args:
        prompts (List[str]): Список входных промптов для тестирования.

    Returns:
        dict: Словарь с ключами:
            - `mean_final`: средний финальный балл по всем промптам,
            - `items`: список детализированных результатов по каждому промпту.
    """
    results = []
    for p in prompts:
        reply = ask(p)
        rule = rule_checks(reply.answer)
        g = llm_grade(reply.answer)
        final = int(0.4 * rule + 0.6 * g.score)
        results.append({
            "prompt": p,
            "answer": reply.answer,
            "actions": reply.actions,
            "tone_model": reply.tone,
            "rule_score": rule,
            "llm_score": g.score,
            "final": final,
            "notes": g.notes
        })
    mean_final = round(statistics.mean(r["final"] for r in results), 2)
    out = {"mean_final": mean_final, "items": results}
    (REPORTS / "style_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    eval_prompts = (BASE / "data/eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    print(eval_prompts)
    report = eval_batch(eval_prompts)
    print("Средний балл:", report["mean_final"])
    print("Отчёт:", REPORTS / "style_eval.json")
