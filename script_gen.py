"""
Генерация текстового сценария для озвучки по заданной теме.
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompts import CHECKLIST_REDOMICILIATION, get_tts_script_prompt

logger = logging.getLogger(__name__)


def verify_script_checklist(script_text: str, checklist: list[str], model: str = "gpt-4o") -> list[str]:
    """
    Проверяет, что в сценарии отражены пункты чек-листа.
    Возвращает список пунктов, которые не найдены (пустой список — всё ок).
    """
    load_dotenv()
    import os
    if not os.getenv("OPENAI_API_KEY"):
        return []
    client = OpenAI()
    items = "\n".join(f"- {s}" for s in checklist)
    prompt = f"""По чек-листу ниже проверь текст сценария. Верни JSON-массив строк — только те пункты чек-листа, которые в тексте НЕ отражены (ни явно, ни по смыслу). Если все отражены — верни пустой массив [].

Чек-лист:
{items}

Текст сценария:
---
{script_text[:6000]}
---

Только JSON, например ["пункт 1", "пункт 2"] или []."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Отвечай только валидным JSON-массивом строк."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        missing = json.loads(content)
        return missing if isinstance(missing, list) else []
    except Exception as e:
        logger.warning("Авто-проверка по чек-листу не выполнена: %s", e)
        return []


def generate_script(
    topic: str,
    model: str = "gpt-4o",
    channel: str | None = None,
    verify_checklist: bool = False,
) -> str:
    """
    Генерирует текстовый сценарий для озвучки по теме.

    Args:
        topic: Тема сценария.
        model: Модель GPT для генерации.
        channel: Канал использования (лента, презентация, встреча).
        verify_checklist: Проверить сценарий по чек-листу (для темы редомициляции).

    Returns:
        Текст сценария (1,5–2 мин озвучки, темп 150–160 слов/мин).
    """
    load_dotenv()
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения."
        )

    client = OpenAI()
    prompt = get_tts_script_prompt(topic, channel=channel)

    logger.info("Генерация сценария по теме: %s", topic)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ты автор образовательных сценариев. Пиши только текст сценария, без заголовков и пометок.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    text = response.choices[0].message.content.strip()
    logger.info("Сценарий сгенерирован, объём: %d символов", len(text))

    if verify_checklist and ("редомициляци" in topic.lower() or "redomiciliation" in topic.lower()):
        missing = verify_script_checklist(text, CHECKLIST_REDOMICILIATION, model=model)
        if missing:
            logger.warning("По чек-листу не отражены пункты: %s", missing)
        else:
            logger.info("Авто-проверка по чек-листу: все пункты отражены.")

    return text
