"""
Генерация текстового сценария для озвучки по заданной теме.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompts import PROMPT_SCRIPT_TTS

logger = logging.getLogger(__name__)


def generate_script(topic: str, model: str = "gpt-4o") -> str:
    """
    Генерирует текстовый сценарий для озвучки по теме.

    Args:
        topic: Тема сценария.
        model: Модель GPT для генерации.

    Returns:
        Текст сценария (1,5–2 мин озвучки).
    """
    load_dotenv()
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения."
        )

    client = OpenAI()
    prompt = PROMPT_SCRIPT_TTS.format(topic=topic)

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
    return text
