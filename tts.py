import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def text_to_speech(
    text: str,
    output_path: Path,
    model: str = "gpt-4o-mini-tts",
    voice: str = "alloy",
) -> None:
    """
    Convert text to speech using OpenAI and save audio to a file.
    """
    # Загружаем переменные окружения из .env (если файл есть)
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения.")

    client = OpenAI()

    # Используем потоковую запись в файл, чтобы не держать аудио в памяти
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(str(output_path))

    logger.info("Аудио сохранено в: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTS (text-to-speech) с помощью OpenAI: текст -> аудиофайл"
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default="Привет! Это тестовый пример синтеза речи через OpenAI.",
        help="Текст для озвучивания (по умолчанию: короткий тестовый текст на русском)",
    )
    parser.add_argument(
        "output_audio",
        type=Path,
        nargs="?",
        default=Path("speech.mp3"),
        help="Путь к выходному аудиофайлу (по умолчанию: speech.mp3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-tts",
        help="Модель для синтеза речи (по умолчанию: gpt-4o-mini-tts)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="alloy",
        help="Голос для синтеза (по умолчанию: alloy)",
    )

    args = parser.parse_args()

    text_to_speech(
        text=args.text,
        output_path=args.output_audio,
        model=args.model,
        voice=args.voice,
    )


if __name__ == "__main__":
    main()


