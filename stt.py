import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def transcribe_audio(input_path: Path, output_path: Path, model: str = "gpt-4o-mini-transcribe") -> None:
    """
    Transcribe an audio file using OpenAI and save the text to a file.
    """
    # Загружаем переменные окружения из .env (если файл есть)
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения.")

    client = OpenAI()

    with input_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
        )

    output_path.write_text(transcription.text, encoding="utf-8")
    logger.info("Транскрипция сохранена в: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STT (speech-to-text) с помощью OpenAI: аудио -> текстовый файл"
    )
    parser.add_argument(
        "input_audio",
        type=Path,
        help="Путь к входному аудиофайлу (например, WAV/MP3/OGG)",
    )
    parser.add_argument(
        "output_text",
        type=Path,
        nargs="?",
        default=Path("transcription.txt"),
        help="Путь к выходному текстовому файлу (по умолчанию: transcription.txt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-transcribe",
        help="Модель для распознавания (по умолчанию: gpt-4o-mini-transcribe)",
    )

    args = parser.parse_args()

    if not args.input_audio.exists():
        raise SystemExit(f"Файл не найден: {args.input_audio}")

    transcribe_audio(args.input_audio, args.output_text, model=args.model)


if __name__ == "__main__":
    main()


