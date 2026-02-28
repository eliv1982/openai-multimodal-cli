"""
Единая точка входа CLI для мультимодальных инструментов OpenAI.

Поддерживаемые команды:
  tts     - текст в речь (TTS)
  stt     - речь в текст (STT)
  img     - генерация изображений по текстовому промпту
  narrate - тема → сценарий → озвучка (сценарий сохраняется в файл)
  video   - создание обучающего видео на заданную тему
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_api_key() -> None:
    """Проверяет наличие OPENAI_API_KEY и завершает работу при его отсутствии."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения.")
        sys.exit(1)


def cmd_tts(args: argparse.Namespace) -> None:
    """Команда TTS: текст в речь."""
    from tts import text_to_speech

    text_to_speech(
        text=args.text,
        output_path=Path(args.output),
        model=args.model,
        voice=args.voice,
    )


def cmd_stt(args: argparse.Namespace) -> None:
    """Команда STT: речь в текст."""
    from stt import transcribe_audio

    input_path = Path(args.input_audio)
    if not input_path.exists():
        logger.error("Файл не найден: %s", input_path)
        sys.exit(1)

    transcribe_audio(
        input_path=input_path,
        output_path=Path(args.output),
        model=args.model,
    )


def cmd_img(args: argparse.Namespace) -> None:
    """Команда генерации изображений."""
    from image_gen import generate_image

    generate_image(
        prompt=args.prompt,
        output_path=Path(args.output),
        model=args.model,
        size=args.size,
    )


def cmd_narrate(args: argparse.Namespace) -> None:
    """Команда: тема → сценарий (в файл) → озвучка."""
    from script_gen import generate_script
    from tts import text_to_speech

    script_text = generate_script(topic=args.topic, model=args.model)
    script_path = Path(args.script_output)
    script_path.write_text(script_text, encoding="utf-8")
    logger.info("Сценарий сохранён: %s", script_path)

    text_to_speech(
        text=script_text,
        output_path=Path(args.output),
        model=args.tts_model,
        voice=args.voice,
    )
    logger.info("Озвучка сохранена: %s", args.output)


def cmd_video(args: argparse.Namespace) -> None:
    """Команда создания обучающего видео."""
    from video_gen import create_video

    create_video(
        topic=args.topic,
        output_path=Path(args.output),
        script_path=Path(args.script_file) if args.script_file else None,
        voice=args.voice,
        model=args.model,
        size=args.size,
        keep_temp=args.keep_temp,
    )


def main() -> None:
    """Главная функция с субпарсерами."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="CLI-инструменты для работы с мультимодальными моделями OpenAI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # --- TTS ---
    tts_parser = subparsers.add_parser("tts", help="Текст в речь (TTS)")
    tts_parser.add_argument("text", type=str, help="Текст для озвучивания")
    tts_parser.add_argument("output", type=str, default="speech.mp3", nargs="?", help="Путь к выходному аудиофайлу")
    tts_parser.add_argument("--voice", type=str, default="alloy", help="Голос (alloy, echo, fable, onyx, nova, shimmer)")
    tts_parser.add_argument("--model", type=str, default="gpt-4o-mini-tts", help="Модель TTS")
    tts_parser.set_defaults(func=cmd_tts)

    # --- STT ---
    stt_parser = subparsers.add_parser("stt", help="Речь в текст (STT)")
    stt_parser.add_argument("input_audio", type=str, help="Путь к входному аудиофайлу")
    stt_parser.add_argument("output", type=str, default="transcript.txt", nargs="?", help="Путь к выходному файлу")
    stt_parser.add_argument("--model", type=str, default="gpt-4o-mini-transcribe", help="Модель транскрибации")
    stt_parser.set_defaults(func=cmd_stt)

    # --- IMG ---
    img_parser = subparsers.add_parser("img", help="Генерация изображений по промпту")
    img_parser.add_argument("prompt", type=str, help="Текстовый промпт для генерации изображения")
    img_parser.add_argument("output", type=str, default="image.png", nargs="?", help="Путь к выходному файлу")
    img_parser.add_argument("--model", type=str, default="gpt-image-1", help="Модель (gpt-image-1 или dall-e-3)")
    img_parser.add_argument("--size", type=str, default="1024x1024", help="Размер изображения")
    img_parser.set_defaults(func=cmd_img)

    # --- NARRATE (доп. опция: тема → сценарий → озвучка) ---
    narrate_parser = subparsers.add_parser("narrate", help="Тема → сценарий (файл) → озвучка")
    narrate_parser.add_argument("--topic", type=str, required=True, help="Тема сценария")
    narrate_parser.add_argument("--output", type=str, default="narration.mp3", help="Аудиофайл")
    narrate_parser.add_argument("--script-output", type=str, default="script.txt", help="Файл с текстом сценария")
    narrate_parser.add_argument("--voice", type=str, default="nova", help="Голос")
    narrate_parser.add_argument("--model", type=str, default="gpt-4o", help="Модель GPT для сценария")
    narrate_parser.add_argument("--tts-model", type=str, default="gpt-4o-mini-tts", help="Модель TTS")
    narrate_parser.set_defaults(func=cmd_narrate)

    # --- VIDEO ---
    video_parser = subparsers.add_parser("video", help="Создание обучающего видео")
    video_parser.add_argument("--topic", type=str, default="редомициляция", help="Тема видео")
    video_parser.add_argument("--output", type=str, default="video.mp4", help="Путь к выходному видео")
    video_parser.add_argument("--script-file", type=str, default=None, help="Путь к JSON-файлу сценария (опционально)")
    video_parser.add_argument("--voice", type=str, default="onyx", help="Голос для озвучки")
    video_parser.add_argument("--model", type=str, default="gpt-4o", help="Модель GPT для сценария")
    video_parser.add_argument("--size", type=str, default="1536x1024", help="Размер кадров (1536x1024 — альбом, без обрезки)")
    video_parser.add_argument("--keep-temp", action="store_true", help="Не удалять временные файлы")
    video_parser.set_defaults(func=cmd_video)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    check_api_key()

    try:
        args.func(args)
    except Exception as e:
        logger.exception("Ошибка при выполнении команды: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
