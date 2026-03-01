"""
Модуль для автоматической сборки короткого обучающего видео на заданную тему.

Использует GPT для генерации сценария, image_gen для изображений,
tts для озвучки и moviepy для сборки видео.
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from image_gen import generate_image
from prompts import PROMPT_SYSTEM_SCRIPT, get_script_prompt
from tts import text_to_speech

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Целевое разрешение видео (1080p Full HD — без ряби, плоские заливки)
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
# Без crossfade — избегаем ряби/артефактов на стыках кадров
CROSSFADE_DURATION = 0.0


def _ensure_temp_dir() -> Path:
    """Создаёт папку temp, если её нет."""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def _generate_script(topic: str, model: str, channel: str | None = None) -> list[dict[str, str]]:
    """
    Генерирует сценарий через GPT.

    Returns:
        Список объектов с полями part_text и image_prompt.
    """
    load_dotenv()
    client = OpenAI()

    prompt = get_script_prompt(topic, channel=channel)

    logger.info("Генерация сценария через GPT...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM_SCRIPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()
    # Убираем возможные markdown-обёртки
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        script = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Некорректный JSON от GPT: %s", e)
        raise ValueError(f"Не удалось распарсить сценарий: {e}") from e

    if not isinstance(script, list) or len(script) < 1:
        raise ValueError("Сценарий должен содержать минимум одну часть")

    for i, part in enumerate(script):
        if not isinstance(part, dict) or "part_text" not in part or "image_prompt" not in part:
            raise ValueError(f"Часть {i + 1} должна содержать part_text и image_prompt")

    logger.info("Сценарий успешно сгенерирован")
    return script


def _load_script(script_path: Path) -> list[dict[str, str]]:
    """Загружает сценарий из JSON-файла."""
    if not script_path.exists():
        raise FileNotFoundError(f"Файл сценария не найден: {script_path}")

    try:
        content = script_path.read_text(encoding="utf-8")
        script = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Некорректный JSON в файле сценария: {e}") from e

    if not isinstance(script, list) or len(script) < 1:
        raise ValueError("Сценарий должен содержать минимум одну часть")

    for i, part in enumerate(script):
        if not isinstance(part, dict) or "part_text" not in part or "image_prompt" not in part:
            raise ValueError(f"Часть {i + 1} должна содержать part_text и image_prompt")

    logger.info("Сценарий загружен из файла")
    return script


def create_video(
    topic: str = "редомициляция",
    output_path: str | Path = "video.mp4",
    script_path: str | Path | None = None,
    voice: str = "onyx",
    model: str = "gpt-4o",
    size: str = "1536x1024",
    keep_temp: bool = False,
    channel: str | None = None,
) -> None:
    """
    Создаёт обучающее видео на заданную тему (1080p, чек-лист в сценарии).

    Args:
        topic: Тема видео (используется при генерации сценария).
        output_path: Путь для сохранения итогового видео.
        script_path: Путь к JSON-файлу сценария (опционально).
        voice: Голос для TTS (onyx, nova, alloy и др.).
        model: Модель GPT для генерации сценария.
        size: Размер генерируемых изображений.
        keep_temp: Не удалять временные файлы после сборки.
        channel: Канал использования (лента, презентация, встреча).
    """
    try:
        from moviepy import (
            AudioFileClip,
            ImageClip,
            concatenate_videoclips,
            vfx,
        )
        _MOVIEPY_V2 = True
    except ImportError:
        try:
            from moviepy.editor import (
                AudioFileClip,
                ImageClip,
                concatenate_videoclips,
            )
            vfx = None
            _MOVIEPY_V2 = False
        except ImportError as e:
            raise ImportError(
                "Для создания видео необходим moviepy. Установите: pip install moviepy"
            ) from e

    load_dotenv()
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения."
        )

    output_path = Path(output_path)
    temp_dir = _ensure_temp_dir()
    temp_files: list[Path] = []

    try:
        # 1. Получение сценария
        if script_path is not None:
            script = _load_script(Path(script_path))
        else:
            script = _generate_script(topic, model, channel=channel)

        # 2. Генерация изображений и сбор текста
        parts_text: list[str] = []
        image_paths: list[Path] = []

        for i, part in enumerate(script):
            part_text = part["part_text"]
            image_prompt = part["image_prompt"]
            parts_text.append(part_text)

            img_path = temp_dir / f"scene_{i}.png"
            temp_files.append(img_path)
            logger.info("Генерация изображения %d/%d...", i + 1, len(script))
            generate_image(image_prompt, img_path, model="gpt-image-1", size=size)
            image_paths.append(img_path)

        # 3. Генерация аудио
        full_text = " ".join(parts_text)
        audio_path = temp_dir / "narration.mp3"
        temp_files.append(audio_path)
        logger.info("Генерация озвучки...")
        text_to_speech(full_text, audio_path, voice=voice)

        # 4. Длительность аудио и пропорциональное распределение
        audio_clip = AudioFileClip(str(audio_path))
        total_duration = audio_clip.duration
        logger.info("Длительность аудио: %.1f сек", total_duration)

        char_counts = [len(p) for p in parts_text]
        total_chars = sum(char_counts)
        if total_chars == 0:
            total_chars = 1
        scene_durations = [
            total_duration * (c / total_chars) for c in char_counts
        ]

        # 5. Создание видеоклипов для каждой сцены
        video_clips = []

        for i, (img_path, duration, part_text) in enumerate(
            zip(image_paths, scene_durations, parts_text)
        ):
            # Загружаем изображение и масштабируем (MoviePy 2: resized/with_duration, v1: resize/set_duration)
            img_clip = ImageClip(str(img_path))
            resize_fn = getattr(img_clip, "resized", None) or getattr(img_clip, "resize")
            img_clip = resize_fn((VIDEO_WIDTH, VIDEO_HEIGHT))
            dur_fn = getattr(img_clip, "with_duration", None) or getattr(img_clip, "set_duration")
            img_clip = dur_fn(duration)

            # Плавный переход только если длительность > 0 (иначе рябь/артефакты)
            if CROSSFADE_DURATION > 0:
                if vfx is not None:
                    img_clip = img_clip.with_effects([
                        vfx.CrossFadeIn(CROSSFADE_DURATION),
                        vfx.CrossFadeOut(CROSSFADE_DURATION),
                    ])
                else:
                    img_clip = img_clip.crossfadein(CROSSFADE_DURATION).crossfadeout(
                        CROSSFADE_DURATION
                    )

            # Текстовые заголовки отключены: стандартный шрифт плохо отображает кириллицу (квадраты/символы).
            # Картинки без надписи — без ряби и без артефактов.
            final_clip = img_clip
            video_clips.append(final_clip)

        # 6. Склейка сцен и наложение аудио
        logger.info("Сборка видео...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        set_audio_fn = getattr(final_video, "with_audio", None) or getattr(final_video, "set_audio")
        final_video = set_audio_fn(audio_clip)

        # 7. Сохранение
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=24,
            logger=None,
        )

        # Закрываем клипы
        audio_clip.close()
        final_video.close()

        logger.info("Видео сохранено: %s", output_path)

    finally:
        if not keep_temp and temp_files:
            for f in temp_files:
                try:
                    if f.exists():
                        f.unlink()
                except OSError as e:
                    logger.warning("Не удалось удалить %s: %s", f, e)


if __name__ == "__main__":
    # Тестовый запуск
    import argparse

    parser = argparse.ArgumentParser(description="Создание обучающего видео")
    parser.add_argument("--topic", default="редомициляция")
    parser.add_argument("--output", default="video.mp4")
    parser.add_argument("--script-file", default=None)
    parser.add_argument("--voice", default="onyx")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--size", default="1536x1024", help="Размер кадров (альбом 1536x1024 — без обрезки)")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--channel", default=None, help="Канал: лента, презентация, встреча")
    args = parser.parse_args()

    create_video(
        topic=args.topic,
        output_path=args.output,
        script_path=args.script_file,
        voice=args.voice,
        model=args.model,
        size=args.size,
        keep_temp=args.keep_temp,
        channel=args.channel,
    )
