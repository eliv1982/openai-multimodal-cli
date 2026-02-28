import argparse
import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_image(
    prompt: str,
    output_path: Path,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
) -> None:
    """
    Generate an image from text using OpenAI and save it to a file.
    """
    # Загружаем переменные окружения из .env (если файл есть)
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Не найден OPENAI_API_KEY. Укажите его в .env или переменных окружения.")

    client = OpenAI()

    # gpt-image-1 всегда возвращает b64_json; dall-e-3 требует явного запроса
    gen_kwargs = {"model": model, "prompt": prompt, "size": size, "n": 1}
    if "dall-e" in model.lower():
        gen_kwargs["response_format"] = "b64_json"

    result = client.images.generate(**gen_kwargs)

    image_data = result.data[0]
    image_base64 = getattr(image_data, "b64_json", None)
    if image_base64 is None:
        raise ValueError(
            f"Модель {model} вернула URL вместо base64. "
            "Для DALL-E укажите response_format='b64_json' или используйте gpt-image-1."
        )
    image_bytes = base64.b64decode(image_base64)
    output_path.write_bytes(image_bytes)

    logger.info("Изображение сохранено в: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерация изображений через OpenAI: текстовый промпт -> изображение"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Футуристичный город на закате в стиле цифровой иллюстрации",
        help="Текстовый промпт для генерации изображения "
        "(по умолчанию: футуристичный город на закате)",
    )
    parser.add_argument(
        "output_image",
        type=Path,
        nargs="?",
        default=Path("image.png"),
        help="Путь к выходному файлу с изображением (по умолчанию: image.png)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-image-1",
        help="Модель для генерации изображений (по умолчанию: gpt-image-1)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Размер: 1024x1024 (квадрат), 1536x1024 (альбом/16:9, для инфографики), 1024x1536 (портрет)",
    )

    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        output_path=args.output_image,
        model=args.model,
        size=args.size,
    )


if __name__ == "__main__":
    main()


