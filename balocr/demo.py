"""Bounded CPU demo of the actual pretrained OCR base, without a BalOCR adapter."""
from functools import lru_cache
from threading import Lock
import time

from PIL import Image, ImageOps

from .model import load_model, predict

_model_lock = Lock()


def prepare_image(image):
    if image is None:
        raise ValueError('Upload a printed page or text crop first.')
    if not isinstance(image, Image.Image):
        raise ValueError('Upload a supported image file.')
    if image.width * image.height > 25_000_000:
        raise ValueError('Use an image below 25 megapixels, or crop the page.')
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', image.size, 'white')
        image = Image.alpha_composite(background, rgba)
    return image.convert('RGB')


def format_result(result, elapsed):
    status = f"{elapsed:.1f} seconds · {result['generated_tokens']} output tokens."
    if not result['finished']:
        status += ' **Output may be incomplete:** the length limit was reached. Increase the limit or use a smaller crop.'
    return result['text'], status


@lru_cache(maxsize=1)
def _load_base():
    import torch
    torch.set_num_threads(2)
    model, processor = load_model(device='cpu')
    model.eval()
    return model, processor


def recognize(image, max_tokens=768):
    image = prepare_image(image)
    if (isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, float))
            or not 128 <= max_tokens <= 2048 or int(max_tokens) != max_tokens):
        raise ValueError('Choose an output length from 128 to 2048 tokens.')
    started = time.monotonic()
    # Also serialize direct calls outside Gradio; concurrent loading duplicates
    # several GB of model weights on the small CPU host.
    with _model_lock:
        model, processor = _load_base()
        result = predict(model, processor, image, max_new_tokens=int(max_tokens), max_pixels=225792)
    return format_result(result, time.monotonic() - started)
