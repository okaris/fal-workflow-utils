import hashlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, download_file, get_image_size
from fal.toolkit.image import ImageSize


@dataclass
class Timings:
    start: float
    end: float | None = None

    @property
    def took(self):
        return (self.end or time.perf_counter()) - self.start


@contextmanager
def timed() -> Iterator[Timings]:
    start = time.perf_counter()
    timer = Timings(start)
    try:
        yield timer
    finally:
        timer.end = time.perf_counter()


@lru_cache(maxsize=64)
def read_image_from_url(url: str, convert_to_rgb: bool = True):
    from fastapi import HTTPException
    from PIL import Image

    TEMP_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
    }

    try:
        request = Request(url, headers=TEMP_HEADERS)
        response = urlopen(request)
        image = Image.open(response)
        if convert_to_rgb:
            image = image.convert("RGB")
    except:
        import traceback

        traceback.print_exc()
        raise HTTPException(422, f"Could not load image from url: {url}")

    return image


def _hash_url(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def download_model_weights(url: str, force: bool = False):

    # This is not a protected path, so the user may change stuff internally
    weights_dir = Path(FAL_MODEL_WEIGHTS_DIR) / _hash_url(url)

    if weights_dir.exists() and not force:
        try:
            weights_path = next(weights_dir.glob("*"))
            return weights_path
        # The model weights directory is empty, so we need to download the weights
        except StopIteration:
            pass

    return download_file(
        url,
        target_dir=weights_dir,
        force=force,
    )


def reshape_image_to_latent_space(image, multiples_of=8):
    # Reshape the given image's dimensions to multiples of 8 so they are
    # compatible with the latent space of diffusion models.
    from PIL import Image

    width, height = image.size
    new_width = width - (width % multiples_of)
    new_height = height - (height % multiples_of)
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def get_compatible_image_size(
    input_image_size,
    multiples_of=8,
    min_w: int | None = None,
    min_h: int | None = None,
    max_w: int | None = None,
    max_h: int | None = None,
):
    image_size = get_image_size(input_image_size)
    width = image_size.width - (image_size.width % multiples_of)
    height = image_size.height - (image_size.height % multiples_of)
    if min_w is not None:
        width = max(width, min_w)
    if min_h is not None:
        height = max(height, min_h)
    if max_w is not None:
        width = min(width, max_w)
    if max_h is not None:
        height = min(height, max_h)
    return ImageSize(width=width, height=height)


def resize_by_preserving_aspect_ratio(
    image: "PILImage",
    max_width: int,
    max_height: int,
) -> "PILImage":
    from PIL import Image

    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def filter_by(
    has_nsfw_concepts: list[bool],
    images: list["PILImage"],
) -> list["PILImage"]:
    from PIL import Image as PILImage

    return [
        (
            PILImage.new("RGB", (image.width, image.height), (0, 0, 0))
            if has_nsfw
            else image
        )
        for image, has_nsfw in zip(images, has_nsfw_concepts)
    ]


def to_sdxl_dimensions(image_size: ImageSize) -> ImageSize:
    # List of SDXL dimensions
    allowed_dimensions = [
        (512, 2048),
        (512, 1984),
        (512, 1920),
        (512, 1856),
        (576, 1792),
        (576, 1728),
        (576, 1664),
        (640, 1600),
        (640, 1536),
        (704, 1472),
        (704, 1408),
        (704, 1344),
        (768, 1344),
        (768, 1280),
        (832, 1216),
        (832, 1152),
        (896, 1152),
        (896, 1088),
        (960, 1088),
        (960, 1024),
        (1024, 1024),
        (1024, 960),
        (1088, 960),
        (1088, 896),
        (1152, 896),
        (1152, 832),
        (1216, 832),
        (1280, 768),
        (1344, 768),
        (1408, 704),
        (1472, 704),
        (1536, 640),
        (1600, 640),
        (1664, 576),
        (1728, 576),
        (1792, 576),
        (1856, 512),
        (1920, 512),
        (1984, 512),
        (2048, 512),
    ]
    # Calculate the aspect ratio
    aspect_ratio = image_size.width / image_size.height
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    # Find the closest allowed dimensions that maintain the aspect ratio
    width, height = min(
        allowed_dimensions,
        key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio),
    )
    return ImageSize(width=width, height=height)
