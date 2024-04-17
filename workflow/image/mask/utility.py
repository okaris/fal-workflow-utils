import fal
import PIL.Image
from fal.toolkit import (
    FAL_REPOSITORY_DIR,
    Image,
    clone_repository,
    download_model_weights,
)
from pydantic import BaseModel, Field

from workflow.common import read_image_from_url

CHECKPOINT = "https://huggingface.co/fal-ai/teed/resolve/main/5_model.pth"
TEED_REPO_NAME = "TEED"
TEED_REPO_PATH = FAL_REPOSITORY_DIR / TEED_REPO_NAME


class MaskInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/retoucher/GGsAolHXsAA58vn.jpeg",
        ],
    )


class MaskOutput(BaseModel):
    image: Image = Field(
        description="The mask",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/teed_output.png",
                content_type="image/png",
                width=1246,
                height=2048,
            )
        ],
    )


def invert_mask(
    input: MaskInput,
) -> MaskOutput:
    import PIL
    import cv2
    import numpy as np

    image = read_image_from_url(input.image_url)
    mask = np.array(image)
    inverted_mask = 255 - mask
    inverted_mask = PIL.Image.fromarray(inverted_mask)
    inverted_mask = Image.from_pil(inverted_mask)
    return MaskOutput(image=inverted_mask)

@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_invert_mask_on_fal(
    input: MaskInput,
) -> MaskOutput:
    return invert_mask(input)

class BlurMaskInput(MaskInput):
    radius: int = Field(
        description="The radius of the Gaussian blur.",
        examples=[5],
    )

def blur_mask(
    input: BlurMaskInput,
) -> MaskOutput:
    import PIL
    import cv2
    import numpy as np

    image = read_image_from_url(input.image_url)
    blurred_mask = image.filter(PIL.ImageFilter.GaussianBlur(input.radius))
    blurred_mask = Image.from_pil(blurred_mask)
    return MaskOutput(image=blurred_mask)
    

@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_blur_mask_on_fal(
    input: BlurMaskInput,
) -> MaskOutput:
    return blur_mask(input)

class GrowMaskInput(MaskInput):
    pixels: int = Field(
        description="The number of pixels to grow the mask.",
        examples=[5],
    )

def grow_mask(
    input: GrowMaskInput,
) -> MaskOutput:
    import PIL
    import cv2
    import numpy as np

    image = read_image_from_url(input.image_url)
    mask = np.array(image)
    grown_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    grown_mask = grown_mask.astype(np.float32)
    kernel = np.ones((2 * input.pixels + 1, 2 * input.pixels + 1), dtype=np.uint8)
    grown_mask = cv2.dilate(grown_mask, kernel, iterations=1)
    grown_mask = (grown_mask * 255).astype(np.uint8)
    grown_mask = PIL.Image.fromarray(grown_mask)
    grown_mask = Image.from_pil(grown_mask)
    fal_image = Image(
        url=grown_mask.url,
        content_type="image/png",
        width=grown_mask.width,
        height=grown_mask.height,
    )
    return MaskOutput(image=fal_image)
    

@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_grow_mask_on_fal(
    input: GrowMaskInput,
) -> MaskOutput:
    return grow_mask(input)

class ShrinkMaskInput(MaskInput):
    pixels: int = Field(
        description="The number of pixels to shrink the mask.",
        examples=[5],
    )

def shrink_mask(
    input: ShrinkMaskInput,
) -> MaskOutput:
    import PIL
    import cv2
    import numpy as np

    image = read_image_from_url(input.image_url)
    mask = np.array(image)
    shrunk_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    shrunk_mask = shrunk_mask.astype(np.float32)
    kernel = np.ones((2 * input.pixels + 1, 2 * input.pixels + 1), dtype=np.uint8)
    shrunk_mask = cv2.erode(shrunk_mask, kernel, iterations=1)
    shrunk_mask = (shrunk_mask * 255).astype(np.uint8)
    shrunk_mask = PIL.Image.fromarray(shrunk_mask)
    shrunk_mask = Image.from_pil(shrunk_mask)
    fal_image = Image(
        url=shrunk_mask.url,
        content_type="image/png",
        width=shrunk_mask.width,
        height=shrunk_mask.height,
    )
    return MaskOutput(image=fal_image)

@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_shrink_mask_on_fal(
    input: ShrinkMaskInput,
) -> MaskOutput:
    return shrink_mask(input)
    
class TransparentImageToMaskInput(MaskInput):
    threshold: int = Field(
        description="The threshold to convert the image to a mask.",
        examples=[128],
    )

def transparent_image_to_mask(
    input: ShrinkMaskInput,
) -> MaskOutput:
    import PIL
    import cv2
    import numpy as np

    image = read_image_from_url(input.image_url, convert_to_rgb=False)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    alpha = np.array(image)[:, :, 3]
    mask = np.where(alpha > input.threshold, 255, 0).astype(np.uint8)
    mask = PIL.Image.fromarray(mask)
    mask = Image.from_pil(mask)
    fal_image = Image(
        url=mask.url,
        content_type="image/png",
        width=mask.width,
        height=mask.height,
    )
    return MaskOutput(image=fal_image)

@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_transparent_image_to_mask_on_fal(
    input: TransparentImageToMaskInput,
) -> MaskOutput:
    return transparent_image_to_mask(input)