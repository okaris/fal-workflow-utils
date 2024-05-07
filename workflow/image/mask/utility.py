import fal
import PIL
from fal.toolkit import Image
from pydantic import BaseModel, Field

from workflow.common import read_image_from_url


class MaskInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/workflow_utils/mask_input.png",
        ],
    )


class InvertMaskOutput(BaseModel):
    image: Image = Field(
        description="The mask",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/invert_mask_output.png",
                content_type="image/png",
                width=610,
                height=700,
            )
        ],
    )


def invert_mask(
    input: MaskInput,
) -> InvertMaskOutput:
    import numpy as np
    import PIL

    image = read_image_from_url(input.image_url)
    mask = np.array(image)
    inverted_mask = 255 - mask
    inverted_mask = PIL.Image.fromarray(inverted_mask)
    inverted_mask = Image.from_pil(inverted_mask)
    return InvertMaskOutput(image=inverted_mask)


@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_invert_mask_on_fal(
    input: MaskInput,
) -> InvertMaskOutput:
    return invert_mask(input)


class BlurMaskInput(MaskInput):
    radius: int = Field(
        description="The radius of the Gaussian blur.",
        examples=[5],
        default=5,
    )


class BlurMaskOutput(BaseModel):
    image: Image = Field(
        description="The mask",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/blur_mask_output.png",
                content_type="image/png",
                width=610,
                height=700,
            )
        ],
    )


def blur_mask(
    input: BlurMaskInput,
) -> BlurMaskOutput:
    from PIL import ImageFilter

    image = read_image_from_url(input.image_url)
    blurred_mask = image.filter(ImageFilter.GaussianBlur(input.radius))
    blurred_mask = Image.from_pil(blurred_mask)
    return BlurMaskOutput(image=blurred_mask)


@fal.function(
    requirements=[
        "Pillow==10.3.0",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_blur_mask_on_fal(
    input: BlurMaskInput,
) -> BlurMaskOutput:
    return blur_mask(input)


class GrowMaskInput(MaskInput):
    pixels: int = Field(
        description="The number of pixels to grow the mask.",
        examples=[5],
        default=5,
    )
    threshold: int = Field(
        description="The threshold to convert the image to a mask. 0-255.",
        examples=[128],
        default=128,
    )


class GrowMaskOutput(BaseModel):
    image: Image = Field(
        description="The mask",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/grow_output.png",
                content_type="image/png",
                width=610,
                height=700,
            )
        ],
    )


def grow_mask(
    input: GrowMaskInput,
) -> GrowMaskOutput:
    import cv2
    import numpy as np
    import PIL

    image = read_image_from_url(input.image_url)
    mask = np.array(image)
    grown_mask = np.where(mask > input.threshold / 255, 1, 0).astype(np.uint8)
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
    return GrowMaskOutput(image=fal_image)


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
) -> GrowMaskOutput:
    return grow_mask(input)


class ShrinkMaskInput(MaskInput):
    pixels: int = Field(
        description="The number of pixels to shrink the mask.",
        examples=[5],
        default=5,
    )
    threshold: int = Field(
        description="The threshold to convert the image to a mask. 0-255.",
        examples=[128],
        default=128,
    )


class ShrinkMaskOutput(BaseModel):
    image: Image = Field(
        description="The mask",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/shrink_mask_output.png",
                content_type="image/png",
                width=610,
                height=700,
            )
        ],
    )


def shrink_mask(
    input: ShrinkMaskInput,
) -> ShrinkMaskOutput:
    import cv2
    import numpy as np
    import PIL

    image = read_image_from_url(input.image_url)
    mask = np.array(image)
    shrunk_mask = np.where(mask > input.threshold / 255, 1, 0).astype(np.uint8)
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
    return ShrinkMaskOutput(image=fal_image)


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
) -> ShrinkMaskOutput:
    return shrink_mask(input)


class TransparentImageToMaskInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/workflow_utils/transparent_image_to_mask_input.png",
        ],
    )
    threshold: int = Field(
        description="The threshold to convert the image to a mask.",
        examples=[128],
        default=128,
    )


class TransparentImageToMaskOutput(BaseModel):
    image: Image = Field(
        description="The mask",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/transparent_image_to_mask_output.png",
                content_type="image/png",
                width=610,
                height=700,
            )
        ],
    )


def transparent_image_to_mask(
    input: TransparentImageToMaskInput,
) -> TransparentImageToMaskOutput:
    import numpy as np
    import PIL

    image = read_image_from_url(input.image_url, convert_to_rgb=False)
    if image.mode != "RGBA":
        image = image.convert("RGBA")

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
    return TransparentImageToMaskOutput(image=fal_image)


@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "numpy==1.26.4",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_transparent_image_to_mask_on_fal(
    input: TransparentImageToMaskInput,
) -> TransparentImageToMaskOutput:
    return transparent_image_to_mask(input)
