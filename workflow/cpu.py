import fal

from workflow.image.edge.teed import TeedInput, TeedOutput, run_teed
from workflow.image.insightface import (
    InsightfaceInput,
    InsightfaceOutput,
    load_insightface,
    run_insightface_with_pipeline,
)
from workflow.image.mask.utility import (
    BlurMaskInput,
    BlurMaskOutput,
    GrowMaskInput,
    GrowMaskOutput,
    InvertMaskOutput,
    MaskInput,
    ShrinkMaskInput,
    ShrinkMaskOutput,
    TransparentImageToMaskInput,
    TransparentImageToMaskOutput,
    blur_mask,
    grow_mask,
    invert_mask,
    shrink_mask,
    transparent_image_to_mask,
)
from workflow.image.utils import ImageOutput, ResizeImageInput, resize_image

dummy_image_url = "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg"

requirements = [
    "Pillow==10.3.0",
    "torch==2.2.2",
    "numpy==1.26.4",
    "opencv-python==4.9.0.80",
    "kornia==0.7.2",
    "scikit-image==0.23.1",
    "scikit-learn==1.4.2",
    "pydantic==1.10.15",
    "git+https://github.com/badayvedat/insightface.git@1ffa3405eedcfe4193c3113affcbfc294d0e684f#subdirectory=python-package",
    "opencv-python",
    "onnxruntime",
    "safetensors",
]


class CpuWorkflowUtils(
    fal.App,  # type: ignore
    _scheduler="nomad",
    max_concurrency=4,
    min_concurrency=1,
    keep_alive=300,
):
    requirements = requirements
    machine_type = "L"

    def setup(self):
        # load models
        _ = run_teed(TeedInput(image_url=dummy_image_url))
        self.insightface_model_url = "buffalo_l"
        self.insightface_model = load_insightface(self.insightface_model_url)

    @fal.endpoint("/teed")
    async def teed(self, input: TeedInput) -> TeedOutput:
        return run_teed(input)

    @fal.endpoint("/invert_mask")
    async def invert_mask(self, input: MaskInput) -> InvertMaskOutput:
        return invert_mask(input)

    @fal.endpoint("/blur_mask")
    async def blur_mask(self, input: BlurMaskInput) -> BlurMaskOutput:
        return blur_mask(input)

    @fal.endpoint("/grow_mask")
    async def grow_mask(self, input: GrowMaskInput) -> GrowMaskOutput:
        return grow_mask(input)

    @fal.endpoint("/shrink_mask")
    async def shrink_mask(self, input: ShrinkMaskInput) -> ShrinkMaskOutput:
        return shrink_mask(input)

    @fal.endpoint("/transparent_image_to_mask")
    async def transparent_image_to_mask(
        self, input: TransparentImageToMaskInput
    ) -> TransparentImageToMaskOutput:
        return transparent_image_to_mask(input)

    @fal.endpoint("/insightface")
    async def insightface(self, input: InsightfaceInput) -> InsightfaceOutput:
        if self.insightface_model_url != input.model_url:
            self.insightface_model_url = input.model_url
            self.insightface_model = load_insightface(self.insightface_model_url)
        return run_insightface_with_pipeline(input, self.insightface_model)

    @fal.endpoint("/resize-image")
    async def resize_image(self, input: ResizeImageInput) -> ImageOutput:
        return resize_image(input)


@fal.function(
    machine_type="L",
    _scheduler="nomad",
    serve=True,
)
def test():
    print("hello fal")


if __name__ == "__main__":
    app = fal.wrap_app(CpuWorkflowUtils)
    app()
