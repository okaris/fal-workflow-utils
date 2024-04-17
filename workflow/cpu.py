import fal

from workflow.image.edge.teed import TeedInput, TeedOutput, run_teed
from workflow.image.mask.utility import MaskInput, MaskOutput, BlurMaskInput, GrowMaskInput, ShrinkMaskInput, TransparentImageToMaskInput, invert_mask, blur_mask, grow_mask, shrink_mask, transparent_image_to_mask
dummy_image_url = "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg"

requirements = [
    "Pillow==10.3.0",
    "torch==2.2.2",
    "numpy==1.26.4",
    "opencv-python==4.9.0.80",
    "kornia==0.7.2",
    "scikit-image==0.23.1",
    "scikit-learn==1.4.2",
    "pydantic==1.10.12",
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

    @fal.endpoint("/teed")
    async def teed(self, input: TeedInput) -> TeedOutput:
        return run_teed(input)
    
    @fal.endpoint("/invert_mask")
    async def invert_mask(self, input: MaskInput) -> MaskOutput:
        return invert_mask(input)
    
    @fal.endpoint("/blur_mask")
    async def blur_mask(self, input: BlurMaskInput) -> MaskOutput:
        return blur_mask(input)
    
    @fal.endpoint("/grow_mask")
    async def grow_mask(self, input: GrowMaskInput) -> MaskOutput:
        return grow_mask(input)
    
    @fal.endpoint("/shrink_mask")
    async def shrink_mask(self, input: ShrinkMaskInput) -> MaskOutput:
        return shrink_mask(input)
    
    @fal.endpoint("/transparent_image_to_mask")
    async def transparent_image_to_mask(self, input: TransparentImageToMaskInput) -> MaskOutput:
        return transparent_image_to_mask(input)

@fal.function(machine_type="L", 
            _scheduler="nomad",
            serve=True,
)
def test():
    print("heello")

if __name__ == "__main__":
    app = fal.wrap_app(CpuWorkflowUtils)
    app()
