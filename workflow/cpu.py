import fal

from workflow.image.edge.teed import TeedInput, TeedOutput, run_teed

dummy_image_url = "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg"

requirements = [
    "Pillow==10.3.0",
    "torch==2.2.2",
    "numpy==1.26.4",
    "opencv-python==4.9.0.80",
    "kornia==0.7.2",
    "scikit-image==0.23.1",
    "scikit-learn==1.4.2",
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


if __name__ == "__main__":
    app = fal.wrap_app(CpuWorkflowUtils)
    app()
