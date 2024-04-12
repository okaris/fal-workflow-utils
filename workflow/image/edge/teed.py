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


class TeedInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/retoucher/GGsAolHXsAA58vn.jpeg",
        ],
    )


class TeedOutput(BaseModel):
    image: Image = Field(
        description="The edge map.",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/teed_output.png",
                content_type="image/png",
                width=1246,
                height=2048,
            )
        ],
    )


@fal.cached
def load_teed():
    clone_repository(
        "https://github.com/xavysp/TEED.git",
        commit_hash="5e074acb7a0d39f81356b3d08801e2bb7526e850",
        repo_name=TEED_REPO_NAME,
    )

    import sys

    if str(TEED_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(TEED_REPO_PATH))

    import torch
    from ted import TED

    device = torch.device("cpu")
    model = TED().to(device)
    checkpoint_path = download_model_weights(CHECKPOINT)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    return model, device


def run_teed_with_pipeline(
    input: TeedInput,
    model,
    device,
) -> TeedOutput:
    import numpy as np
    import PIL
    import torch
    from utils.img_processing import image_normalization

    # cpu is faster in my tests
    with torch.no_grad():
        image = read_image_from_url(input.image_url)
        # resize to the nearest multiple of 8
        width, height = image.size
        new_width = (width + 7) // 8 * 8
        new_height = (height + 7) // 8 * 8
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

        # convert to pytorch tensor
        image = (
            (torch.tensor(np.array(image)).permute(2, 0, 1).float())
            .unsqueeze(0)
            .to(device)
        )

        # subtract the pixel mean
        mean_rgb = [123.68, 116.779, 103.939]
        image = image - torch.tensor(mean_rgb).view(1, 3, 1, 1).to(device)

        # put in bgr order
        image = image[:, [2, 1, 0], :, :]

        # keep as 0-255 because this model is totally unhinged
        pred = model(image)[-1][0, 0]
        pred = torch.sigmoid(pred).cpu().detach().numpy()
        pred = np.uint8(image_normalization(pred))
        detected_map = PIL.Image.fromarray(pred)
        if detected_map.size != (width, height):
            detected_map = detected_map.resize(
                (width, height), PIL.Image.LANCZOS
            ).convert("RGB")

    return TeedOutput(image=Image.from_pil(detected_map))


def run_teed(
    input: TeedInput,
) -> TeedOutput:
    model, device = load_teed()
    return run_teed_with_pipeline(input, model, device)


@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "torch==2.2.2",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "kornia==0.7.2",
        "scikit-image==0.23.1",
        "scikit-learn==1.4.2",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_teed_on_fal(
    input: TeedInput,
) -> TeedOutput:
    return run_teed(input)


if __name__ == "__main__":
    local = run_teed_on_fal.on(serve=False)
    result = local(
        TeedInput(
            image_url="https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
        )
    )
    print(result)
