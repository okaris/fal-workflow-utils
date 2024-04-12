import fal
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
            "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
        ],
    )


class TeedOutput(BaseModel):
    image: Image = Field(description="The edge map.")


def run_teed(
    input: TeedInput,
) -> TeedOutput:
    clone_repository(
        "https://github.com/xavysp/TEED.git",
        commit_hash="5e074acb7a0d39f81356b3d08801e2bb7526e850",
        repo_name=TEED_REPO_NAME,
    )

    import sys

    if str(TEED_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(TEED_REPO_PATH))

    import numpy as np
    import PIL
    import torch
    from ted import TED
    from torchvision.transforms import ToPILImage

    # cpu is faster
    device = torch.device("cpu")
    model = TED().to(device)
    checkpoint_path = download_model_weights(CHECKPOINT)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    with torch.no_grad():
        image = read_image_from_url(input.image_url)
        image = image.convert("RGB")

        # resize to the nearest multiple of 8
        width, height = image.size
        print("original image shape", width, height)
        new_width = width - (width % 8)
        new_height = height - (height % 8)
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

        # convert to pytorch tensor
        image = (
            (torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0)
            .unsqueeze(0)
            .to(device)
        )
        print("image shape", image.shape)

        preds = model(image)
        output_tensor = preds[-1]
        print("output_tensor shape", output_tensor.shape)

        assert len(output_tensor.shape) == 4, output_tensor.shape

        image_vis = torch.sigmoid(output_tensor)
        image_vis = 1.0 - image_vis
        detected_map = ToPILImage()(image_vis[0] * 255.0)

        detected_map = detected_map.resize((width, height), PIL.Image.LANCZOS).convert(
            "RGB"
        )

        print("detected_map shape", detected_map.size)

        # resize to original size

    result = TeedOutput(image=Image.from_pil(detected_map))

    print("result", result)

    return result


@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "torch==2.2.2",
        "torchvision",
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
