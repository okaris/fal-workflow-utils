from pathlib import Path

import fal
import PIL.Image
from fal.toolkit import FAL_REPOSITORY_DIR, File, Image, download_model_weights
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fastapi import HTTPException
from pydantic import BaseModel, Field

from workflow.common import read_image_from_url

INSIGHTFACE_REPO_NAME = "insightface"
INSIGHTFACE_REPO_PATH = FAL_REPOSITORY_DIR / INSIGHTFACE_REPO_NAME


INSIGHTFACE_MODEL_CACHE_PATH = Path("/data/models")


class InsightfaceInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/retoucher/GGsAolHXsAA58vn.jpeg",
        ],
    )
    threshold: float = Field(
        description="Threshold for the edge map.",
        default=0.5,
        examples=[0.5],
    )
    det_size_width: int = Field(
        description="Size of the detection.",
        default=640,
        examples=[640],
    )
    det_size_height: int = Field(
        description="Size of the detection.",
        default=640,
        examples=[640],
    )
    max_face_num: int = Field(
        description="Maximum number of faces to detect.",
        default=1,
        examples=[1],
    )
    model_url: str = Field(
        description="URL of the model weights.",
        examples=["buffalo_l", "antelopev2"],
        default="buffalo_l",
    )
    sorting: str = Field(
        description="Sorting of the faces.",
        examples=[
            "largest-to-smallest",
            "smallest-to-largest",
            "left-to-right",
            "right-to-left",
        ],
        default="size",
    )
    sync_mode: bool = Field(
        description="Whether to run in sync mode.",
        default=True,
    )


class FaceDetection(BaseModel):
    bbox: list[int] = Field(
        description="Bounding box of the face.",
        examples=[[0, 0, 100, 100]],
    )
    kps: list[list[int]] | None = Field(
        description="Keypoints of the face.",
        examples=[],
    )
    kps_image: Image = Field(
        description="Keypoints of the face on the image.",
        examples=[
            # Image(
            #     url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/invert_mask_output.png",
            #     content_type="image/png",
            #     width=610,
            #     height=700,
            # )
        ],
    )
    det_score: float = Field(
        description="Confidence score of the detection.",
        examples=[0.9],
    )
    embedding_file: File = Field(
        description="Embedding of the face.",
        examples="",
    )
    sex: str | None = Field(
        description="Either M or F if available.",
        examples=["M"],
    )


class InsightfaceOutput(BaseModel):
    faces: list[FaceDetection] = Field(
        description="faces detected sorted by size",
    )
    bbox: list[float] = Field(
        description="Bounding box of the face.",
        examples=[[0, 0, 100, 100]],
    )
    kps: list[list[float]] | None = Field(
        description="Keypoints of the face.",
        examples=[],
    )
    kps_image: Image = Field(
        description="Keypoints of the face on the image.",
        examples=[
            # Image(
            #     url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/invert_mask_output.png",
            #     content_type="image/png",
            #     width=610,
            #     height=700,
            # )
        ],
    )
    det_score: float = Field(
        description="Confidence score of the detection.",
        examples=[0.9],
    )
    embedding_file: File = Field(
        description="Embedding of the face.",
        examples="",
    )
    sex: str | None = Field(
        description="Either M or F if available.",
        examples=["M"],
    )


def patch_onnx_runtime(
    inter_op_num_threads: int = 16,
    intra_op_num_threads: int = 16,
    omp_num_threads: int = 16,
):
    import os

    import onnxruntime as ort

    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = inter_op_num_threads
        _default_session_options.intra_op_num_threads = intra_op_num_threads
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new


@fal.cached
def load_insightface(model_url: str):
    import insightface.utils.storage

    insightface.utils.storage.BASE_REPO_URL = (
        "https://storage.googleapis.com/fal-models/insightface"
    )

    import os

    from insightface.app import FaceAnalysis

    os.environ["HF_HOME"] = "/data/models"

    patch_onnx_runtime()

    insightface_model_name = None
    try:

        # check if it is a url
        if model_url.startswith("https://"):
            print("Assuming insightface model path is a URL")
            insightface_path = download_model_weights(
                model_url.insight_face_model_path,
            )
            insightface_model_dir = Path(insightface_path).parent
            insightface_model_name = Path(insightface_path).name
        elif model_url.startswith("http://"):
            # raise an error if the path is an http link
            raise HTTPException(
                422,
                detail="HTTP links are not supported for insightface model weights. Please use HTTPS links or local paths.",
            )
        # see if there is a single forward slash in the path
        elif model_url.count("/") == 1:
            raise HTTPException(
                422,
                detail="Huggingface models for insightface are not supported!",
            )
        else:
            # assume it is a model name
            insightface_model_name = model_url
            insightface_model_dir = None

    except Exception as e:
        raise HTTPException(
            422,
            detail=f"Failed to download insightface model: {e}",
        )

    if insightface_model_dir is None:
        app = FaceAnalysis(
            name=insightface_model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        app = FaceAnalysis(
            name=insightface_model_name,
            root=insightface_model_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    return app


def draw_kps(
    image_pil,
    kps,
    color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)],
):
    import math

    import cv2
    import numpy as np
    import PIL.Image

    stickwidth = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    if isinstance(image_pil, PIL.Image.Image):
        out_img = np.array(image_pil)
    else:
        out_img = image_pil

    for i, _ in enumerate(limb_seq):
        index = limb_seq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def checksum(
    obj,
    method="md5",
    *,
    # If dicts with different key order are to be treated as identical.
    # Set False otherwise.
    sort_keys=True,
    # A dict with circular referencing objects can never be hashed and is
    # out of scope of this topic. Skip checking such cases to save an
    # in-memory mapping as we don't expect them. Set True otherwise.
    check_circular=False,
    # Set True to output bytes instead of hex string to save memory, if the
    # checksum is used only for internal comparison and not to be output.
    ouput_bytes=False,
):
    import hashlib
    import json

    m = hashlib.new(method)
    encoder = json.JSONEncoder(
        check_circular=check_circular,
        sort_keys=sort_keys,
        ensure_ascii=False,  # don't escape Unicode chars to save bytes
        separators=(",", ":"),  # reduce default spaces to be more compact
    )
    for chunk in encoder.iterencode(obj):
        m.update(chunk.encode("UTF-8"))

    if ouput_bytes:
        return m.digest()

    return m.hexdigest()


def run_insightface_with_pipeline(
    input: InsightfaceInput,
    model,
) -> InsightfaceOutput:
    import os

    import cv2
    import numpy as np
    import torch
    from safetensors.torch import save_file

    output_dir = "/data/face_embeddings/"
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    remote_repository = None
    if os.getenv("GCLOUD_SA_JSON"):
        remote_repository = GoogleStorageRepository(
            url_expiration=2 * 24 * 60,  # 2 days, same as fal,
            bucket_name=os.getenv("GCS_BUCKET_NAME", "fal_file_storage"),
            folder="face_embeddings",
        )

    # cpu is faster in my tests
    with torch.no_grad():
        model.prepare(ctx_id=0, det_size=(input.det_size_width, input.det_size_height))

        image = read_image_from_url(input.image_url)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        faces = model.get(image)

        if len(faces) == 0:
            raise HTTPException(
                422,
                detail="No faces detected",
            )

        face_detections = []
        zero_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for face in faces:
            kps_image = draw_kps(zero_image, face.kps)
            embedding_hash = checksum(face.embedding.tolist())

            file_path = output_dir + embedding_hash + ".safetensors"
            embedding_dict = {
                "embedding": torch.from_numpy(face.embedding),
            }

            save_file(embedding_dict, file_path)
            if remote_repository:
                embedding_file = File.from_path(
                    path=file_path, repository=remote_repository
                )
            else:
                embedding_file = File.from_path(path=file_path)

            face_detections.append(
                FaceDetection(
                    bbox=face.bbox.tolist(),
                    kps=face.kps.tolist(),
                    kps_image=Image.from_pil(kps_image),
                    det_score=face.det_score,
                    embedding_file=embedding_file,
                    sex=face.sex,
                )
            )

        if input.sorting == "largest-to-smallest":
            face_detections = sorted(
                face_detections, key=lambda x: -x["bbox"][2] * x["bbox"][3]
            )
        elif input.sorting == "smallest-to-largest":
            face_detections = sorted(
                face_detections, key=lambda x: x["bbox"][2] * x["bbox"][3]
            )
        elif input.sorting == "left-to-right":
            face_detections = sorted(face_detections, key=lambda x: x["bbox"][0])
        elif input.sorting == "right-to-left":
            face_detections = sorted(face_detections, key=lambda x: -x["bbox"][0])

        top_face = face_detections[0]
        face_detections = face_detections[: input.max_face_num]

    output = InsightfaceOutput(
        faces=face_detections,
        bbox=top_face.bbox,
        kps=top_face.kps,
        kps_image=top_face.kps_image,
        det_score=top_face.det_score,
        embedding_file=top_face.embedding_file,
    )

    return output


def run_insightface(
    input: InsightfaceInput,
) -> InsightfaceOutput:
    model = load_insightface(input.model_url)
    return run_insightface_with_pipeline(input, model)


@fal.function(
    requirements=[
        "Pillow==10.3.0",
        "torch==2.2.2",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "git+https://github.com/badayvedat/insightface.git@1ffa3405eedcfe4193c3113affcbfc294d0e684f#subdirectory=python-package",
        "opencv-python",
        "onnxruntime",
        "safetensors",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_insightface_on_fal(
    input: InsightfaceInput,
) -> InsightfaceOutput:
    return run_insightface(input)


if __name__ == "__main__":
    local = run_insightface_on_fal.on(serve=False)
    # result = local(
    #     InsightfaceInput(
    #         image_url="https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
    #         model_url="buffalo_l"
    #     )
    # )
    result = local(
        InsightfaceInput(
            image_url="https://storage.googleapis.com/falserverless/model_tests/omni_zero/identity.jpg",
            model_url="buffalo_l",
        )
    )
    # result = local(
    #     InsightfaceInput(
    #         image_url="https://storage.googleapis.com/falserverless/model_tests/omni_zero/identity.jpg",
    #         model_url="antelopev2"
    #     )
    # )
    print("Result:")
    print(result)
