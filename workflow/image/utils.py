import fal
import PIL
from fal.toolkit import File, Image
from pydantic import BaseModel, Field

from workflow.common import read_image_from_url


class ImageInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/workflow_utils/mask_input.png",
        ],
    )


class ImageOutput(BaseModel):
    image: Image = Field(
        description="The output image",
        examples=[
            Image(
                url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/invert_mask_output.png",
                content_type="image/png",
                width=610,
                height=700,
            )
        ],
    )


class ResizeImageInput(ImageInput):
    width: int = Field(
        description="Width of the resized image",
        examples=[610],
    )
    height: int = Field(
        description="Height of the resized image",
        examples=[700],
    )
    mode: str = Field(
        description="Resizing mode",
        examples=["crop", "pad", "scale"],
    )
    resampling: str = Field(
        description="Resizing strategy. Only used when mode is 'scale', default is nearest",
        examples=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    scaling_proportions: str = Field(
        description="Proportions of the image. Only used when mode is 'scale', default is fit",
        examples=["fit", "fill", "stretch"],
        default="fit",
    )
    cropping_position: str = Field(
        description="Position of cropping. Only used when mode is 'crop', default is center",
        examples=["center", "top_left", "top_right", "bottom_left", "bottom_right"],
        default="center",
    )
    padding_color: str = Field(
        description="Color of padding. Only used when mode is 'pad', default is black",
        examples=["black", "white", "red", "green", "blue"],
        default="black",
    )


def resize_image(
    input: ResizeImageInput,
) -> ImageOutput:
    import PIL

    image = read_image_from_url(input.image_url, convert_to_rgb=False)
    width, height = image.size

    if input.mode == "crop":
        if input.cropping_position == "center":
            left = (width - input.width) // 2
            top = (height - input.height) // 2
            right = (width + input.width) // 2
            bottom = (height + input.height) // 2
        elif input.cropping_position == "top_left":
            left = 0
            top = 0
            right = input.width
            bottom = input.height
        elif input.cropping_position == "top_right":
            left = width - input.width
            top = 0
            right = width
            bottom = input.height
        elif input.cropping_position == "bottom_left":
            left = 0
            top = height - input.height
            right = input.width
            bottom = height
        elif input.cropping_position == "bottom_right":
            left = width - input.width
            top = height - input.height
            right = width
            bottom = height
        image = image.crop((left, top, right, bottom))

    elif input.mode == "pad":
        if input.padding_color == "black":
            padding_color = (0, 0, 0)
        elif input.padding_color == "white":
            padding_color = (255, 255, 255)
        elif input.padding_color == "red":
            padding_color = (255, 0, 0)
        elif input.padding_color == "green":
            padding_color = (0, 255, 0)
        elif input.padding_color == "blue":
            padding_color = (0, 0, 255)
        padding = PIL.Image.new("RGB", (input.width, input.height), padding_color)
        padding.paste(image, ((input.width - width) // 2, (input.height - height) // 2))
        image = padding

    elif input.mode == "scale":
        algo = PIL.Image.Resampling.NEAREST
        if input.resampling == "bilinear":
            algo = PIL.Image.Resampling.BILINEAR
        elif input.resampling == "bicubic":
            algo = PIL.Image.Resampling.BICUBIC
        elif input.resampling == "lanczos":
            algo = PIL.Image.Resampling.LANCZOS

        if input.scaling_proportions == "fit":
            image.thumbnail((input.width, input.height), algo)
        elif input.scaling_proportions == "fill":
            target_ratio = input.width / input.height
            original_ratio = image.width / image.height

            if target_ratio > original_ratio:
                new_width = input.width
                new_height = int(new_width / original_ratio)
            else:
                new_height = input.height
                new_width = int(new_height * original_ratio)

            image = image.resize((new_width, new_height), algo)

            left = (new_width - input.width) // 2
            top = (new_height - input.height) // 2
            right = left + input.width
            bottom = top + input.height

            image = image.crop((left, top, right, bottom))
        elif input.scaling_proportions == "stretch":
            image = image.resize((input.width, input.height), algo)

    fal_image = Image.from_pil(image)
    return ImageOutput(image=fal_image)


@fal.function(
    requirements=[
        "Pillow==10.3.0",
    ],
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_resize_image_on_fal(
    input: ResizeImageInput,
) -> ImageOutput:
    return resize_image(input)


@fal.function(
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def create_markdown_file(
    input: str,
) -> File:
    with open("resize_image_test_results.md", "w") as f:
        f.write(input)
    file = File.from_path("resize_image_test_results.md")
    return file


if __name__ == "__main__":
    local = run_resize_image_on_fal.on(serve=False)
    # test input size - 610x700

    # Test all the functions
    print("Testing the functions")

    mode_list = ["crop", "pad", "scale"]
    resampling_list = ["nearest", "bilinear", "bicubic", "lanczos"]
    scaling_proportions_list = ["fit", "fill", "stretch"]
    cropping_position_list = [
        "center",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]
    padding_color_list = ["black", "white", "red", "green", "blue"]
    target_sizes = [(610, 700), (500, 500), (700, 610), (1000, 1000)]
    output_dict = {}
    for target_size in target_sizes:
        print("Testing target size: ", target_size)

        # crop
        for cropping_position in cropping_position_list:
            print("Testing cropping position: ", cropping_position)
            test_input = ResizeImageInput(
                image_url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/mask_input.png",
                width=target_size[0],
                height=target_size[1],
                mode="crop",
                cropping_position=cropping_position,
            )
            output = local(test_input)
            output_dict[
                "crop_"
                + cropping_position
                + "_"
                + str(target_size[0])
                + "x"
                + str(target_size[1])
            ] = output.image.url

        # pad
        for padding_color in padding_color_list:
            print("Testing padding color: ", padding_color)
            test_input = ResizeImageInput(
                image_url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/mask_input.png",
                width=target_size[0],
                height=target_size[1],
                mode="pad",
                padding_color=padding_color,
            )
            output = local(test_input)
            output_dict[
                "pad_"
                + padding_color
                + "_"
                + str(target_size[0])
                + "x"
                + str(target_size[1])
            ] = output.image.url

        # scale
        for resampling in resampling_list:
            print("Testing resampling: ", resampling)
            for scaling_proportions in scaling_proportions_list:
                print("Testing scaling proportions: ", scaling_proportions)
                test_input = ResizeImageInput(
                    image_url="https://storage.googleapis.com/falserverless/model_tests/workflow_utils/mask_input.png",
                    width=target_size[0],
                    height=target_size[1],
                    mode="scale",
                    resampling=resampling,
                    scaling_proportions=scaling_proportions,
                )
                output = local(test_input)
                output_dict[
                    "scale_"
                    + resampling
                    + "_"
                    + scaling_proportions
                    + "_"
                    + str(target_size[0])
                    + "x"
                    + str(target_size[1])
                ] = output.image.url

    # Create a markdown file with the results
    print("Creating markdown file")
    markdown_text = "# Resize Image\n\n"
    markdown_text += "## Test Results\n\n"
    markdown_text += "| Test Name | Output Image |\n"
    markdown_text += "| --- | --- |\n"
    for key, value in output_dict.items():
        markdown_text += "| " + key + " | ![Image](" + value + ") |\n"

    markdown_maker = create_markdown_file.on(serve=False)
    markdown_file = markdown_maker(markdown_text)
    print("Markdown file created: ", markdown_file.url)
