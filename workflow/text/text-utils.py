import fal
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(
        description="Input text",
        examples=["Hello, World!"],
    )


class TextOutput(BaseModel):
    text: str = Field(
        description="The output text",
        examples=["Hello, World!"],
    )


class RegexReplaceInput(BaseModel):
    text: str = Field(
        description="Input text",
        examples=["Hello, World!"],
    )
    pattern: str = Field(
        description="Pattern to replace",
        examples=["World"],
    )
    replace: str = Field(
        description="Replacement text",
        examples=["Universe"],
    )


def regex_replace(
    input: TextInput,
) -> TextOutput:
    import re

    output_text = re.sub(input.pattern, input.replace, input.text)
    return TextOutput(text=output_text)


@fal.function(
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_regex_replace_on_fal(
    input: RegexReplaceInput,
) -> TextOutput:
    return regex_replace(input)


class InsertTextInput(TextInput):
    template: str = Field(
        description="Template to insert text into",
        examples=["Hello, {}!"],
    )


def insert_text(
    input: InsertTextInput,
) -> TextOutput:
    output_text = input.template.format(*([input.text] * input.template.count("{}")))
    return TextOutput(text=output_text)


@fal.function(
    _scheduler="nomad",
    resolver="uv",
    serve=True,
    machine_type="L",
)
def run_insert_text_on_fal(
    input: InsertTextInput,
) -> TextOutput:
    return insert_text(input)


if __name__ == "__main__":
    local = run_regex_replace_on_fal.on(serve=False)
    results: dict[str, str] = {}

    # regex_replace tests
    test_cases = {
        "basic_replacement": RegexReplaceInput(
            text="Hello, World!", pattern="World", replace="Universe"
        ),
        "multiple_matches": RegexReplaceInput(
            text="cat dog cat", pattern="cat", replace="mouse"
        ),
        "case_sensitivity": RegexReplaceInput(
            text="Cat cat CAT", pattern="cat", replace="dog"
        ),
        "wildcard_character": RegexReplaceInput(
            text="a1 b2 c3", pattern=r"\d", replace="X"
        ),
        "start_anchor": RegexReplaceInput(
            text="start middle end", pattern="^start", replace="begin"
        ),
        "end_anchor": RegexReplaceInput(
            text="start middle end", pattern="end$", replace="finish"
        ),
        "special_characters": RegexReplaceInput(
            text="a+b=c", pattern=r"\+", replace="-"
        ),
        "group_replacement": RegexReplaceInput(
            text="123-456-7890",
            pattern=r"(\d{3})-(\d{3})-(\d{4})",
            replace="($1) $2-$3",
        ),
        "non_greedy_matches": RegexReplaceInput(
            text="a <b> c <d>", pattern="<.*?>", replace="[]"
        ),
        "lookahead": RegexReplaceInput(
            text="1a2b3c", pattern=r"(?<=\d)[a-z]", replace="X"
        ),
        "lookbehind": RegexReplaceInput(
            text="a1b2c3", pattern=r"[a-z](?=\d)", replace="X"
        ),
        "no_matches": RegexReplaceInput(text="abcdef", pattern="xyz", replace="123"),
        "empty_pattern": RegexReplaceInput(text="abc", pattern="", replace="X"),
        "empty_replacement": RegexReplaceInput(
            text="a1b2c3", pattern=r"\d", replace=""
        ),
        "escaped_characters": RegexReplaceInput(
            text="a\\b\\c", pattern="\\\\", replace="/"
        ),
        "unicode_characters": RegexReplaceInput(
            text="こんにちは世界", pattern="世界", replace="world"
        ),
        "multiline_strings": RegexReplaceInput(
            text="line1\nline2\nline3", pattern="^line", replace="row"
        ),
    }

    results = {}
    for test_name, test_case in test_cases.items():
        print("Running test case", test_name)
        output = local(test_case)
        results[test_name] = output.text
        print("Output:", output.text)

    # insert_text tests
    local = run_insert_text_on_fal.on(serve=False)
    test_cases = {
        "basic_insert": InsertTextInput(text="World", template="Hello, {}!"),
        "multiple_inserts": InsertTextInput(
            text="World", template="Hello, {}! How are you, {}?"
        ),
        "empty_template": InsertTextInput(text="World", template=""),
        "empty_text": InsertTextInput(text="", template="Hello, {}!"),
        "special_characters": InsertTextInput(
            text="World", template="Hello, {}! How are you, {}?"
        ),
        "unicode_characters": InsertTextInput(text="世界", template="こんにちは, {}!"),
    }

    for test_name, test_case in test_cases.items():
        print("Running test case", test_name)
        output = local(test_case)
        results[test_name] = output.text
        print("Output:", output.text)
