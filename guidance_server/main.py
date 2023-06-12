import logging
from typing import Any, Dict, List

from fastapi import FastAPI
import guidance
from guidance import Program
from pydantic import BaseModel

from utils import settings, load_gptq, load_hf, load_llama_cpp
from utils.no_buffer import print

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)


print("------------------Loading config from environment------------------")
environment_variables = settings.EnvironmentVariables()
print("Environment variables: ", environment_variables)

general_settings = settings.GeneralSettings(environment_variables)
print("General settings: ", general_settings)

hugging_face_settings = settings.HuggingFaceSettings(environment_variables)
print("Hugging Face Settings: ", hugging_face_settings)

gptq_settings = settings.GPTQSettings(environment_variables)
print("GPTQ Settings: ", gptq_settings)

cpp_settings = settings.LlamaCppSettings(environment_variables)
print("CPP Settings: ", cpp_settings)

tokenizer_settings = settings.TokenizerSettings(environment_variables)
print("Tokenizer Settings: ", tokenizer_settings)

guidance_settings = settings.GuidanceSettings(environment_variables)
print("Tokenizer Settings: ", tokenizer_settings)


detected_gptq_in_path = "gptq" in general_settings.model_path.lower()
print("Loading model, this may take a while...")
print("MODEL_PATH: ", general_settings.model_path)
print("DETECTED_GPTQ_IN_PATH: ", detected_gptq_in_path)
print("--------------------------------------------------------------------")

print("--------------------------Loading model-----------------------------")
llama = None
if detected_gptq_in_path or general_settings.loading_method == "GPTQ":
    if general_settings.base_image == "CPU":
        raise ValueError(f"Unsupported Loading Method for CPU: {general_settings.loading_method}")
    llama = load_gptq.load_gptq_model(
        general_settings,
        guidance_settings,
        gptq_settings,
        tokenizer_settings,
    )
elif general_settings.loading_method == "CPP":
    llama = load_llama_cpp.load_llama_cpp(
        general_settings,
        guidance_settings,
        cpp_settings,
    )
elif general_settings.loading_method == "HUGGING_FACE":
    llama = load_hf.load_hf_model(
        general_settings,
        guidance_settings,
        hugging_face_settings,
        tokenizer_settings,
    )
else:
    raise ValueError(f"Invalid loading method: {general_settings.loading_method}")

print("--------------------------Model loaded!-----------------------------")

print("Starting server...")


class Request(BaseModel):
    input_vars: Dict[str, Any]
    output_vars: List[str]
    guidance_kwargs: Dict[str, str]
    prompt_template: str


class RawRequest(BaseModel):
    prompt: str
    max_new_tokens: int
    temperature: float
    stop: str


app = FastAPI()


@app.post("/")
def call_llama(request: Request):
    input_vars = request.input_vars
    kwargs = request.guidance_kwargs
    output_vars = request.output_vars

    guidance_program: Program = guidance(request.prompt_template)
    program_result = guidance_program(
        **kwargs,
        stream=False,
        async_mode=False,
        caching=False,
        **input_vars,
        llm=llama,
    )
    output = {"__main__": str(program_result)}
    for output_var in output_vars:
        output[output_var] = program_result[output_var]
    return output


@app.post("/raw")
def call_raw_llm(request: RawRequest):
    prompt = request.prompt + "{{"
    if request.stop:
        prompt += "gen 'output' temperature={} max_tokens={} stop='{}'".format(
            request.temperature, request.max_new_tokens, request.stop
        )

    else:
        prompt += "gen 'output' temperature={} max_tokens={}".format(
            request.temperature, request.max_new_tokens
        )

    prompt += "}}"

    guidance_program = guidance(prompt)

    program_result = guidance_program(
        stream=False,
        async_mode=False,
        caching=False,
        llm=llama,
    )
    return {"output": program_result["output"]}
