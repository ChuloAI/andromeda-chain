import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI
import guidance
from guidance import Program
from pydantic import BaseModel
import torch


logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

nf4_config = None

# Try to load quantization library
try:
    from transformers import BitsAndBytesConfig

    # New 4 bit quantized
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

except ImportError:
    pass


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


try:
    model = os.environ["MODEL_PATH"]
except KeyError:
    raise KeyError(
        "You must set the 'MODEL_PATH' environment variable where the model to be loaded can be found."
    )

print("Loading model, this may take a while...")


model_config = {"revision": "main"}
if nf4_config:
    model_config["quantization_config"] = nf4_config

llama = guidance.llms.Transformers(model, **model_config)

print("Server loaded!")


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
