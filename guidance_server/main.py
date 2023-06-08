import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI
import guidance
from guidance import Program
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

gptq_is_available = False
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

# Try to load GPTQ-For-LLaMA
from gptq_for_llama.llama_inference import load_quant
gptq_is_available = True


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
    model_path = os.environ["MODEL_PATH"]
except KeyError:
    raise KeyError(
        "You must set the 'MODEL_PATH' environment variable where the model to be loaded can be found."
    )

print("Loading model, this may take a while...")


model_config = {}

llama = None

if gptq_is_available and ("gptq" in model_path.lower() or os.getenv("USE_GPTQ")):
    print("Loading GPTQ model...")
    wbits = int(os.getenv("GPTQ_WBITS", 4))
    group_size = int(os.getenv("GROUP_SIZE", 128))
    gptq_device = os.getenv("GPTQ_DEVICE", "cuda")
    print("WBITS: ", wbits)
    print("GroupSize: ", group_size)
    print("GPTQ Device: ", gptq_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    files = os.listdir(model_path)
    for file in files:
        if "safetensors" in file:
            checkpoint = os.path.join(model_path, file)

    model = load_quant(model_path, checkpoint, wbits=wbits, groupsize=group_size)
    model.to(gptq_device)
    llama = guidance.llms.Transformers(model, tokenizer, **model_config)

else:
    print("Loading HF model...")
    if nf4_config:
        model_config["revision"] = "main"
        model_config["quantization_config"] = nf4_config
    llama = guidance.llms.Transformers(model_path, **model_config)

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
