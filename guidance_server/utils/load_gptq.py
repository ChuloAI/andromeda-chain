import os

import guidance
import torch
from transformers import AutoTokenizer

from utils import settings
from utils.no_buffer import print

def load_gptq_model(model_path: str, general_settings: settings.GeneralSettings, gptq_settings: settings.GPTQSettings):
    print("Loading GPTQ model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    files = os.listdir(model_path)
    for file in files:
        if file.endswith(".safetensors") or file.endswith(".pt"):
            checkpoint = os.path.join(model_path, file)

    if general_settings.cpu_offloading:
        from gptq_for_llama.llama_inference_offload import (
            load_quant as load_quant_with_offload,
        )
        print("Using CPU Offloading option. Please note that the env var 'GPTQ_DEVICE' is ignored with this option.")
        model = load_quant_with_offload(model_path, checkpoint, wbits=gptq_settings.wbits, groupsize=gptq_settings.group_size)

    else:
        from gptq_for_llama.llama_inference import load_quant
        model = load_quant(model_path, checkpoint, wbits=gptq_settings.wbits, groupsize=gptq_settings.group_size)
        model.to(gptq_settings.device)

    llama = guidance.llms.Transformers(model, tokenizer)
    return llama
