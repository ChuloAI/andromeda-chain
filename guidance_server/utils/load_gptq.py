import os

import guidance
import torch
from transformers import AutoTokenizer

from utils import settings
from utils.no_buffer import print


def load_gptq_model(
    general_settings: settings.GeneralSettings,
    guidance_settings: settings.GuidanceSettings,
    gptq_settings: settings.GPTQSettings,
    tokenizer_settings: settings.TokenizerSettings,
):
    print("Loading GPTQ model...")
    files = os.listdir(model_path)
    for file in files:
        if file.endswith(".safetensors") or file.endswith(".pt"):
            checkpoint = os.path.join(model_path, file)

    if gptq_settings.cpu_offloading:
        print(
            "Using CPU Offloading option. Please note that the env var 'GPTQ_DEVICE' is ignored with this option."
        )
        from gptq_for_llama.llama_inference_offload import (
            load_quant as load_quant_with_offload,
        )

        model = load_quant_with_offload(
            general_settings.model_path,
            checkpoint,
            wbits=gptq_settings.wbits,
            groupsize=gptq_settings.group_size,
            pre_layer=gptq_settings.pre_loaded_layers,
        )

    else:
        print(
            "Skipping CPU Offloading option. Please try enabling it you run into out of memory issues."
        )
        from gptq_for_llama.llama_inference import load_quant

        model = load_quant(
            model_path,
            checkpoint,
            wbits=gptq_settings.wbits,
            groupsize=gptq_settings.group_size,
        )
        model.to(gptq_settings.device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=tokenizer_settings.use_fast
    )
    llama = guidance.llms.Transformers(model, tokenizer, **guidance_settings.build_args())
    return llama
