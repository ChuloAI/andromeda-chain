from utils import settings
from utils.no_buffer import print
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_hf_model(
    general_settings: settings.GeneralSettings,
    guidance_settings: settings.GuidanceSettings,
    hf_settings: settings.HuggingFaceSettings,
    tokenizer_settings: settings.TokenizerSettings,
):
    print("Loading HF model...")

    import torch
    import guidance

    # Try to load quantization library
    from transformers import BitsAndBytesConfig

    # New 4 bit quantized
    model_config = {}
    if hf_settings.load_in_4bit:
        if general_settings.base_image == "CPU":
            raise ValueError("HF 4bit quant is unsupported on CPU")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=hf_settings.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_config["revision"] = "main"
        model_config["quantization_config"] = nf4_config

    if hf_settings.load_in_8bit:
        model_config["load_in_8bit"] = True

    if hf_settings.device_map:
        model_config["device_map"] = hf_settings.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        general_settings.tokenizer_path, use_fast=tokenizer_settings.use_fast
    )
    model = AutoModelForCausalLM.from_pretrained(general_settings.model_path, **model_config)
    llama = guidance.llms.Transformers(model, tokenizer, **guidance_settings.build_args())
    return llama
