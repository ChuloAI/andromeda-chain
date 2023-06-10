from utils import settings
from utils.no_buffer import print

def load_hf_model(model_path: str, general_settings: settings.GeneralSettings, hf_settings: settings.HuggingFaceSettings):
    print("Loading HF model...")

    import torch
    import guidance

    # Try to load quantization library
    from transformers import BitsAndBytesConfig

    # New 4 bit quantized
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=hf_settings.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    loaded_bits_and_bytes = True
    nf4_config = nf4_config

    model_config = {}

    print("LOADED_BITS_AND_BYTES: ", loaded_bits_and_bytes)
    if nf4_config:
        model_config["revision"] = "main"
        model_config["quantization_config"] = nf4_config

    model_config["device_map"] = "auto"
    llama = guidance.llms.Transformers(model_path, **model_config)
    return llama