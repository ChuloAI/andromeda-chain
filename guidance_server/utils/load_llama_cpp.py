# Code stolen from Karajan (BrainChulo)
from utils import settings
import guidance


def load_llama_cpp(
    model_path: str,
    tokenizer_path: str,
    guidance_settings: settings.GuidanceSettings,
    cpp_settings: settings.LlamaCppSettings,
):
    print("Loading guidance model...")
    guidance.llms.LlamaCpp(
        model=model_path,  # Must point to .bin file
        tokenizer=tokenizer_path,
        n_gpu_layers=cpp_settings.n_gpu_layers,
        n_threads=cpp_settings.n_threads,
        caching=cpp_settings.caching,
        **guidance_settings.build_args()
    )
    return guidance.llm
