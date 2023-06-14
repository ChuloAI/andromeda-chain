# Code stolen from Karajan (BrainChulo)
from utils import settings
import guidance


def load_llama_cpp(
    general_settings: settings.GeneralSettings,
    guidance_settings: settings.GuidanceSettings,
    cpp_settings: settings.LlamaCppSettings,
):
    print("Loading guidance model...")
    llama = guidance.llms.LlamaCpp(
        model=general_settings.model_path,  # Must point to .bin file
        tokenizer=general_settings.tokenizer_path,
        n_gpu_layers=cpp_settings.n_gpu_layers,
        n_threads=cpp_settings.n_threads,
        caching=cpp_settings.caching,
        **guidance_settings.build_args()
    )
    guidance.llm = llama 
    return guidance.llm
