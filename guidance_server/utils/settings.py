import os
from utils.no_buffer import print


class EnvironmentVariables:
    def __init__(self) -> None:
        # General Settings
        self.general_loading_method = os.getenv("GENERAL_LOADING_METHOD")

        try:
            self.general_model_path = os.environ["GENERAL_MODEL_PATH"]
        except KeyError:
            error_msg = "You must set the 'MODEL_PATH' environment variable where the model to be loaded can be found."
            raise KeyError(error_msg)

        self.general_tokenizer_path = os.getenv("GENERAL_TOKENIZER_PATH", self.general_model_path)

        # Guidance Settings
        self.guidance_after_role = os.getenv("GUIDANCE_AFTER_ROLE", "|>")
        self.guidance_before_role = os.getenv("GUIDANCE_BEFORE_ROLE", "<|")

        # Tokenizer
        self.tk_bool_use_fast = os.getenv("TK_BOOL_USE_FAST")

        # HuggingFace
        self.hf_bool_load_in_4bit = os.getenv("HF_BOOL_USE_4_BIT")
        self.hf_bool_load_in_8bit = os.getenv("HF_BOOL_USE_8_BIT")
        self.hf_bool_low_cpu_usage = os.getenv("HF_BOOL_LOW_CPU_USAGE")
        self.hf_device_map = os.getenv("HF_DEVICE_MAP")

        # GPTQ
        self.gptq_int_wbits = os.getenv("GPTQ_INT_WBITS", "4")
        self.gptq_int_group_size = os.getenv("GPTQ_INT_GROUP_SIZE", "128")
        self.gptq_int_pre_loaded_layers = os.getenv("GPTQ_INT_PRE_LOADED_LAYERS", "50")
        self.gptq_device = os.getenv("GPTQ_DEVICE", "cuda")
        self.gptq_bool_cpu_offloading = os.getenv("GPTQ_BOOL_CPU_OFFLOADING")


        # LLaMA CPP
        self.cpp_int_n_threads = os.getenv("CPP_INT_N_THREADS", "12")
        self.cpp_int_n_gpu_layers = os.getenv("CPP_INT_N_GPU_LAYERS", "500")
        self.cpp_bool_caching = os.getenv("CPP_BOOL_CACHING", "false")

    def __repr__(self) -> str:
        return str(self.__dict__)


class BaseSettings:
    def __init__(self, env: EnvironmentVariables, prefix: str) -> None:
        for key in dir(env):
            # key, e.g., gptq_device
            if prefix in key:
                # value, e.g., 'cuda'
                raw_env_value = getattr(env, key)

                # Copy environment variable key, removing it's prefix
                # For instance 'gptq_device' becomes 'device
                scoped_env_var_name = key.replace(prefix, "")[1:]

                # Check types preffixes

                # Edge case, no '_' left in var name
                splits = scoped_env_var_name.split("_")
                if len(splits) <= 0:
                    resolved_env_var_name = scoped_env_var_name
                    resolved_env_value = raw_env_value

                # Convert 'int' and 'bool' types
                else:
                    maybe_type_prefix = splits[0]
                    if maybe_type_prefix == "int":
                        resolved_env_var_name = "_".join(splits[1:])
                        resolved_env_value = int(raw_env_value)
                    elif maybe_type_prefix == "bool":
                        resolved_env_var_name = "_".join(splits[1:])
                        resolved_env_value = raw_env_value == "true"
                    elif maybe_type_prefix == "float":
                        resolved_env_var_name = "_".join(splits[1:])
                        resolved_env_value = float(raw_env_value)
                    else:
                        resolved_env_var_name = scoped_env_var_name
                        resolved_env_value = raw_env_value
                print(
                    f"Var '{key}:{raw_env_value}' resolved to '{resolved_env_var_name}:{resolved_env_value}'"
                )
                setattr(self, resolved_env_var_name, resolved_env_value)

    def __repr__(self) -> str:
        return str(self.__dict__)


class GeneralSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "general")
        supported_list = ["HUGGING_FACE", "GPTQ", "CPP"]
        if self.loading_method not in supported_list:
            raise ValueError(
                f"Loading method {self.loading_method} not in supported list: {supported_list}"
            )


class HuggingFaceSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "hf")


class GPTQSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "gptq")


class TokenizerSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "tk")


class LlamaCppSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "cpp")


class GuidanceSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "guidance")


    def build_args(self):
        guidance_args = {}
        if self.after_role and self.before_role:
            guidance_args["after_role"] = self.after_role
            guidance_args["before_role"] = self.before_role
        return guidance_args