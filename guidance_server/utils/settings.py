import os
from utils.no_buffer import print


class EnvironmentVariables:
    def __init__(self) -> None:
        # General Settings
        self.general_loading_method = os.getenv("GENERAL_LOADING_METHOD")
        self.general_bool_cpu_offloading = os.getenv("GENERAL_BOOL_CPU_OFFLOADING")

        # HuggingFace
        self.hf_bool_use_quant = os.getenv("HF_BOOL_USE_QUANT")
        self.hf_bool_load_in_4bit = os.getenv("HF_BOOL_USE_4_BIT")

        # GPTQ
        self.gptq_int_wbits = os.getenv("GPTQ_INT_WBITS", "4")
        self.gptq_int_group_size = os.getenv("GPTQ_INT_GROUP_SIZE", "128")
        self.gptq_device = os.getenv("GPTQ_DEVICE", "cuda")

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
                    else:
                        resolved_env_var_name = scoped_env_var_name
                        resolved_env_value = raw_env_value
                print(f"Var '{key}:{raw_env_value}' resolved to '{resolved_env_var_name}:{resolved_env_value}'")
                setattr(self, resolved_env_var_name, resolved_env_value)

    def __repr__(self) -> str:
        return str(self.__dict__)

class GeneralSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "general")
        supported_list = ["HUGGING_FACE", "GPTQ"]
        if self.loading_method not in supported_list:
            raise ValueError(f"Loading method {self.loading_method} not in supported list: {supported_list}")


class HuggingFaceSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "hf")

class GPTQSettings(BaseSettings):
    def __init__(self, env: EnvironmentVariables) -> None:
        super().__init__(env, "gptq")
