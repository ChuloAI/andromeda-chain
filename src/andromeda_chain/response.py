from dataclasses import dataclass
from typing import Dict


@dataclass
class AndromedaResponse:
    expanded_generation: str
    result_vars: Dict[str, str]
