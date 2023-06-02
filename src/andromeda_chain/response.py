from typing import Dict
from dataclasses import dataclass

@dataclass
class AndromedaResponse:
    expanded_generation: str
    result_vars: Dict[str, str]
 