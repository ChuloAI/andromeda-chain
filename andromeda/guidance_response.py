from typing import Dict
from dataclasses import dataclass

@dataclass
class GuidanceResponse:
    expanded_generation: str
    result_vars: Dict[str, str]
 