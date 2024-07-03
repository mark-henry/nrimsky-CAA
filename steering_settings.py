from typing import Optional, Literal
import os
from dataclasses import dataclass
from behaviors import ALL_BEHAVIORS

@dataclass
class SteeringSettings:
    behavior: str = "sycophancy"
    type: Literal["open_ended", "ab", "truthful_qa", "mmlu"] = "ab"
    system_prompt: Optional[Literal["pos", "neg"]] = None
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    model_name_path: str = "meta-llama/Llama-2-7b-hf"
    use_chat: bool = False
    override_model_weights_path: Optional[str] = None

    def __post_init__(self):
        assert self.behavior in ALL_BEHAVIORS, f"Invalid behavior {self.behavior}"

    def make_result_save_suffix(
            self,
            layer: Optional[int] = None,
            multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "model": self.model_name_path,
            "use_chat": self.use_chat,
            "override_model_weights_path": self.override_model_weights_path,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])