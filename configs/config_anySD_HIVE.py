from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import torch

@dataclass
class AnySD_HIVE:
    """Configuration for Multiwise/Pairwise Bayesian Optimization experiments."""

    # Experiment setup
    num_trials: int = 2
    mode: str = 'low-synthetic'
    obj_name: str = 'x_squared'
    json_input: str = '.json'
    
    seed: int = 1234
    use_random_seeds: bool = False
    
    
    output_path: str = 'outputs'

    #ImageGen parameters
    image_models: List[str] = field(default_factory=lambda: ["sdxl","pixart","flux","sd3"])
    prompts: str = ""
    num_inference_steps: int = 50
    img_seed : int = 864
    
    model_name: str = 'anysd' # anysd, hivec, hivew
    model_path: Dict[str,str] = field(default_factory=lambda: {'hivec':"./models/checkpoints/hive_rw_condition.ckpt", 'hivew':"./models/checkpoints/hive_rw.ckpt"})
    
    edit: str = "./" 
    edit_object: str = "./"
    edit_type: str = "./"
    image_file: str = "./"
                
    def __post_init__(self):
        """Validation and setup logic."""
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        
        os.makedirs(self.output_path, exist_ok=True)

    def update(self, **kwargs):
        """Update config parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        self.__post_init__()


@dataclass
class AnySD_HIVE_promptset(AnySD_HIVE):
    """Additional promptset-specific parameters"""

    img_random_seeds: List[int] = field(default_factory=lambda: [7876, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536, 864]) #
    promptset_path: str = 'promptset/AnyBench-test'
    num_samples_per_prompt: int = 10
    prompts_per_category: int = -1
    
    
# Probit, Pairwise, EUBO, qEI, rand - [2, 7883, 4420, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, rand - [2, 7876, 4415, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, rand - [5, 7876, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, taf, rand - [5, 7881, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, taf, 2s-taf, rand - [7847, 4, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]