from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import torch

@dataclass
class DragDiff:
    """Configuration for Multiwise/Pairwise Bayesian Optimization experiments."""

    # Experiment setup
    num_trials: int = 2

    # objective function
    mode: str = 'low-synthetic'
    obj_name: str = 'x_squared'
    json_input: str = '.json'

    # Noise / randomness
    noise: float = 0.1
    seed: int = 1234
    use_random_seeds: bool = False
    
    save_results: bool = True
    output_path: str = 'outputs'

    #ImageGen parameters
    image_models: List[str] = field(default_factory=lambda: ["sdxl","pixart","flux","sd3"])
    prompts: str = ""
    num_inference_steps: int = 50
    img_seed : int = 864
    
    obj: str = f"objectives.low_dims.ImageGen.sdxl"

    max_resolution: int = 512
    words: List[str] = field(default_factory=lambda: ["",""])
    selected_points: List[List[int]] = field(default_factory=lambda: [[1,2],[3,4]])
    COCOEE_path:str = "./promptsets/COCOEE"
    vae_path: str = "default"
    inversion_strength: float = 0.7
    unet_feature_idx: int = 3
    r_m: int = 1
    r_p: int = 3
    lam: float = 0.1
    latent_lr: float = 0.01
    n_pix_step: int = 80
    lora_steps: int = 50
    lora_str: str = "lora_tmp"
    lora_lr: float = 0.0005
    lora_batch_size: int = 4
    lora_rank: int = 16
    lora_path: str = ""
    start_step: int = 0
    start_layer: int = 10
    
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
class DragDiff_promptset(DragDiff):
    """Additional promptset-specific parameters"""

    img_random_seeds: List[int] = field(default_factory=lambda: [7876, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536, 864]) #
    promptset_path: str = 'prompt_datasets/ae_prompts5/animals.txt'
    num_samples_per_prompt: int = 10
    prompts_per_category: int = -1
    
    
# Probit, Pairwise, EUBO, qEI, rand - [2, 7883, 4420, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, rand - [2, 7876, 4415, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, rand - [5, 7876, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, taf, rand - [5, 7881, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, taf, 2s-taf, rand - [7847, 4, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]