import pkgutil
import os
import torch
import yaml
from stripedhyena.model import StripedHyena
from stripedhyena.utils import dotdict
from transformers import AutoConfig, AutoModelForCausalLM

from .tokenizer import CharLevelTokenizer

MODEL_NAMES = ["evo-1-8k-base", "evo-1-131k-base"]

HF_MODEL_NAME_MAP = {
    "evo-1-8k-base": "togethercomputer/evo-1-8k-base",
    "evo-1-131k-base": "togethercomputer/evo-1-131k-base",
}


class Evo:
    def __init__(
        self,
        model_name: str = MODEL_NAMES[1],
        model_backbone_state_path: str = None,
        device: str = None,
    ):
        """
        Loads an Evo model checkpoint given a model name.
        If the checkpoint does not exist, we automatically download it from HuggingFace.
        """
        self.device = device
        self.model_backbone_state_path = model_backbone_state_path

        # Check model name.

        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Invalid model name {model_name}. Should be one of: "
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Assign config path.

        if model_name == "evo-1-8k-base":
            config_path = "configs/evo-1-8k-base_inference.yml"
        elif model_name == "evo-1-131k-base":
            config_path = "configs/evo-1-131k-base_inference.yml"
        else:
            raise ValueError(
                f"Invalid model name {model_name}. Should be one of: "
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Prepare for saving the backbone state dict
        assert (
            self.model_backbone_state_path is not None
        ), "model_backbone_state_path must be provided"
        if not os.path.exists(self.model_backbone_state_path):
            print("Saving SH model backbone state dict...")
            self.save_backbone_state(model_name, self.model_backbone_state_path)

        # Load model.

        self.model = self.load_checkpoint(config_path=config_path, device=self.device)

        # Load tokenizer.

        self.tokenizer = CharLevelTokenizer(512)

    def save_backbone_state(self, model_name, path: str):
        """
        Save SH model backbone state dict.
        """

        hf_model_name = HF_MODEL_NAME_MAP[model_name]

        model_config = AutoConfig.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            revision="1.1_fix",
        )

        model_config.use_cache = True

        # Load model.

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            config=model_config,
            trust_remote_code=True,
            revision="1.1_fix",
        )

        # Load model state dict & cleanup.
        torch.save(model.backbone.state_dict(), path)

    def load_checkpoint(
        self,
        config_path: str = "evo/configs/evo-1-131k-base_inference.yml",
        device: str = None,
        *args,
        **kwargs,
    ):
        """
        Load checkpoint from HuggingFace and place it into SH model.
        """

        state_dict = torch.load(self.model_backbone_state_path)

        # Load SH config.

        config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
        global_config = dotdict(config, Loader=yaml.FullLoader)

        # Load SH Model.

        model = StripedHyena(global_config)
        model.load_state_dict(state_dict, strict=True)
        model.to_bfloat16_except_poles_residues()
        if device is not None:
            model = model.to(device)

        return model
