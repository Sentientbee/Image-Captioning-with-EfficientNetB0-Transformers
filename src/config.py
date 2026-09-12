from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import yaml


@dataclass
class ModelConfig:
    image_size: Tuple[int, int] = (299, 299)
    channels: int = 3
    seq_length: int = 25
    vocab_size: int = 10000
    embed_dim: int = 512
    ff_dim: int = 512
    num_heads_encoder: int = 2
    num_heads_decoder: int = 3
    dropout_rate_attn: float = 0.1
    dropout_rate_dense: float = 0.3
    dropout_rate_out: float = 0.5
    cnn_trainable: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    warmup_steps_ratio: float = 0.0667
    learning_rate: float = 1e-4
    early_stopping_patience: int = 3
    num_captions_per_image: int = 5


@dataclass
class InferenceConfig:
    max_decoded_length: int = 24
    beam_width: int = 3
    length_penalty_alpha: float = 0.7
    temperature: float = 1.0


@dataclass
class PathsConfig:
    vocab_path: str = "data/vocab.json"
    weights_path: str = "weights/model_weights.h5"
    sample_output_dir: str = "outputs"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AppConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        model_dict = raw_cfg.get("model", {})
        if "image_size" in model_dict and isinstance(model_dict["image_size"], list):
            model_dict["image_size"] = tuple(model_dict["image_size"])

        return cls(
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**raw_cfg.get("training", {})),
            inference=InferenceConfig(**raw_cfg.get("inference", {})),
            paths=PathsConfig(**raw_cfg.get("paths", {})),
        )


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from a YAML file or return defaults if not specified."""
    if config_path is None:
        default_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"
        if default_path.exists():
            return AppConfig.from_yaml(default_path)
        return AppConfig()
    return AppConfig.from_yaml(config_path)
