import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model parameters
    model_name: str = "facebook/wav2vec2-base-960h"
    num_classes: int = 2
    freeze_feature_extractor: bool = True
    freeze_transformer_layers: int = 6
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-4
    
    # Data parameters
    max_audio_length: int = 64000  # 4 seconds at 16kHz
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Paths
    data_dir: str = "data/raw/ASVspoof2019_LA_train/flac"
    protocol_file: str = "data/protocols/ASVspoof2019.LA.cm.train.trn.txt"
    model_save_dir: str = "models"
    log_dir: str = "logs"
    
    # Training settings
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "asvspoof2019-wav2vec2"
    experiment_name: Optional[str] = None
