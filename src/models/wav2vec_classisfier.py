import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Dict, Any

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base-960h",
                 num_classes: int = 2,
                 freeze_feature_extractor: bool = True,
                 freeze_transformer_layers: int = 0):
        """
        wav2vec 2.0 based classifier for spoofing detection
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes (2 for bonafide/spoof)
            freeze_feature_extractor: Whether to freeze feature extractor
            freeze_transformer_layers: Number of transformer layers to freeze
        """
        super().__init__()
        
        # Load pretrained wav2vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Freeze transformer layers if specified
        if freeze_transformer_layers > 0:
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < freeze_transformer_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Classification head
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_values: Audio waveform tensor [batch_size, sequence_length]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Extract features using wav2vec2
        outputs = self.wav2vec2(input_values)
        
        # Pool features (mean pooling over time dimension)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def freeze_wav2vec_layers(self, num_layers: int):
        """Freeze specified number of wav2vec2 layers"""
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_wav2vec_layers(self):
        """Unfreeze all wav2vec2 layers"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
